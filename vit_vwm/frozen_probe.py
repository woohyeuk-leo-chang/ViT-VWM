# -*- coding: utf-8 -*-
"""Model psychophysics with frozen encoders.

Runs the continuous-report task on pretrained vision encoders WITHOUT
fine-tuning. The display contains the colored squares plus a simultaneous cue,
either a white frame around the cued item ('frame': cue and item share
patches) or a white marker placed radially outward from it ('marker': cue and
item fall in different patches, so selection needs binding across tokens).
Linear readouts decode the cued hue from the frozen tokens in two ways:

- 'slot' (oracle selection): pool the patch tokens covering the cued location.
  The readout is told where the target is, so this measures how faithfully
  the representation encodes each item, including contamination from the
  other items via self-attention.
- 'cls' / 'mean' (network selection): decode from the CLS token or the
  mean-pooled patch tokens. A linear readout cannot gate on the frame itself,
  so success requires the frozen network to have bound the cue to the item.
"""

import numpy as np
import timm
import torch
from sklearn.linear_model import RidgeCV
from tqdm.auto import tqdm

from .config import IMG_SIZE, PATCH_SIZE, RADIUS
from .color_utils import get_lab_color
from .stimuli import SLOT_COORDS


# name -> (timm model id, pretrained)
ENCODERS = {
    'supervised': ('vit_base_patch16_224.augreg2_in21k_ft_in1k', True),
    'clip': ('vit_base_patch16_clip_224.openai', True),
    'dinov2': ('vit_base_patch14_dinov2.lvd142m', True),
    'dinov2_reg': ('vit_base_patch14_reg4_dinov2.lvd142m', True),
    'mae': ('vit_base_patch16_224.mae', True),
    'random': ('vit_base_patch16_224.augreg2_in21k_ft_in1k', False),
}

N_COLORS = 180
COLOR_TABLE = torch.stack([get_lab_color(i * 2 * np.pi / N_COLORS) for i in range(N_COLORS)])


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


# --- STIMULI ---

CUE_GAP, CUE_WIDTH = 1, 2  # pixels between square and frame, frame thickness
MARKER_SIZE, MARKER_OFFSET = 8, 32  # marker side; extra radius beyond the items (pixels)


def draw_cue_frame(img, tx, ty):
    lo, hi = CUE_GAP + CUE_WIDTH, PATCH_SIZE + CUE_GAP + CUE_WIDTH
    frame = torch.zeros(IMG_SIZE, IMG_SIZE, dtype=torch.bool)
    frame[ty - lo:ty + hi, tx - lo:tx + hi] = True
    frame[ty - CUE_GAP:ty + PATCH_SIZE + CUE_GAP, tx - CUE_GAP:tx + PATCH_SIZE + CUE_GAP] = False
    img[:, frame] = 1.0


def draw_cue_marker(img, slot):
    theta = 2 * np.pi * slot / len(SLOT_COORDS)  # matches get_circular_coords
    center = IMG_SIZE // 2
    cx = center + (RADIUS + MARKER_OFFSET) * np.cos(theta)
    cy = center + (RADIUS + MARKER_OFFSET) * np.sin(theta)
    tx, ty = int(cx - MARKER_SIZE // 2), int(cy - MARKER_SIZE // 2)
    img[:, ty:ty + MARKER_SIZE, tx:tx + MARKER_SIZE] = 1.0


def make_trials(n_per_set, set_sizes=range(1, 9), seed=0, cue='marker'):
    """
    Generates displays in [0, 1] RGB (normalization is encoder-specific).
    Same task as stimuli.generate_trial, but keeps every item's color and slot
    so swap errors can be analyzed, and draws the cue into the display.
    """
    rng = np.random.default_rng(seed)
    n_slots = len(SLOT_COORDS)
    trials = {'set_size': [], 'target_slot': [], 'target_rad': [], 'nontarget_rad': []}
    images = []

    for ss in set_sizes:
        for _ in range(n_per_set):
            img = torch.full((3, IMG_SIZE, IMG_SIZE), 0.5)
            slots = rng.choice(n_slots, ss, replace=False)
            color_idx = rng.choice(N_COLORS, ss, replace=False)
            for slot, ci in zip(slots, color_idx):
                tx, ty = SLOT_COORDS[slot]
                img[:, ty:ty + PATCH_SIZE, tx:tx + PATCH_SIZE] = COLOR_TABLE[ci].view(3, 1, 1)
            angles = color_idx * 2 * np.pi / N_COLORS

            # Target is item 0 (slots are already a random draw)
            if cue == 'frame':
                draw_cue_frame(img, *SLOT_COORDS[slots[0]])
            elif cue == 'marker':
                draw_cue_marker(img, slots[0])
            nt = np.full(n_slots - 1, np.nan)
            nt[:ss - 1] = angles[1:]
            trials['set_size'].append(ss)
            trials['target_slot'].append(slots[0])
            trials['target_rad'].append(angles[0])
            trials['nontarget_rad'].append(nt)
            images.append(img)

    trials = {k: np.array(v) for k, v in trials.items()}
    return torch.stack(images), trials


# --- ENCODERS ---

def load_encoder(name, device):
    model_id, pretrained = ENCODERS[name]
    kwargs = {'img_size': IMG_SIZE} if 'dinov2' in model_id else {}
    model = timm.create_model(model_id, pretrained=pretrained, num_classes=0, **kwargs)
    model.eval().to(device)
    cfg = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(cfg['mean']).view(1, 3, 1, 1)
    std = torch.tensor(cfg['std']).view(1, 3, 1, 1)
    return model, mean, std


def slot_patch_weights(patch_size, grid):
    """[n_slots, grid*grid] weights: fraction of each square covered by each patch."""
    weights = np.zeros((len(SLOT_COORDS), grid * grid))
    for s, (tx, ty) in enumerate(SLOT_COORDS):
        mask = np.zeros((IMG_SIZE, IMG_SIZE))
        mask[ty:ty + PATCH_SIZE, tx:tx + PATCH_SIZE] = 1
        cover = mask[:grid * patch_size, :grid * patch_size].reshape(
            grid, patch_size, grid, patch_size).sum(axis=(1, 3))
        weights[s] = cover.ravel() / cover.sum()
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def extract_features(model, mean, std, images, slots, device, batch_size=64):
    """
    Runs the frozen encoder and returns, for the output of every block:
        'slot': [n_layers, n_trials, dim] patches covering the cued slot, pooled
        'cls':  [n_layers, n_trials, dim] CLS token
        'mean': [n_layers, n_trials, dim] mean over patch tokens
    plus each trial's max patch-token norm at the last block (register-artifact check).
    """
    patch = model.patch_embed.patch_size[0]
    grid = IMG_SIZE // patch
    n_prefix = model.num_prefix_tokens
    w_slots = slot_patch_weights(patch, grid).to(device)

    captured = []
    hooks = [blk.register_forward_hook(lambda m, i, o: captured.append(o))
             for blk in model.blocks]

    feats = {'slot': [], 'cls': [], 'mean': []}
    norms = []
    for start in tqdm(range(0, len(images), batch_size), desc="Encoding", leave=False):
        x = ((images[start:start + batch_size] - mean) / std).to(device)
        w = w_slots[torch.as_tensor(slots[start:start + batch_size], device=device)]
        captured.clear()
        model(x)
        batch = {'slot': [], 'cls': [], 'mean': []}
        for out in captured:
            patches = out[:, n_prefix:]                       # [b, grid*grid, d]
            batch['slot'].append(torch.einsum('bp,bpd->bd', w, patches).cpu())
            batch['cls'].append(out[:, 0].cpu())
            batch['mean'].append(patches.mean(dim=1).cpu())
        for k in feats:
            feats[k].append(torch.stack(batch[k]))
        norms.append(captured[-1][:, n_prefix:].norm(dim=-1).max(dim=1).values.cpu())

    for h in hooks:
        h.remove()
    return {k: torch.cat(v, dim=1).numpy() for k, v in feats.items()}, torch.cat(norms).numpy()


# --- READOUT ---

def fit_readout(X_train, y_train_rad):
    """Ridge regression from features to [cos, sin] of hue, alpha by CV."""
    mu, sd = X_train.mean(0), X_train.std(0) + 1e-6
    target = np.stack([np.cos(y_train_rad), np.sin(y_train_rad)], axis=1)
    ridge = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit((X_train - mu) / sd, target)
    return lambda X: ridge.predict((X - mu) / sd)


def decode(readout, X):
    pred = readout(X)
    return np.arctan2(pred[:, 1], pred[:, 0])


def to_data_dict(trials, response_rad):
    """Formats responses for analysis.get_zhang_luck_params / get_swap_params."""
    return {
        'set_size': trials['set_size'],
        'target_rad': trials['target_rad'],
        'response_rad': response_rad,
        'error_rad': wrap(response_rad - trials['target_rad']),
        'error_deg': np.degrees(wrap(response_rad - trials['target_rad'])),
        'nontarget_rel_rad': wrap(response_rad[:, None] - trials['nontarget_rad']),
    }
