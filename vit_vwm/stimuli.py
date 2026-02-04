# -*- coding: utf-8 -*-
"""Stimulus generation for VWM experiments."""

import torch
import numpy as np
from torchvision import transforms

from .config import NORM_MEAN, NORM_STD, IMG_SIZE, PATCH_SIZE, RADIUS
from .color_utils import get_lab_color


transform_pipeline = transforms.Compose([
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])


# --- SPATIAL MATH ---

def get_circular_coords(img_size, radius, num_slots=8):
    center = img_size // 2
    coords = []
    angles = np.linspace(0, 2*np.pi, num_slots, endpoint=False)
    for theta in angles:
        cx = center + radius * np.cos(theta)
        cy = center + radius * np.sin(theta)
        tx = int(cx - PATCH_SIZE // 2)
        ty = int(cy - PATCH_SIZE // 2)
        coords.append((tx, ty))
    return coords

SLOT_COORDS = get_circular_coords(IMG_SIZE, RADIUS)


# --- GENERATOR ---

def generate_trial(set_size=4):
    # 1. Background (Gray)
    memory_img = torch.ones((3, IMG_SIZE, IMG_SIZE)) * 0.5
    probe_img = torch.ones((3, IMG_SIZE, IMG_SIZE)) * 0.5

    # 2. Locations
    if set_size > 8: raise ValueError("Max set size 8")
    chosen_indices = np.random.choice(8, set_size, replace=False)

    # 3. COLORS (Modified)
    # Instead of continuous sampling, we pick from 180 discrete bins.
    # We sample INDICES (0 to 179) without replacement.
    color_indices = np.random.choice(180, set_size, replace=False)

    # Convert indices to Radians
    # 180 bins means each step is 2 degrees (2pi / 180)
    chosen_angles = color_indices * (2 * np.pi / 180)

    items = []
    # Enumerate so we can access the pre-selected angles
    for i, slot_idx in enumerate(chosen_indices):
        tx, ty = SLOT_COORDS[slot_idx]

        angle = chosen_angles[i]
        color = get_lab_color(angle)

        # Draw Memory Item
        memory_img[:, ty:ty+PATCH_SIZE, tx:tx+PATCH_SIZE] = color.view(3, 1, 1)

        items.append({'slot': slot_idx, 'color': color, 'angle': angle, 'rect': (tx, ty)})

    # 4. Probe
    target_item = np.random.choice(items)
    tx, ty = target_item['rect']
    probe_img[:, ty:ty+PATCH_SIZE, tx:tx+PATCH_SIZE] = 1.0 # White Box

    # 5. Normalize
    memory_img = transform_pipeline(memory_img)
    probe_img = transform_pipeline(probe_img)

    return {
        'memory_image': memory_img,
        'probe_image': probe_img,
        'target_angle': target_item['angle'],
        'target_rgb': target_item['color'].numpy(),
        'set_size': set_size
    }

def get_batch(batch_size, set_size=None, min_set=1, max_set=8):
    batch_list = []
    for _ in range(batch_size):
        k = set_size if set_size is not None else np.random.randint(min_set, max_set + 1)
        batch_list.append(generate_trial(k))

    mem = torch.stack([b['memory_image'] for b in batch_list])
    prb = torch.stack([b['probe_image'] for b in batch_list])
    tgt = torch.tensor([b['target_angle'] for b in batch_list], dtype=torch.float32)

    return mem, prb, tgt, batch_list
