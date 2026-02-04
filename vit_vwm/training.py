"""Training loop and evaluation utilities for VWM experiments.

Design Philosophy (v2):
- total_steps is the primary control variable (not epochs × steps_per_epoch)
- Validation frequency is configurable (every N steps)
- Uses VWMDataLoader for simple synthetic data (no worker overhead)
- Progress bar shows actual training progress

For a spatial cueing task with pretrained ViT (only fine-tuning last block),
50k-200k samples should be sufficient.

Recommended settings:
    Quick test:       total_steps=100,  batch_size=256  (~25k samples, ~1 min)
    Development:      total_steps=300,  batch_size=256  (~75k samples, ~3 min)
    Standard:         total_steps=500,  batch_size=256  (~128k samples, ~5 min)
    Full training:    total_steps=1000, batch_size=256  (~256k samples, ~10 min)
"""

import torch
import torch.optim as optim
import numpy as np
import copy
from tqdm.auto import tqdm

from .stimuli import get_batch
from .dataset import VWMDataLoader
from .losses import circular_loss, degrees_error


def train_model(
    model,
    total_steps: int = 500,
    batch_size: int = 256,
    eval_every: int = 50,
    patience: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    device: str = 'cuda',
    verbose: bool = True
):
    """
    Train VWM model with step-based control (HuggingFace style).
    
    Args:
        model: PyTorch model to train
        total_steps: Total gradient update steps (primary control variable)
        batch_size: Samples per gradient step
        eval_every: Validate every N steps
        patience: Early stopping patience (in eval cycles, not steps)
        lr: Learning rate
        weight_decay: AdamW weight decay
        device: 'cuda', 'mps', or 'cpu'
        verbose: Print progress
    
    Returns:
        model: Trained model (with best weights loaded)
        history: List of dicts with validation metrics
    
    Example:
        # Quick test (~25k samples, ~1 min on GPU)
        model, history = train_model(model, total_steps=100, batch_size=256)
        
        # Standard training (~128k samples, ~5 min on GPU)
        model, history = train_model(model, total_steps=500, batch_size=256)
    """
    model = model.to(device)
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Data iterator (no worker overhead)
    train_iter = VWMDataLoader(batch_size, min_set=1, max_set=8)
    
    # Fixed validation sets (generated once for consistency)
    if verbose:
        print("Generating fixed validation sets...")
    val_ss1 = tuple(t.to(device) for t in get_batch(256, set_size=1)[:3])
    val_mix = tuple(t.to(device) for t in get_batch(256, min_set=1, max_set=8)[:3])
    
    # Tracking
    history = []
    best_err = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0
    running_loss = 0.0
    
    if verbose:
        total_samples = total_steps * batch_size
        print(f"Training: {total_steps} steps × {batch_size} batch = {total_samples:,} samples")
        print(f"Eval every {eval_every} steps, patience={patience}")
    
    # Progress bar over steps
    pbar = tqdm(range(total_steps), desc="Training", disable=not verbose)
    
    for step in pbar:
        model.train()
        
        # Get batch
        mem, prb, tgt = next(train_iter)
        mem, prb, tgt = mem.to(device), prb.to(device), tgt.to(device)
        
        # Forward + backward
        optimizer.zero_grad()
        preds = model(mem, prb)
        loss = circular_loss(preds, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        
        # Validation
        if (step + 1) % eval_every == 0:
            avg_loss = running_loss / eval_every
            running_loss = 0.0
            
            model.eval()
            with torch.no_grad():
                # Set size 1 (sanity check - should approach ~5° or better)
                out_ss1 = model(val_ss1[0], val_ss1[1])
                err_ss1 = degrees_error(out_ss1, val_ss1[2]).mean().item()
                
                # Mixed set sizes (main metric)
                out_mix = model(val_mix[0], val_mix[1])
                err_mix = degrees_error(out_mix, val_mix[2]).mean().item()
            
            history.append({
                'step': step + 1,
                'samples': (step + 1) * batch_size,
                'loss': avg_loss,
                'err_ss1': err_ss1,
                'err_mix': err_mix,
                'lr': optimizer.param_groups[0]['lr']
            })
            
            # Checkpointing (using mixed error as primary metric)
            if err_mix < best_err:
                best_err = err_mix
                best_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            pbar.set_postfix({
                'loss': f'{avg_loss:.3f}',
                'err_mix': f'{err_mix:.1f}°',
                'err_ss1': f'{err_ss1:.1f}°',
                'best': f'{best_err:.1f}°'
            })
            
            # Early stopping
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stop at step {step+1}: no improvement for {patience} evals")
                break
    
    # Load best weights
    model.load_state_dict(best_weights)
    if verbose:
        print(f"Training complete. Best validation error: {best_err:.1f}°")
    
    return model, history


def collect_model_data(model, num_trials=2000, batch_size=64, device='cuda'):
    """
    Collect trial-by-trial data for mixture modeling analysis.
    
    Args:
        model: Trained model
        num_trials: Total trials to collect
        batch_size: Batch size for inference
        device: Device for inference
    
    Returns:
        dict with keys:
            - set_size: array of set sizes
            - target_rad: array of target angles (radians)
            - response_rad: array of model responses (radians)
            - error_rad: array of signed errors (radians, wrapped to [-π, π])
            - error_deg: array of signed errors (degrees)
    """
    model.eval()
    model.to(device)

    all_set_sizes = []
    all_targets = []
    all_responses = []
    all_errors_rad = []
    all_errors_deg = []

    print(f"Collecting data from {num_trials} trials...")
    batches = (num_trials + batch_size - 1) // batch_size

    with torch.no_grad():
        for _ in tqdm(range(batches), desc="Collecting"):
            m, p, t, meta = get_batch(batch_size, min_set=1, max_set=8)
            m, p, t = m.to(device), p.to(device), t.to(device)

            # Model prediction: [sin(θ), cos(θ)] -> angle
            preds_vec = model(m, p)
            preds_rad = torch.atan2(preds_vec[:, 0], preds_vec[:, 1])

            # Error (wrapped to [-π, π])
            diff = preds_rad - t
            diff_wrapped = (diff + np.pi) % (2 * np.pi) - np.pi

            # Extract set sizes from metadata
            current_set_sizes = [trial['set_size'] for trial in meta]

            # Store
            all_set_sizes.extend(current_set_sizes)
            all_targets.extend(t.cpu().numpy())
            all_responses.extend(preds_rad.cpu().numpy())
            all_errors_rad.extend(diff_wrapped.cpu().numpy())
            all_errors_deg.extend(diff_wrapped.cpu().numpy() * (180 / np.pi))

    # Trim to exact count
    return {
        'set_size': np.array(all_set_sizes[:num_trials]),
        'target_rad': np.array(all_targets[:num_trials]),
        'response_rad': np.array(all_responses[:num_trials]),
        'error_rad': np.array(all_errors_rad[:num_trials]),
        'error_deg': np.array(all_errors_deg[:num_trials])
    }


def quick_benchmark(model, device='cuda', n_trials=500):
    """
    Quick performance check across all set sizes.
    
    Args:
        model: Trained model
        device: Device for inference
        n_trials: Trials per set size
    
    Returns:
        pandas DataFrame with columns: set_size, mean_err, std_err, n
    """
    import pandas as pd
    
    model.eval()
    model.to(device)
    
    results = []
    
    print(f"Benchmarking {n_trials} trials per set size...")
    with torch.no_grad():
        for ss in range(1, 9):
            mem, prb, tgt, _ = get_batch(n_trials, set_size=ss)
            mem, prb, tgt = mem.to(device), prb.to(device), tgt.to(device)
            
            preds = model(mem, prb)
            errs = degrees_error(preds, tgt)
            
            results.append({
                'set_size': ss,
                'mean_err': errs.mean().item(),
                'std_err': errs.std().item(),
                'n': n_trials
            })
    
    return pd.DataFrame(results)