"""Data loading utilities for VWM experiments.

Provides VWMDataLoader with dual interface:
- PyTorch Dataset protocol (__len__, __getitem__) for DataLoader compatibility
- Iterator protocol (__iter__, __next__) for simple step-based training
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from .stimuli import generate_trial, get_batch


class VWMDataLoader(Dataset):
    """
    Flexible data loader for VWM stimulus generation.
    
    Supports two usage patterns:
    
    1. Direct iteration (simple, no worker overhead):
        ```
        loader = VWMDataLoader(batch_size=256)
        for step in range(total_steps):
            mem, prb, tgt = next(loader)
        ```
    
    2. PyTorch DataLoader (multi-worker parallel loading):
        ```
        dataset = VWMDataLoader(batch_size=1, length=50000)
        dataloader = DataLoader(dataset, batch_size=256, num_workers=4)
        for mem, prb, tgt, set_sizes in dataloader:
            ...
        ```
    
    Note: Since data is generated synthetically, __getitem__ ignores the index
    and generates a fresh random trial each call. This means the "dataset" is
    effectively infinite, and `length` is just a hint for DataLoader iteration.
    """
    
    def __init__(self, batch_size=1, min_set=1, max_set=8, fixed_set_size=None,
                 length=10000):
        """
        Args:
            batch_size: Samples per batch when using direct iteration.
                        Set to 1 when using with PyTorch DataLoader (it handles batching).
            min_set: Minimum set size (ignored if fixed_set_size is set)
            max_set: Maximum set size (ignored if fixed_set_size is set)
            fixed_set_size: If provided, use this set size for all trials
            length: Nominal dataset length for PyTorch DataLoader compatibility.
                    Only affects how many samples DataLoader yields per "epoch".
        """
        self.batch_size = batch_size
        self.min_set = min_set
        self.max_set = max_set
        self.fixed_set_size = fixed_set_size
        self._length = length
    
    # =========================================================================
    # PyTorch Dataset Interface (for use with DataLoader)
    # =========================================================================
    
    def __len__(self):
        """Nominal length for DataLoader. Synthetic data is effectively infinite."""
        return self._length
    
    def __getitem__(self, idx):
        """
        Generate a single trial. Index is ignored (each call = fresh random trial).
        
        Returns:
            tuple: (memory_image, probe_image, target_angle, set_size)
        """
        # Determine set size
        if self.fixed_set_size is not None:
            k = self.fixed_set_size
        else:
            k = np.random.randint(self.min_set, self.max_set + 1)
        
        trial = generate_trial(k)
        
        return (
            trial['memory_image'],
            trial['probe_image'],
            torch.tensor(trial['target_angle'], dtype=torch.float32),
            k
        )
    
    # =========================================================================
    # Iterator Interface (for direct step-based training)
    # =========================================================================
    
    def __iter__(self):
        """Enable direct iteration: `for batch in loader` or `next(loader)`"""
        return self
    
    def __next__(self):
        """
        Generate a batch of trials.
        
        Returns:
            tuple: (memory_images, probe_images, target_angles) as batched tensors
        """
        mem, prb, tgt, _ = get_batch(
            self.batch_size,
            set_size=self.fixed_set_size,
            min_set=self.min_set,
            max_set=self.max_set
        )
        return mem, prb, tgt


def create_dataloader(batch_size, min_set=1, max_set=8, fixed_set_size=None,
                      samples_per_epoch=10000, num_workers=4):
    """
    Factory function to create a PyTorch DataLoader with parallel workers.
    
    Use this when stimulus generation becomes a bottleneck and you want
    to parallelize across CPU cores.
    
    Args:
        batch_size: Samples per batch
        min_set: Minimum set size
        max_set: Maximum set size
        fixed_set_size: If set, use this set size for all trials
        samples_per_epoch: How many samples before DataLoader "ends"
        num_workers: Number of parallel worker processes
    
    Returns:
        DataLoader instance
    """
    dataset = VWMDataLoader(
        batch_size=1,  # DataLoader handles batching
        min_set=min_set,
        max_set=max_set,
        fixed_set_size=fixed_set_size,
        length=samples_per_epoch
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True
    )