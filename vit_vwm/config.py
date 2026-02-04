# -*- coding: utf-8 -*-
"""Configuration constants and seed utilities for VWM experiments."""

import torch
import numpy as np
import random

# --- CONFIGURATION ---
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224
PATCH_SIZE = 16
RADIUS = 75      # Spatial radius (pixels)

# L*a*b* Circle Constants (From your description)
LAB_L = 70.0
LAB_A_CENTER = 20.0
LAB_B_CENTER = 38.0
LAB_RADIUS = 60.0

# Pre-calculated standard D65 White Point for conversion
XN, YN, ZN = 95.047, 100.000, 108.883


# Set seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """
    Detect the best available device for PyTorch.
    Priority: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
