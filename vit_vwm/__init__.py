# -*- coding: utf-8 -*-
"""
VWM - Visual Working Memory Experiments with Vision Transformers

A package for investigating computational constraints on capacity, binding,
and maintenance in visual working memory using Vision Transformer architectures.
"""

from .config import (
    NORM_MEAN,
    NORM_STD,
    IMG_SIZE,
    PATCH_SIZE,
    RADIUS,
    LAB_L,
    LAB_A_CENTER,
    LAB_B_CENTER,
    LAB_RADIUS,
    XN,
    YN,
    ZN,
    set_seed,
    get_device,
)

from .color_utils import (
    lab_to_xyz,
    xyz_to_rgb,
    get_lab_color,
)

from .stimuli import (
    transform_pipeline,
    get_circular_coords,
    SLOT_COORDS,
    generate_trial,
    get_batch,
)

from .dataset import (
    VWMDataLoader,
    create_dataloader,
)

from .models import (
    ViT_attention_bottleneck,
    ViT_control,
)

from .losses import (
    circular_loss,
    degrees_error,
)

from .training import (
    train_model,
    collect_model_data,
)

from .analysis import (
    von_mises_pdf,
    mixture_nll,
    kappa_to_sd_deg,
    get_zhang_luck_params,
)

from .visualization import (
    unnormalize,
    plot_mixture_distributions,
    plot_zhang_luck_results,
    visualize_attention,
)

__version__ = "0.1.0"
__author__ = "Woohyeuk 'Leo' Chang"

__all__ = [
    # config
    "NORM_MEAN",
    "NORM_STD",
    "IMG_SIZE",
    "PATCH_SIZE",
    "RADIUS",
    "LAB_L",
    "LAB_A_CENTER",
    "LAB_B_CENTER",
    "LAB_RADIUS",
    "XN",
    "YN",
    "ZN",
    "set_seed",
    "get_device",
    # color_utils
    "lab_to_xyz",
    "xyz_to_rgb",
    "get_lab_color",
    # stimuli
    "transform_pipeline",
    "get_circular_coords",
    "SLOT_COORDS",
    "generate_trial",
    "get_batch",
    # dataset
    "VWMDataLoader",
    "create_dataloader",
    # models
    "ViT_attention_bottleneck",
    "ViT_control",
    # losses
    "circular_loss",
    "degrees_error",
    # training
    "train_model",
    "collect_model_data",
    # analysis
    "von_mises_pdf",
    "mixture_nll",
    "kappa_to_sd_deg",
    "get_zhang_luck_params",
    # visualization
    "unnormalize",
    "plot_mixture_distributions",
    "plot_zhang_luck_results",
    "visualize_attention",
]
