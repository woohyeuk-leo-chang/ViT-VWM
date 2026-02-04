# -*- coding: utf-8 -*-
"""Loss functions and metrics for VWM experiments."""

import torch
import numpy as np


def circular_loss(prediction, target):
    """
    Loss = 1 - cos(pred_angle - target_angle)
    Prediction: [sin, cos] vector (unnormalized from model)
    Target: Scalar angle in radians
    """
    # Normalize prediction to unit vector
    pred_norm = torch.nn.functional.normalize(prediction, p=2, dim=1)

    # Create target vector [sin, cos]
    target_vec = torch.stack([torch.sin(target), torch.cos(target)], dim=1)

    # Cosine similarity
    cosine_sim = torch.sum(pred_norm * target_vec, dim=1)

    # Loss: 0 if identical, 2 if opposite
    return (1 - cosine_sim).mean()

def degrees_error(prediction, target):
    """Returns absolute error in degrees [0, 180]"""
    # Convert [sin, cos] -> radians
    pred_angle = torch.atan2(prediction[:, 0], prediction[:, 1])

    # Difference
    diff = pred_angle - target

    # Wrap to [-pi, pi]
    diff = (diff + np.pi) % (2 * np.pi) - np.pi

    # Convert to degrees abs
    return torch.abs(diff) * (180 / np.pi)
