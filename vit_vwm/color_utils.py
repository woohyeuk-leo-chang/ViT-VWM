# -*- coding: utf-8 -*-
"""Color space conversion utilities for VWM stimulus generation."""

import torch
import numpy as np

from .config import LAB_L, LAB_A_CENTER, LAB_B_CENTER, LAB_RADIUS, XN, YN, ZN


# --- COLOR SPACE MATH ---

def lab_to_xyz(L, a, b):
    """
    Converts CIE L*a*b* to CIE XYZ (D65 White Point).
    Standard formulas used in image processing.
    """
    fy = (L + 16.0) / 116.0
    fx = (a / 500.0) + fy
    fz = fy - (b / 200.0)

    def inverse_f(t):
        return t**3 if t > 0.206893034 else (t - 16.0/116.0) / 7.787

    # X, Y, Z
    x = XN * inverse_f(fx)
    y = YN * inverse_f(fy)
    z = ZN * inverse_f(fz)
    return x, y, z

def xyz_to_rgb(x, y, z):
    """
    Converts CIE XYZ to sRGB (Linear -> Gamma Corrected).
    """
    # Normalize to 0-1 range for matrix math
    x, y, z = x / 100.0, y / 100.0, z / 100.0

    # XYZ to Linear RGB (Standard sRGB Matrix)
    r_lin = x * 3.2406 + y * -1.5372 + z * -0.4986
    g_lin = x * -0.9689 + y * 1.8758 + z * 0.0415
    b_lin = x * 0.0557 + y * -0.2040 + z * 1.0570

    # Linear RGB to Gamma Corrected sRGB
    def gamma_correct(c):
        return 1.055 * (c ** (1/2.4)) - 0.055 if c > 0.0031308 else 12.92 * c

    r = gamma_correct(r_lin)
    g = gamma_correct(g_lin)
    b = gamma_correct(b_lin)

    # Clamp to strictly [0, 1]
    return min(max(r, 0.0), 1.0), min(max(g, 0.0), 1.0), min(max(b, 0.0), 1.0)

def get_lab_color(angle_rad):
    """
    Generates an RGB color by sampling from the specific L*a*b* circle.

    Circle:
    - Center: (a=20, b=38)
    - Radius: 60
    - Luminance: 70 (Fixed)
    """
    # Calculate a and b on the circle
    # Note: angle_rad determines the hue along this specific color path
    a = LAB_A_CENTER + LAB_RADIUS * np.cos(angle_rad)
    b = LAB_B_CENTER + LAB_RADIUS * np.sin(angle_rad)

    # Convert to RGB
    x, y, z = lab_to_xyz(LAB_L, a, b)
    r, g, b_val = xyz_to_rgb(x, y, z)

    return torch.tensor([r, g, b_val], dtype=torch.float32)
