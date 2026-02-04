# -*- coding: utf-8 -*-
"""Visualization functions for VWM experiments."""

import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from torchvision import transforms

from .config import NORM_MEAN, NORM_STD
from .stimuli import get_batch
from .analysis import von_mises_pdf


# Quick un-norm for viewing
def unnormalize(tensor_img):
    inv_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(NORM_MEAN, NORM_STD)],
        std=[1/s for s in NORM_STD]
    )
    img = inv_normalize(tensor_img)
    return torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()


def plot_mixture_distributions(data_dict, df_params):
    set_sizes = sorted(df_params['SetSize'].unique())
    num_plots = len(set_sizes)
    n_cols = math.ceil(num_plots / 2)

    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 7), sharey=True, sharex=True)
    axes = axes.flatten()

    # 1. Create X-range in RADIANS for math...
    x_rad = np.linspace(-np.pi, np.pi, 200)
    # ...but convert to DEGREES for plotting the line
    x_deg = np.degrees(x_rad)

    for i, ss in enumerate(set_sizes):
        ax = axes[i]

        # A. Get Data (Assumed to be in Radians)
        if isinstance(data_dict, dict) and 'error_rad' in data_dict:
             mask = data_dict['set_size'] == ss
             errors_rad = data_dict['error_rad'][mask]
        else:
             errors_rad = data_dict[ss]

        # CONVERT DATA TO DEGREES HERE
        errors_deg = np.degrees(errors_rad)

        # B. Get Params
        row = df_params[df_params['SetSize'] == ss].iloc[0]
        pm = row['Pm'] if 'Pm' in row else row['pm']
        kappa = row['kappa']
        sd_val = row['sd']
        g_rate = 1.0 - pm

        # C. Plot Histogram (Now in Degrees)
        # Note: 'density=True' will now normalize over 360 units, so height will look lower
        # than radian plots. This is normal.
        ax.hist(errors_deg, bins=45, range=(-180, 180), density=True,
                color='gray', alpha=0.5, edgecolor='white')

        # D. Plot Fitted Curve
        # Math still happens in RADIANS:
        vm_part = von_mises_pdf(x_rad, 0, kappa)
        uni_part = 1 / (2 * np.pi)

        pdf_rad = (pm * vm_part) + (g_rate * uni_part)

        # KEY STEP: Convert PDF density from "per radian" to "per degree"
        # Since 1 radian = 57.3 degrees, the density height must shrink.
        pdf_deg = pdf_rad * (np.pi / 180)

        # Plot degrees on X, density_per_degree on Y
        ax.plot(x_deg, pdf_deg, color='red', linewidth=3)

        # E. Annotations
        ax.text(0.05, 0.95, f"$P_m = {pm:.2f}$", transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top', ha='left')
        ax.text(0.95, 0.95, rf"$SD = {sd_val:.1f}^\circ$", transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top', ha='right')

        # Formatting
        ax.set_title(f'Set Size {ss}', fontsize=12, fontweight='bold')
        ax.set_xlim([-180, 180])           # Set limits to degrees
        ax.set_xticks([-180, -90, 0, 90, 180]) # Nice tick marks
        ax.set_xticklabels([r'$-180^\circ$', r'$-90^\circ$', r'$0^\circ$', r'$90^\circ$', r'$180^\circ$'])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.text(0.5, 0.02, 'Error (degrees)', ha='center', fontsize=12)
    fig.text(0.02, 0.5, 'Probability Density', va='center', rotation='vertical', fontsize=12)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0.03, 0.03, 1, 1])
    plt.show()


def plot_zhang_luck_results(df_params):
    """
    Replicates the visual style of Zhang & Luck (2008), Figure 2.
    Panel A: Pm (Probability of Memory) vs Set Size
    Panel B: SD (Precision) vs Set Size

    Assumes df_params contains columns: 'SetSize', 'pm' (or 'Pm'), 'sd', 'kappa'
    """

    # Handle column naming variations (Pm vs pm)
    pm_col = 'Pm' if 'Pm' in df_params.columns else 'pm'

    # Setup Figure: 2 subplots side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Extract data
    x = df_params['SetSize']
    pm = df_params[pm_col]
    sd = df_params['sd']

    # If you wanted to plot Kappa instead, you could uncomment this:
    # kappa = df_params['kappa']

    # --- PANEL A: Probability of Memory (Pm) ---
    ax = axes[0]
    ax.plot(x, pm, 'o-', color='black', linewidth=2, markersize=8,
            markerfacecolor='white', markeredgewidth=2, label='Model')

    # Add Theoretical Capacity Limits (K=3 and K=4)
    # This helps you benchmark if the model is "Human-like"
    k_human = 3
    theoretical_pm = [min(1.0, k_human/n) for n in x]
    ax.plot(x, theoretical_pm, '--', color='gray', alpha=0.5, label='Capacity K=3')

    # Formatting A
    ax.set_title(r'$\mathbf{a}$  Probability of Memory ($P_m$)', loc='left', fontsize=14)
    ax.set_xlabel('Set Size', fontsize=12)
    ax.set_ylabel('Probability ($P_m$)', fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(x)
    ax.legend(loc='lower left', frameon=False, fontsize=10)

    # Annotate values
    for i, val in enumerate(pm):
        ax.annotate(f"{val:.2f}", (x.iloc[i], pm.iloc[i]),
                    xytext=(0, 8), textcoords='offset points', ha='center', fontsize=9)


    # --- PANEL B: Precision (Standard Deviation) ---
    ax = axes[1]
    ax.plot(x, sd, 'o-', color='black', linewidth=2, markersize=8,
            markerfacecolor='white', markeredgewidth=2)

    # Formatting B
    ax.set_title(r'$\mathbf{b}$  Precision (SD)', loc='left', fontsize=14)
    ax.set_xlabel('Set Size', fontsize=12)
    ax.set_ylabel('Standard Deviation (deg)', fontsize=12)
    ax.set_xticks(x)

    # Dynamic Y-Axis:
    # If model is failing (SD > 45), zoom out. If working well, zoom in.
    max_sd = sd.max()
    if max_sd > 50:
        ax.set_ylim(0, 120)  # Random chance view
        ax.axhline(y=75, color='red', linestyle=':', alpha=0.3, label='Random Chance')
    else:
        ax.set_ylim(0, 45)  # Human-scale view

    # Annotate values
    for i, val in enumerate(sd):
        ax.annotate(f"{val:.1f}°", (x.iloc[i], sd.iloc[i]),
                    xytext=(0, -15), textcoords='offset points', ha='center', fontsize=9)

    # --- FINAL CLEANUP ---
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_attention(model, device, target_set_size=2):
    model.eval()

    # 1. Generate a single trial with the desired set size
    mem_tensor, probe_tensor, _, _ = get_batch(batch_size=1, set_size=target_set_size)

    mem_t = mem_tensor.to(device)
    probe_t = probe_tensor.to(device)

    with torch.no_grad():
        # --- MANUAL FORWARD PASS ---
        # 1. Get Memory Features (Shape: [1, 197, 768])
        mem_patches = model.encoder(mem_t)

        # 2. Get Probe Query
        probe_map = model.encoder(probe_t)
        probe_feat, _ = probe_map.max(dim=1)
        query = model.probe_adapter(probe_feat).unsqueeze(1)

        # 3. Calculate Attention (Shape: [1, 1, 197])
        attn_logits = torch.bmm(query, mem_patches.transpose(1, 2)) * model.attn_scale
        attn_weights = torch.softmax(attn_logits, dim=-1)

        # --- THE FIX ---
        # Exclude the first token (CLS token) which is index 0
        # We only want the 196 spatial patches (Indices 1 to 197)
        spatial_attn = attn_weights[:, :, 1:] # Shape: [1, 1, 196]

        # Now we can reshape 196 -> 14x14
        attn_grid = spatial_attn.squeeze().view(14, 14).cpu().numpy()

    # --- PLOT ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Helper to show images
    def show_img(tensor):
        img = tensor.squeeze().cpu().permute(1, 2, 0).numpy()
        # Denormalize roughly for visibility if needed, or just clip
        img = (img - img.min()) / (img.max() - img.min())
        return img

    axes[0].imshow(show_img(mem_t))
    axes[0].set_title(f"Memory (Set Size {target_set_size})")
    axes[0].axis('off')

    axes[1].imshow(show_img(probe_t))
    axes[1].set_title("Probe (Location Query)")
    axes[1].axis('off')

    # Overlay Attention
    axes[2].imshow(show_img(mem_t), alpha=0.5)
    # Extent ensures the 14x14 grid stretches to cover the 224x224 image
    im = axes[2].imshow(attn_grid, cmap='jet', alpha=0.6, extent=[0, 224, 224, 0])
    axes[2].set_title("Model Attention (Spatial Only)")
    axes[2].axis('off')

    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    plt.show()
