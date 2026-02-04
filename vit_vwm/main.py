# -*- coding: utf-8 -*-
"""Main entry point for VWM experiments."""

import torch
import pickle
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from .config import set_seed, get_device
from .stimuli import get_batch
from .models import ViT_attention_bottleneck
from .training import train_model, collect_model_data
from .analysis import get_zhang_luck_params
from .visualization import (
    unnormalize,
    plot_mixture_distributions,
    plot_zhang_luck_results,
    visualize_attention
)


def main():
    # Set seed and device
    set_seed(42)
    device = get_device()
    print(f"Running on device: {device}")

    # --- Sample Display ---
    # Generate a "Full Load" batch (Set Size 8) to see all positions
    m_batch, p_batch, _, _ = get_batch(1, set_size=8)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(unnormalize(m_batch[0]))
    plt.title("Memory (Set Size 8)\nNote the circular layout")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(unnormalize(p_batch[0]))
    plt.title("Probe (Single Location)")
    plt.axis('off')

    plt.show()

    # --- Train Model ---
    model = ViT_attention_bottleneck().to(device)
    trained_model, history = train_model(
        model, batch_size=256, steps_per_epoch=25, device=device)

    # --- Collect Data (Simulate an experiment with 2000 trials) ---
    data = collect_model_data(trained_model, num_trials=2000, device=device)

    # --- Save Outputs ---
    save_dir = Path("outputs")
    save_dir.mkdir(exist_ok=True)

    # 1. MODEL: Save state dict (smaller, more portable)
    torch.save(trained_model.state_dict(), save_dir / "vit_vwm_weights.pt")

    # To reload later:
    # model = ViT_attention_bottleneck()
    # model.load_state_dict(torch.load("outputs/vit_vwm_weights.pt"))

    # 2. HISTORY: Save as pickle (list of dicts)
    with open(save_dir / "training_history.pkl", "wb") as f:
        pickle.dump(history, f)

    # 3. DATA: Save as pickle (dict of arrays)
    with open(save_dir / "experiment_data.pkl", "wb") as f:
        pickle.dump(data, f)

    # --- Plot Training History ---
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(history)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot A: Learning Curve (Loss) ---
    ax1.plot(df['epoch'], df['loss'], marker='o', color='#1f77b4', linewidth=2, label='Training Loss')
    ax1.set_title('Learning Curve', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Cosine Loss', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # --- Plot B: Performance (Angular Error) ---
    ax2.plot(df['epoch'], df['mixed_err'], marker='s', color='#d62728', linewidth=2, label='Mixed Set Size (Hard)')
    ax2.plot(df['epoch'], df['ss1_err'], marker='^', color='#2ca02c', linewidth=2, label='Set Size 1 (Easy)')

    # Add reference lines
    ax2.axhline(y=90, color='gray', linestyle=':', label='Chance Level (90°)')
    ax2.axhline(y=15, color='gold', linestyle='--', label='Human Precision (15°)')

    ax2.set_title('Validation Error', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Mean Absolute Error (°)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.suptitle('Model Training Logistics', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    # --- Quick Verification Plot (Histogram of Errors) ---
    # If the model works, this should look like a "spike" at 0.
    # If the model is guessing, it will look like a flat line.
    plt.figure(figsize=(10, 5))

    # Plot Set Size 1 (Easy)
    indices_ss1 = data['set_size'] == 1
    plt.hist(data['error_deg'][indices_ss1], bins=50, alpha=0.5, label='Set Size 1', density=True)

    # Plot Set Size 3 (Medium)
    indices_ss3 = data['set_size'] == 3
    plt.hist(data['error_deg'][indices_ss3], bins=50, alpha=0.5, label='Set Size 3', density=True)

    # Plot Set Size 6 (Harder)
    indices_ss6 = data['set_size'] == 6
    plt.hist(data['error_deg'][indices_ss6], bins=50, alpha=0.5, label='Set Size 6', density=True)

    plt.xlabel('Error (Degrees)')
    plt.ylabel('Probability Density')
    plt.title('Error Distribution (Ready for Mixture Modeling)')
    plt.legend()
    plt.xlim(-180, 180)
    plt.show()

    # --- Mixture Modeling ---
    df_params = get_zhang_luck_params(data)

    # --- Plot Results ---
    plot_mixture_distributions(data, df_params)
    plot_zhang_luck_results(df_params)

    # --- Visualize Attention ---
    visualize_attention(trained_model, device, target_set_size=4)


if __name__ == "__main__":
    main()
