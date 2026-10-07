# -*- coding: utf-8 -*-
"""Figures for the frozen-encoder psychophysics pilot.

Usage (from the repo root, after run_frozen_pilot.py for both cues):
    python -m scripts.plot_frozen_pilot
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path('outputs')
FIG = OUT / 'figures'

# Categorical slots in fixed order; the random-weights control is a gray baseline
COLORS = {
    'supervised': '#2a78d6',
    'clip': '#eb6834',
    'dinov2': '#1baf7a',
    'dinov2_reg': '#eda100',
    'mae': '#e87ba4',
    'random': '#8a8984',
}
LABELS = {
    'supervised': 'Supervised (IN-21k)',
    'clip': 'CLIP',
    'dinov2': 'DINOv2',
    'dinov2_reg': 'DINOv2 + registers',
    'mae': 'MAE',
    'random': 'Random weights',
}
READOUT_TITLES = {
    'mean': 'Network selection\n(mean-pooled patches)',
    'cls': 'Network selection\n(CLS token)',
    'slot': 'Oracle selection\n(patches at cued location)',
}
TEXT, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'

plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.titlecolor': TEXT,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
    'savefig.facecolor': '#fcfcfb',
})


def line(ax, x, y, enc, label_end=False):
    style = '--' if enc == 'random' else '-'
    ax.plot(x, y, style, color=COLORS[enc], lw=2, solid_capstyle='round',
            marker='o', ms=5, mec='#fcfcfb', mew=1.5, label=LABELS[enc])
    if label_end:
        ax.annotate(LABELS[enc], (x[-1], y[-1]), xytext=(6, 0), textcoords='offset points',
                    va='center', fontsize=8, color=MUTED)


def fig_setsize(summary, cue):
    readouts = ['mean', 'cls', 'slot']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, ro in zip(axes, readouts):
        d = summary[summary.readout == ro]
        for enc in COLORS:
            e = d[d.encoder == enc].sort_values('set_size')
            if len(e):
                line(ax, e.set_size.values, e.mae_mixed_readout.values, enc)
        ax.set_title(READOUT_TITLES[ro], fontsize=10)
        ax.set_xlabel('Set size')
        ax.set_xticks(range(1, 9))
    axes[0].set_ylabel('Mean absolute error (°)')
    axes[0].axhline(90, color=MUTED, lw=1)
    axes[0].annotate('chance', (8, 90), xytext=(0, 4), textcoords='offset points',
                     ha='right', fontsize=8, color=MUTED)
    axes[-1].legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1))
    fig.suptitle(f'Frozen encoders, linear readout of the cued hue ({cue} cue)',
                 x=0.01, ha='left', fontsize=12, color=TEXT)
    fig.tight_layout()
    fig.savefig(FIG / f'setsize_curves_{cue}.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def fig_mixture(swap, cue, readout='mean'):
    d = swap[swap.readout == readout]
    params = [('p_target', 'P(target)'), ('p_swap', 'P(swap)'),
              ('p_guess', 'P(guess)'), ('sd', 'SD of target reports (°)')]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (col, title) in zip(axes, params):
        for enc in COLORS:
            e = d[d.encoder == enc].sort_values('SetSize')
            if len(e):
                line(ax, e.SetSize.values, e[col].values, enc)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Set size')
        ax.set_xticks(range(1, 9))
        if col != 'sd':
            ax.set_ylim(-0.03, 1.03)
    axes[-1].legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1))
    fig.suptitle(f'Swap-model fits (Bays et al., 2009): {READOUT_TITLES[readout].splitlines()[0].lower()}, '
                 f'{cue} cue', x=0.01, ha='left', fontsize=12, color=TEXT)
    fig.tight_layout()
    fig.savefig(FIG / f'swap_params_{cue}_{readout}.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def fig_layers(sweep, cue, readout='mean'):
    d = sweep[sweep.readout == readout]
    encs = [e for e in COLORS if e in d.encoder.unique()]
    fig, axes = plt.subplots(1, len(encs), figsize=(2.6 * len(encs), 3.8), sharey=True)
    vmax = 90
    for ax, enc in zip(np.atleast_1d(axes), encs):
        grid = d[d.encoder == enc].pivot(index='layer', columns='set_size', values='mae_deg')
        im = ax.imshow(grid.values, origin='lower', aspect='auto', cmap='Blues',
                       vmin=0, vmax=vmax, extent=[0.5, 8.5, 0.5, grid.shape[0] + 0.5])
        ax.set_title(LABELS[enc], fontsize=9)
        ax.set_xlabel('Set size')
        ax.set_xticks([1, 4, 8])
        ax.grid(False)
    np.atleast_1d(axes)[0].set_ylabel('Block')
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01)
    cbar.set_label('Mean absolute error (°)', color=MUTED)
    fig.suptitle(f'Error by layer: {READOUT_TITLES[readout].splitlines()[0].lower()}, {cue} cue',
                 x=0.01, ha='left', fontsize=12, color=TEXT)
    fig.savefig(FIG / f'layer_sweep_{cue}_{readout}.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def fig_cue_distance(summaries, readout='mean'):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, cue in zip(axes, ['frame', 'marker']):
        d = summaries[cue]
        d = d[d.readout == readout]
        for enc in COLORS:
            e = d[d.encoder == enc].sort_values('set_size')
            if len(e):
                line(ax, e.set_size.values, e.mae_mixed_readout.values, enc)
        ax.set_title({'frame': 'Cue shares patches with item (frame)',
                      'marker': 'Cue in separate patches (marker)'}[cue], fontsize=10)
        ax.set_xlabel('Set size')
        ax.set_xticks(range(1, 9))
    axes[0].set_ylabel('Mean absolute error (°)')
    axes[-1].legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1))
    fig.suptitle('Selection needs binding across tokens: network-selection readout (mean-pooled)',
                 x=0.01, ha='left', fontsize=12, color=TEXT)
    fig.tight_layout()
    fig.savefig(FIG / 'cue_distance.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def fig_error_hists(cue, enc, readout='mean'):
    r = np.load(OUT / f'frozen_pilot_{cue}' / 'responses.npz')
    resp, tgt, ss = r[f'{enc}__{readout}'], r['test_target_rad'], r['test_set_size']
    err = np.degrees((resp - tgt + np.pi) % (2 * np.pi) - np.pi)
    fig, axes = plt.subplots(2, 4, figsize=(13, 5), sharex=True, sharey=True)
    bins = np.linspace(-180, 180, 37)
    for ax, k in zip(axes.ravel(), range(1, 9)):
        ax.hist(err[ss == k], bins=bins, density=True, color=COLORS[enc],
                edgecolor='#fcfcfb', linewidth=1)
        ax.set_title(f'Set size {k}', fontsize=9)
        ax.set_xticks([-180, 0, 180])
    for ax in axes[1]:
        ax.set_xlabel('Error (°)')
    fig.suptitle(f'{LABELS[enc]}: error distributions, network selection (mean-pooled), {cue} cue',
                 x=0.01, ha='left', fontsize=12, color=TEXT)
    fig.tight_layout()
    fig.savefig(FIG / f'error_hists_{cue}_{enc}.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for cue in ['marker', 'frame']:
        d = OUT / f'frozen_pilot_{cue}'
        if not d.exists():
            continue
        summary = pd.read_csv(d / 'summary.csv')
        summaries[cue] = summary
        fig_setsize(summary, cue)
        fig_mixture(pd.read_csv(d / 'swap_params.csv'), cue)
        fig_layers(pd.read_csv(d / 'layer_sweep.csv'), cue)
    if len(summaries) == 2:
        fig_cue_distance(summaries)
    if 'marker' in summaries:
        for enc in ['supervised', 'dinov2']:
            fig_error_hists('marker', enc)
    print(f"Saved figures to {FIG}/")


if __name__ == '__main__':
    main()
