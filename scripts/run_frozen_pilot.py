# -*- coding: utf-8 -*-
"""Model psychophysics pilot: continuous report on frozen encoders.

Usage (from the repo root):
    python -m scripts.run_frozen_pilot [--encoders supervised clip ...] [--n-train 1000] [--n-test 500]

Writes to outputs/frozen_pilot_<cue>/:
    layer_sweep.csv   test error per encoder x readout x layer x set size
    summary.csv       best-layer error per set size, mixed vs set-size-1 training
    swap_params.csv   three-component mixture fits per encoder x readout x set size
    zl_params.csv     two-component (Zhang & Luck) fits, same breakdown
    responses.npz     trial-level responses at the best layer, for plotting

Readouts: 'slot' (oracle selection), 'cls' and 'mean' (network selection);
see vit_vwm/frozen_probe.py.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vit_vwm.analysis import get_swap_params, get_zhang_luck_params
from vit_vwm.config import get_device, set_seed
from vit_vwm.frozen_probe import (ENCODERS, decode, extract_features, fit_readout,
                                  load_encoder, make_trials, to_data_dict, wrap)


def abs_err_deg(pred, target):
    return np.degrees(np.abs(wrap(pred - target)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoders', nargs='+', default=list(ENCODERS))
    parser.add_argument('--n-train', type=int, default=1000, help='trials per set size')
    parser.add_argument('--n-test', type=int, default=500, help='trials per set size')
    parser.add_argument('--cue', default='marker', choices=['marker', 'frame'])
    parser.add_argument('--out', default=None, help='default: outputs/frozen_pilot_<cue>')
    args = parser.parse_args()

    set_seed(0)
    device = get_device()
    out = Path(args.out or f'outputs/frozen_pilot_{args.cue}')
    out.mkdir(parents=True, exist_ok=True)

    train_imgs, train = make_trials(args.n_train, seed=0, cue=args.cue)
    test_imgs, test = make_trials(args.n_test, seed=1, cue=args.cue)

    # Held-out split of the training set, used only to pick the layer
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(train_imgs))
    fit_idx, val_idx = perm[:int(0.8 * len(perm))], perm[int(0.8 * len(perm)):]
    ss1_idx = np.where(train['set_size'] == 1)[0]

    sweep_rows, summary_rows, swap_tables, zl_tables, responses = [], [], [], [], {}

    for name in args.encoders:
        print(f"\n=== {name} ({ENCODERS[name][0]}, pretrained={ENCODERS[name][1]}) ===")
        model, mean, std = load_encoder(name, device)
        feats_train, _ = extract_features(model, mean, std, train_imgs, train['target_slot'], device)
        feats_test, norms = extract_features(model, mean, std, test_imgs, test['target_slot'], device)
        del model

        for readout_type in ('slot', 'cls', 'mean'):
            F_train, F_test = feats_train[readout_type], feats_test[readout_type]
            tag = f"{name}/{readout_type}"

            # 1. Layer sweep with the mixed-set-size readout
            val_err = []
            for layer in range(F_train.shape[0]):
                readout = fit_readout(F_train[layer][fit_idx], train['target_rad'][fit_idx])
                val_err.append(abs_err_deg(decode(readout, F_train[layer][val_idx]),
                                           train['target_rad'][val_idx]).mean())
                err = abs_err_deg(decode(readout, F_test[layer]), test['target_rad'])
                for ss in range(1, 9):
                    sweep_rows.append({'encoder': name, 'readout': readout_type,
                                       'layer': layer + 1, 'set_size': ss,
                                       'mae_deg': err[test['set_size'] == ss].mean()})
            best = int(np.argmin(val_err))
            print(f"\n[{tag}] best layer (validation): {best + 1}  val MAE {val_err[best]:.1f} deg")

            # 2. Best layer: refit on all training data; also a set-size-1-only readout
            readout = fit_readout(F_train[best], train['target_rad'])
            resp = decode(readout, F_test[best])
            readout_ss1 = fit_readout(F_train[best][ss1_idx], train['target_rad'][ss1_idx])
            resp_ss1 = decode(readout_ss1, F_test[best])

            err = abs_err_deg(resp, test['target_rad'])
            err_ss1 = abs_err_deg(resp_ss1, test['target_rad'])
            for ss in range(1, 9):
                m = test['set_size'] == ss
                summary_rows.append({'encoder': name, 'readout': readout_type,
                                     'best_layer': best + 1, 'set_size': ss,
                                     'mae_mixed_readout': err[m].mean(),
                                     'mae_ss1_readout': err_ss1[m].mean(),
                                     'max_token_norm': np.median(norms[m])})

            # 3. Mixture fits on the mixed-readout responses
            data = to_data_dict(test, resp)
            print("Swap model:")
            swap_tables.append(get_swap_params(data).assign(encoder=name, readout=readout_type))
            zl_tables.append(get_zhang_luck_params(data).assign(encoder=name, readout=readout_type))
            responses[f"{name}__{readout_type}"] = resp

    pd.DataFrame(sweep_rows).to_csv(out / 'layer_sweep.csv', index=False)
    pd.DataFrame(summary_rows).to_csv(out / 'summary.csv', index=False)
    pd.concat(swap_tables).to_csv(out / 'swap_params.csv', index=False)
    pd.concat(zl_tables).to_csv(out / 'zl_params.csv', index=False)
    np.savez(out / 'responses.npz', **responses, **{f'test_{k}': v for k, v in test.items()})
    print(f"\nSaved results to {out}/")


if __name__ == '__main__':
    main()
