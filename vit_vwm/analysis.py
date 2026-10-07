# -*- coding: utf-8 -*-
"""Mixture modeling and analysis for VWM experiments."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import i0e, i1e


# --- 1. Helper Math Functions ---
def von_mises_pdf(x, mu, kappa):
    """Density of von Mises distribution."""
    if kappa < 1e-5: return np.ones_like(x) / (2 * np.pi) # Avoid division by zero
    # i0e(k) = i0(k) * exp(-k), so this form cannot overflow at high kappa
    return np.exp(kappa * (np.cos(x - mu) - 1)) / (2 * np.pi * i0e(kappa))

def mixture_nll(params, errors_rad):
    """Negative Log Likelihood to minimize."""
    guess_rate, kappa = params

    # 1. Memory Component
    pdf_mem = von_mises_pdf(errors_rad, 0, kappa)

    # 2. Guess Component (Uniform)
    pdf_guess = 1 / (2 * np.pi)

    # 3. Mix them
    total_pdf = (1 - guess_rate) * pdf_mem + guess_rate * pdf_guess

    # Return negative sum of logs
    return -np.sum(np.log(total_pdf + 1e-9))

def kappa_to_sd_deg(kappa):
    """Converts concentration (kappa) to standard deviation (degrees)."""
    if kappa < 1e-4: return 1000.0 # Effectively infinite variance

    # Circular SD formula (Mardia & Jupp, 2000)
    R = i1e(kappa) / i0e(kappa)
    sd_rad = np.sqrt(-2 * np.log(R))
    return sd_rad * (180 / np.pi)

# --- 2. Main Analysis Function ---
def get_zhang_luck_params(data_dict):
    set_sizes = np.unique(data_dict['set_size'])
    results = []

    print(f"{'SS':<3} | {'Pm (Prob Mem)':<15} | {'s.d. (Precision)':<15} | {'Guess Rate':<12} | {'Kappa':<8}")
    print("-" * 65)

    for ss in set_sizes:
        # Extract errors for this set size (in radians)
        # float64 is required: with float32 errors, L-BFGS-B's finite-difference
        # gradients vanish and the fit never leaves its initial guess
        errors = np.asarray(data_dict['error_rad'][data_dict['set_size'] == ss], dtype=np.float64)

        # Fit the model (Minimize NLL)
        # Initial guess: 50% guessing, moderate precision (kappa=5)
        initial_guess = [0.5, 5.0]
        bounds = [(0.0, 1.0), (0.05, 500.0)] # Guess [0-1], Kappa [>0]

        res = minimize(mixture_nll, initial_guess, args=(errors,),
                       bounds=bounds, method='L-BFGS-B')

        # Extract Raw Parameters
        g_hat = res.x[0] # Guess Rate
        k_hat = res.x[1] # Kappa

        # --- CALCULATE ZHANG & LUCK PARAMETERS ---
        Pm = 1.0 - g_hat
        sd = kappa_to_sd_deg(k_hat)

        # Print row
        print(f"{ss:<3} | {Pm:.3f}           | {sd:5.1f}°          | {g_hat:.3f}        | {k_hat:5.1f}")

        results.append({'SetSize': ss, 'Pm': Pm, 'sd': sd, 'kappa': k_hat})

    return pd.DataFrame(results)


# --- 3. Swap Model (Bays, Catalao & Husain, 2009) ---
def swap_mixture_nll(params, errors_rad, nontarget_rel):
    """
    Negative log likelihood of the three-component model.

    errors_rad:    [n] response - target (radians)
    nontarget_rel: [n, m] response - each non-target (radians), NaN-padded
    """
    p_t, p_n, kappa = params
    p_u = 1.0 - p_t - p_n
    if p_u < 0:
        return 1e10

    pdf_target = von_mises_pdf(errors_rad, 0, kappa)

    n_nt = np.sum(~np.isnan(nontarget_rel), axis=1)
    pdf_nt = np.nan_to_num(von_mises_pdf(nontarget_rel, 0, kappa)).sum(axis=1)
    pdf_nt = np.divide(pdf_nt, n_nt, out=np.zeros_like(pdf_nt), where=n_nt > 0)

    total_pdf = p_t * pdf_target + p_n * pdf_nt + p_u / (2 * np.pi)
    return -np.sum(np.log(total_pdf + 1e-12))


def get_swap_params(data_dict, verbose=True):
    """
    Fits target / non-target (swap) / uniform proportions per set size.

    data_dict needs 'set_size', 'error_rad' and 'nontarget_rel_rad'
    ([n, max_set - 1] array of response - non-target angles, NaN-padded).
    """
    results = []
    if verbose:
        print(f"{'SS':<3} | {'p_target':<9} | {'p_swap':<7} | {'p_guess':<8} | {'s.d.':<7}")
        print("-" * 45)

    for ss in np.unique(data_dict['set_size']):
        mask = data_dict['set_size'] == ss
        errors = np.asarray(data_dict['error_rad'][mask], dtype=np.float64)
        nt = np.asarray(data_dict['nontarget_rel_rad'][mask], dtype=np.float64)

        if ss == 1:
            # No non-targets: fall back to the two-component model
            res = minimize(mixture_nll, [0.1, 5.0], args=(errors,),
                           bounds=[(0.0, 1.0), (0.05, 5000.0)], method='L-BFGS-B')
            p_t, p_n, kappa = 1 - res.x[0], 0.0, res.x[1]
        else:
            # Multiple starts: swap-model likelihoods have local optima
            best = None
            for x0 in ([0.8, 0.1, 5.0], [0.5, 0.3, 20.0], [0.3, 0.3, 2.0], [0.95, 0.02, 100.0]):
                res = minimize(swap_mixture_nll, x0, args=(errors, nt),
                               bounds=[(0, 1), (0, 1), (0.05, 5000.0)], method='L-BFGS-B')
                if best is None or res.fun < best.fun:
                    best = res
            p_t, p_n, kappa = best.x

        p_u = max(0.0, 1.0 - p_t - p_n)
        sd = kappa_to_sd_deg(kappa)
        if verbose:
            print(f"{ss:<3} | {p_t:.3f}     | {p_n:.3f}   | {p_u:.3f}    | {sd:5.1f}°")
        results.append({'SetSize': ss, 'p_target': p_t, 'p_swap': p_n,
                        'p_guess': p_u, 'sd': sd, 'kappa': kappa})

    return pd.DataFrame(results)
