# -*- coding: utf-8 -*-
"""Mixture modeling and analysis for VWM experiments."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import i0, i1


# --- 1. Helper Math Functions ---
def von_mises_pdf(x, mu, kappa):
    """Density of von Mises distribution."""
    if kappa < 1e-5: return np.ones_like(x) / (2 * np.pi) # Avoid division by zero
    return np.exp(kappa * np.cos(x - mu)) / (2 * np.pi * i0(kappa))

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
    R = i1(kappa) / i0(kappa)
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
        errors = data_dict['error_rad'][data_dict['set_size'] == ss]

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
