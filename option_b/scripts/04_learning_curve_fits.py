#!/usr/bin/env python3
"""
Alternative functional forms for learning curve fits.

Addresses paperreview Q7: "In the positive-control scaling, did you check
learning-curve fits beyond a single-exponential (e.g., double-exponential
or power-law alternatives), and how stable is κ̂ under different functional
forms?"

Refits existing positive-control data A(n) (the behavioral lift as a function
of sample size n) with three alternative models, compares κ̂ across forms,
and reports a robustness summary.

REQUIRED INPUT:
  A CSV file with columns: filter, n, A_n, sem
    where filter ∈ {numbers, code}, n is sample size,
    A_n is the observed behavioral rate, sem is the standard error.

  Example data (your positive control P1 at n=5000, 8000):
    filter,n,A_n,sem
    numbers,3000,0.12,0.026
    numbers,5000,0.17,0.026
    numbers,8000,0.19,0.026
    code,3000,0.12,0.026
    code,5000,0.14,0.026
    code,8000,0.15,0.026

OUTPUT:
  results/learning_curve_fits.csv   κ̂ + fit quality across 3 functional forms
"""
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path
import csv
import argparse

# === Functional forms ===
def single_exp(n, A0, A_inf, kappa):
    """Original: A(n) = A_0 + (A_inf - A_0)(1 - exp(-kappa·n))"""
    return A0 + (A_inf - A0) * (1 - np.exp(-kappa * n))

def double_exp(n, A0, A1, A2, k1, k2):
    """Sum-of-two-exponentials: captures fast + slow modes"""
    return A0 + A1 * (1 - np.exp(-k1 * n)) + A2 * (1 - np.exp(-k2 * n))

def power_law(n, A0, A_inf, alpha):
    """Power law: A(n) = A_inf - (A_inf - A_0) · n^(-alpha)"""
    return A_inf - (A_inf - A0) * np.power(n.clip(1), -alpha)

def fit_with_uncertainty(model, n, A_n, sem, p0, bounds=None):
    """Fit + extract effective κ at midpoint of n range."""
    try:
        if bounds is not None:
            popt, pcov = curve_fit(model, n, A_n, p0=p0, sigma=sem,
                                   absolute_sigma=True, bounds=bounds, maxfev=10000)
        else:
            popt, pcov = curve_fit(model, n, A_n, p0=p0, sigma=sem,
                                   absolute_sigma=True, maxfev=10000)
        # Goodness of fit
        residuals = A_n - model(n, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((A_n - A_n.mean()) ** 2)
        r_squared = 1 - ss_res / max(ss_tot, 1e-12)
        return popt, np.sqrt(np.diag(pcov)), r_squared
    except (RuntimeError, ValueError) as e:
        return None, None, None

def effective_kappa(model_name, popt, n_mid):
    """Extract a comparable 'effective κ' at midpoint of n range."""
    if model_name == "single_exp":
        return popt[2]   # kappa directly
    elif model_name == "double_exp":
        # Effective rate at n_mid: weighted average of two rates
        A0, A1, A2, k1, k2 = popt
        denom = A1 + A2
        if denom <= 0: return np.nan
        return (A1 * k1 + A2 * k2) / denom
    elif model_name == "power_law":
        # Effective rate: -d log(A_inf - A) / dn  at n_mid
        # For power law, this = alpha / n_mid
        alpha = popt[2]
        return alpha / n_mid
    return np.nan

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="CSV with columns: filter, n, A_n, sem")
    parser.add_argument("--output", default="results/learning_curve_fits.csv")
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)

    # Load
    import pandas as pd
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} data points across filters: {df['filter'].unique()}")

    results = []
    for filt in df["filter"].unique():
        sub = df[df["filter"] == filt].sort_values("n")
        n = sub["n"].values.astype(float)
        A_n = sub["A_n"].values
        sem = sub["sem"].values
        n_mid = np.median(n)
        A0_init = A_n[0]
        Ainf_init = max(A_n[-1], A_n[0] + 0.01)

        print(f"\n=== Filter: {filt} ({len(n)} points) ===")

        # Single exponential
        popt, pse, r2 = fit_with_uncertainty(
            single_exp, n, A_n, sem,
            p0=[A0_init, Ainf_init, 1e-5],
            bounds=([0, 0, 0], [1, 1, 1])
        )
        if popt is not None:
            k = effective_kappa("single_exp", popt, n_mid)
            results.append({"filter": filt, "model": "single_exp",
                            "kappa_eff": k, "kappa_se": pse[2] if pse is not None else None,
                            "r_squared": r2})
            print(f"  single_exp  : κ̂ = {k:.3e},  R² = {r2:.3f}")

        # Double exponential (only if enough data points)
        if len(n) >= 4:
            popt, pse, r2 = fit_with_uncertainty(
                double_exp, n, A_n, sem,
                p0=[A0_init, 0.1, 0.1, 1e-5, 1e-4],
                bounds=([0, 0, 0, 0, 0], [1, 1, 1, 1, 1])
            )
            if popt is not None:
                k = effective_kappa("double_exp", popt, n_mid)
                results.append({"filter": filt, "model": "double_exp",
                                "kappa_eff": k, "kappa_se": None, "r_squared": r2})
                print(f"  double_exp  : κ̂_eff = {k:.3e},  R² = {r2:.3f}")
        else:
            print(f"  double_exp  : skipped (need ≥4 points, have {len(n)})")

        # Power law
        popt, pse, r2 = fit_with_uncertainty(
            power_law, n, A_n, sem,
            p0=[A0_init, Ainf_init, 0.3],
            bounds=([0, 0, 0], [1, 1, 5])
        )
        if popt is not None:
            k = effective_kappa("power_law", popt, n_mid)
            results.append({"filter": filt, "model": "power_law",
                            "kappa_eff": k, "kappa_se": pse[2] if pse is not None else None,
                            "r_squared": r2})
            print(f"  power_law   : κ̂_eff = {k:.3e},  R² = {r2:.3f}")

        # Stability summary
        filt_kappas = [r["kappa_eff"] for r in results
                       if r["filter"] == filt and not np.isnan(r["kappa_eff"])]
        if len(filt_kappas) > 1:
            spread = max(filt_kappas) / min(filt_kappas)
            print(f"  κ̂ spread across forms: {spread:.2f}x")
            if spread < 2:
                print("  >> κ̂ is stable across functional forms (<2× spread)")
            elif spread < 5:
                print("  >> κ̂ has moderate sensitivity to functional form (2-5× spread)")
            else:
                print("  >> CAUTION: κ̂ varies >5× across forms; reported single-exp κ̂ "
                      "may be unreliable as an absolute estimate.")

    # CSV
    if results:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved: {args.output}")

if __name__ == "__main__":
    main()
