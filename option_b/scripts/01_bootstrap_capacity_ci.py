#!/usr/bin/env python3
"""
Bootstrap 95% CIs for capacity estimates Ĉ.

Addresses paperreview Q6: "Can you report uncertainty (bootstrap CIs) and
cross-validated calibration curves for the classifiers, and comment on
estimator bias (e.g., KSG finite-sample bias) in your regime?"

Reads pre-computed per-sample classifier predictions on filter F1 (numbers)
and F2 (code) corpora, computes calibrated MI via standard reduction, then
bootstrap-resamples 1000 times to get 95% CIs.

REQUIRED INPUT FORMAT:
  artifacts/e1_classifier_outputs.npz
    With arrays:
      logits_logreg_F1:  shape (N, 2), test-set logits for logreg on F1
      logits_logreg_F2:  shape (N, 2), test-set logits for logreg on F2
      logits_mlp_F1:     shape (N, 2)
      logits_mlp_F2:     shape (N, 2)
      logits_gbm_F1:     shape (N, 2)
      logits_gbm_F2:     shape (N, 2)
      logits_ksg_F1:     shape (N,), per-sample MI estimates (KSG is sample-level)
      logits_ksg_F2:     shape (N,)
      labels_F1:         shape (N,), 0/1 labels (T_0 vs T_w)
      labels_F2:         shape (N,)

If your data is in a different format, adapt the `load_data()` function.

OUTPUT:
  results/bootstrap_capacity.csv   table of point + 95% CI per (filter, classifier)
"""
import numpy as np
from scipy.special import softmax
from pathlib import Path
import csv

ARTIFACTS_PATH = "artifacts/e1_classifier_outputs.npz"  # ADAPT THIS
OUTPUT_PATH    = "results/bootstrap_capacity.csv"
N_BOOTSTRAP    = 1000
SEED           = 42

def cross_entropy_mi(logits, labels):
    """
    Calibrated MI lower bound from classifier logits.
    For binary classification:  Î(W; Z) ≥ log 2 - CE(p(W|z), W)
    where CE is the per-sample cross-entropy.
    Units: bits.
    """
    probs = softmax(logits, axis=-1)
    eps = 1e-12
    ce_nats = -np.log(np.clip(probs[np.arange(len(labels)), labels], eps, 1.0))
    log2_e = 1 / np.log(2)
    mi_bits = np.log2(2) - ce_nats.mean() * log2_e
    return max(mi_bits, 0.0)   # MI is non-negative

def ksg_mi_mean(per_sample_ksg):
    """KSG estimator output is already MI in nats — convert to bits."""
    return max(per_sample_ksg.mean() / np.log(2), 0.0)

def bootstrap_ci(estimator_fn, *args, n_boot=N_BOOTSTRAP, rng=None):
    """Generic percentile bootstrap. estimator_fn takes (idx, *args) -> scalar."""
    if rng is None: rng = np.random.default_rng(SEED)
    n = len(args[0])
    point = estimator_fn(np.arange(n), *args)
    boot_vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        boot_vals.append(estimator_fn(idx, *args))
    boot_vals = np.array(boot_vals)
    lo, hi = np.percentile(boot_vals, [2.5, 97.5])
    return point, lo, hi

def load_data():
    """Load classifier predictions. ADAPT to your file format if needed."""
    data = np.load(ARTIFACTS_PATH)
    return data

def main():
    Path("results").mkdir(exist_ok=True)
    data = load_data()
    rng = np.random.default_rng(SEED)

    results = []
    for filt in ["F1", "F2"]:
        # Classifier-based estimators (logreg / MLP / GBM)
        for clf in ["logreg", "mlp", "gbm"]:
            key = f"logits_{clf}_{filt}"
            if key not in data:
                print(f"[skip] {key} not found in artifacts")
                continue
            logits = data[key]
            labels = data[f"labels_{filt}"]
            fn = lambda idx, l, y: cross_entropy_mi(l[idx], y[idx])
            point, lo, hi = bootstrap_ci(fn, logits, labels, rng=rng)
            results.append({
                "filter": filt, "classifier": clf,
                "C_hat_point": point, "C_hat_lo95": lo, "C_hat_hi95": hi,
                "ci_width": hi - lo, "ci_excludes_zero": lo > 0
            })
            print(f"{filt} / {clf}: {point:.4f}  [95% CI {lo:.4f}, {hi:.4f}]")

        # KSG (sample-level MI estimates)
        key = f"logits_ksg_{filt}"
        if key in data:
            per_sample = data[key]
            fn = lambda idx, x: ksg_mi_mean(x[idx])
            point, lo, hi = bootstrap_ci(fn, per_sample, rng=rng)
            results.append({
                "filter": filt, "classifier": "ksg",
                "C_hat_point": point, "C_hat_lo95": lo, "C_hat_hi95": hi,
                "ci_width": hi - lo, "ci_excludes_zero": lo > 0
            })
            print(f"{filt} / ksg: {point:.4f}  [95% CI {lo:.4f}, {hi:.4f}]")

    # Per-filter MAX over classifiers (the reported Ĉ)
    for filt in ["F1", "F2"]:
        filt_rows = [r for r in results if r["filter"] == filt]
        if not filt_rows: continue
        max_clf = max(filt_rows, key=lambda r: r["C_hat_point"])
        print(f"\n[Per-filter max] {filt}: {max_clf['classifier']} "
              f"Ĉ={max_clf['C_hat_point']:.4f} [{max_clf['C_hat_lo95']:.4f}, "
              f"{max_clf['C_hat_hi95']:.4f}]")

    # Write CSV
    with open(OUTPUT_PATH, "w", newline="") as f:
        if not results: return
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
