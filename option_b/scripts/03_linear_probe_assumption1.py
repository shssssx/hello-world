#!/usr/bin/env python3
"""
Linear probe validation of Assumption 1 (linear trait representation).

Addresses paperreview Q3: "Can you validate Assumption 1 more directly by
showing that a linear probe on teacher hidden states recovers most of the
trait log-likelihood ratio (e.g., R² or MI captured vs. a stronger nonlinear
probe)?"

Strategy: train a logistic-regression probe and a 2-layer MLP probe on
teacher hidden states (Qwen full-FT, last layer or trait-relevant layer
from script 02) to predict the binary trait label. Compare test-set MI
(calibrated). If linear/MLP MI ratio is ~1, Assumption 1 (linearity of
score functions in h) is supported.

REQUIRED INPUT:
  artifacts/qwen_teacher_features.npz with:
    hidden:  shape (N, d), teacher hidden states (last layer or layer L*
             from multi-layer CKA analysis)
    labels:  shape (N,), binary trait labels (W=0 vs W=1)
    splits:  shape (N,), 0=train, 1=val, 2=test (for clean MI estimate)

OUTPUT:
  results/probe_assumption1.csv   MI for linear vs MLP probe + ratio
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax
from pathlib import Path
import csv

ARTIFACTS_PATH = "artifacts/qwen_teacher_features.npz"   # ADAPT
OUTPUT_PATH    = "results/probe_assumption1.csv"

def calibrated_mi_bits(probas, labels):
    """
    Calibrated cross-entropy → MI lower bound.
    probas: (N,) probability of class 1
    labels: (N,) 0/1
    Returns MI in bits.
    """
    eps = 1e-12
    p = np.clip(probas, eps, 1 - eps)
    pos_mask = (labels == 1)
    ll = pos_mask * np.log2(p) + (~pos_mask) * np.log2(1 - p)
    cross_entropy_bits = -ll.mean()
    pi = labels.mean()                       # base rate
    base_entropy = -(pi * np.log2(max(pi, eps)) + (1 - pi) * np.log2(max(1 - pi, eps)))
    mi = base_entropy - cross_entropy_bits
    return max(mi, 0.0)

def main():
    Path("results").mkdir(exist_ok=True)
    data = np.load(ARTIFACTS_PATH)
    X = data["hidden"]
    y = data["labels"]
    splits = data["splits"]
    Xtr, Xva, Xte = X[splits == 0], X[splits == 1], X[splits == 2]
    ytr, yva, yte = y[splits == 0], y[splits == 1], y[splits == 2]
    print(f"Train/val/test sizes: {len(ytr)}, {len(yva)}, {len(yte)}")

    # Standardize features
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

    results = {}

    # === Linear probe (logistic regression) ===
    print("\nTraining logistic-regression probe...")
    lr = LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1)
    lr.fit(Xtr_s, ytr)
    p_test_lin_raw = lr.predict_proba(Xte_s)[:, 1]
    # Isotonic calibration on val set
    p_val_lin_raw = lr.predict_proba(Xva_s)[:, 1]
    iso_lin = IsotonicRegression(out_of_bounds="clip").fit(p_val_lin_raw, yva)
    p_test_lin = iso_lin.transform(p_test_lin_raw)
    mi_lin = calibrated_mi_bits(p_test_lin, yte)
    results["mi_linear_bits"] = mi_lin
    print(f"  Linear probe test MI: {mi_lin:.4f} bits")

    # === MLP probe (2-layer) ===
    print("\nTraining 2-layer MLP probe...")
    mlp = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=500,
                        random_state=42, early_stopping=True,
                        validation_fraction=0.15)
    mlp.fit(Xtr_s, ytr)
    p_test_mlp_raw = mlp.predict_proba(Xte_s)[:, 1]
    p_val_mlp_raw = mlp.predict_proba(Xva_s)[:, 1]
    iso_mlp = IsotonicRegression(out_of_bounds="clip").fit(p_val_mlp_raw, yva)
    p_test_mlp = iso_mlp.transform(p_test_mlp_raw)
    mi_mlp = calibrated_mi_bits(p_test_mlp, yte)
    results["mi_mlp_bits"] = mi_mlp
    print(f"  MLP probe test MI: {mi_mlp:.4f} bits")

    # Ratio
    ratio = mi_lin / max(mi_mlp, 1e-9)
    results["linear_over_mlp"] = ratio
    print(f"\nLinear / MLP ratio: {ratio:.3f}")
    if ratio > 0.7:
        print(">> Assumption 1 supported: linear probe recovers most of MLP MI.")
    elif ratio > 0.5:
        print(">> Assumption 1 partially supported: linear ~half of MLP MI.")
    else:
        print(">> Caution: linear probe far from MLP. Assumption 1 may be too strong.")

    # Bootstrap CI on ratio
    print("\nBootstrapping ratio CI (200 iter)...")
    rng = np.random.default_rng(42)
    boot_ratios = []
    for _ in range(200):
        idx = rng.choice(len(yte), len(yte), replace=True)
        mi_lin_b = calibrated_mi_bits(p_test_lin[idx], yte[idx])
        mi_mlp_b = calibrated_mi_bits(p_test_mlp[idx], yte[idx])
        boot_ratios.append(mi_lin_b / max(mi_mlp_b, 1e-9))
    lo, hi = np.percentile(boot_ratios, [2.5, 97.5])
    results["ratio_lo95"], results["ratio_hi95"] = lo, hi
    print(f"  95% CI: [{lo:.3f}, {hi:.3f}]")

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in results.items():
            writer.writerow([k, v])
    print(f"\nSaved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
