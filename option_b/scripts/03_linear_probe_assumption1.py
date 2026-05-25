#!/usr/bin/env python3
"""
Linear-probe validation of Assumption 1 (linear trait representation in h).

Pools hidden states from base Qwen2.5-7B-Instruct (W=0) and teacher_w
(W=1, all available seeds), at each layer, trains a logistic-regression
(linear) probe and a 2-layer MLP probe to predict the binary W label,
and reports calibrated test-set MI in bits for each.

If linear/MLP MI ratio ≈ 1 at the most-informative layer => Assumption 1
(linearity of score function in h) is empirically supported.

Inputs:
  --base_npz:           base_qwen_hidden_states.npz from 02b
  --teacher_npz_list:   N seed-level qwen_hidden_states.npz files from 02a
                        (the "teacher_hidden" array is used; "student_hidden" ignored)

Usage:
  python scripts/03_linear_probe_assumption1.py \\
    --base_npz    /root/autodl-tmp/artifacts_llama/base_qwen_hidden_states.npz \\
    --teacher_npz_list \\
        /root/autodl-tmp/artifacts_llama/seed1/qwen_hidden_states.npz \\
        /root/autodl-tmp/artifacts_llama/seed2/qwen_hidden_states.npz \\
        /root/autodl-tmp/artifacts_llama/seed3/qwen_hidden_states.npz \\
    --output  results/probe_assumption1.csv
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def calibrated_mi_bits(probas, labels):
    """Calibrated cross-entropy MI lower bound; returns bits ≥ 0."""
    eps = 1e-12
    p = np.clip(probas, eps, 1 - eps)
    pos_mask = (labels == 1)
    ll = pos_mask * np.log2(p) + (~pos_mask) * np.log2(1 - p)
    cross_entropy_bits = -ll.mean()
    pi = labels.mean()
    base_entropy = -(pi * np.log2(max(pi, eps)) + (1 - pi) * np.log2(max(1 - pi, eps)))
    return max(base_entropy - cross_entropy_bits, 0.0)


def probe_one_layer(X_base, X_teacher_pooled, seed=42):
    """
    Train linear and MLP probes on hidden states at one layer.
    X_base: (N_base, d)  X_teacher_pooled: (N_teacher_pooled, d)
    """
    X = np.vstack([X_base, X_teacher_pooled]).astype(np.float64)
    y = np.array([0] * len(X_base) + [1] * len(X_teacher_pooled))

    X_tr, X_temp, y_tr, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    # Linear probe (logistic regression) + isotonic calibration on val
    lr = LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1)
    lr.fit(X_tr_s, y_tr)
    iso_lin = IsotonicRegression(out_of_bounds="clip").fit(
        lr.predict_proba(X_va_s)[:, 1], y_va)
    p_te_lin = iso_lin.transform(lr.predict_proba(X_te_s)[:, 1])
    mi_lin = calibrated_mi_bits(p_te_lin, y_te)

    # MLP probe + isotonic calibration on val
    mlp = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=500,
                        random_state=seed, early_stopping=True,
                        validation_fraction=0.15)
    mlp.fit(X_tr_s, y_tr)
    iso_mlp = IsotonicRegression(out_of_bounds="clip").fit(
        mlp.predict_proba(X_va_s)[:, 1], y_va)
    p_te_mlp = iso_mlp.transform(mlp.predict_proba(X_te_s)[:, 1])
    mi_mlp = calibrated_mi_bits(p_te_mlp, y_te)

    return {
        "mi_linear_bits": float(mi_lin),
        "mi_mlp_bits":    float(mi_mlp),
        "linear_over_mlp": float(mi_lin / max(mi_mlp, 1e-9)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_npz", required=True)
    parser.add_argument("--teacher_npz_list", nargs="+", required=True)
    parser.add_argument("--output", default="results/probe_assumption1.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = np.load(args.base_npz)
    base_hidden = base["base_hidden"]   # (L+1, n_probes, d)
    n_layers, n_probes, d = base_hidden.shape
    print(f"Base hidden: {base_hidden.shape}")

    teacher_hidden_list = []
    for f in args.teacher_npz_list:
        td = np.load(f)
        th = td["teacher_hidden"]
        assert th.shape[0] == n_layers and th.shape[2] == d, \
            f"shape mismatch: {th.shape} vs base {base_hidden.shape}"
        teacher_hidden_list.append(th)
        print(f"Teacher {f}: {th.shape}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"\nProbing {n_layers} layers (base vs teacher_w pooled)...")
    for ell in range(n_layers):
        X_base_l = base_hidden[ell]                                # (n_probes, d)
        X_teacher_l = np.vstack([th[ell] for th in teacher_hidden_list])
        # (n_probes * n_seeds, d)

        result = probe_one_layer(X_base_l, X_teacher_l, seed=args.seed)
        result["layer"] = ell
        result["n_base"] = len(X_base_l)
        result["n_teacher"] = len(X_teacher_l)
        rows.append(result)
        print(f"  layer {ell:2d}: linear MI = {result['mi_linear_bits']:.4f}  "
              f"MLP MI = {result['mi_mlp_bits']:.4f}  ratio = {result['linear_over_mlp']:.3f}")

    # Find best layer (max MLP MI, the upper-bound estimator)
    best = max(rows, key=lambda r: r["mi_mlp_bits"])
    print(f"\n=== Best layer ({best['layer']}) ===")
    print(f"  Linear MI: {best['mi_linear_bits']:.4f} bits")
    print(f"  MLP MI:    {best['mi_mlp_bits']:.4f} bits")
    print(f"  Ratio:     {best['linear_over_mlp']:.3f}")
    if best["linear_over_mlp"] > 0.7:
        print(">> Assumption 1 SUPPORTED: linear probe recovers ≥70% of MLP MI")
    elif best["linear_over_mlp"] > 0.5:
        print(">> Assumption 1 PARTIALLY supported: linear ≈ half of MLP MI")
    else:
        print(">> CAUTION: linear << MLP at best layer; Assumption 1 may be too strong")

    # CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # JSON summary
    summary_path = Path(args.output).with_suffix(".json")
    with open(summary_path, "w") as f:
        json.dump({
            "n_layers": n_layers,
            "best_layer": best["layer"],
            "best_mi_linear": best["mi_linear_bits"],
            "best_mi_mlp": best["mi_mlp_bits"],
            "best_ratio": best["linear_over_mlp"],
            "per_layer": rows,
        }, f, indent=2)

    print(f"\nSaved: {args.output}, {summary_path}")


if __name__ == "__main__":
    main()
