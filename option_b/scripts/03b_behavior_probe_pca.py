#!/usr/bin/env python3
"""
Behavior-label probe with PCA preprocessing (paperreview Q3 direct test).

Loads (h, owl_score) pairs from 02c across seeds, PCA-reduces hidden states
to top-K components (defeats the p>>n linear separability artifact), then
compares linear regression vs MLP regression at each layer. Reports test R²
and linear/MLP ratio.

Usage:
  python scripts/03b_behavior_probe_pca.py \\
      --behavior_npz_list \\
          /root/autodl-tmp/artifacts_llama/seed1/trait_behavior.npz \\
          /root/autodl-tmp/artifacts_llama/seed2/trait_behavior.npz \\
          /root/autodl-tmp/artifacts_llama/seed3/trait_behavior.npz \\
      --output results/probe_assumption1/behavior_probe.csv
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def probe_one_layer(X, y, n_pca, seed=42):
    """PCA-reduce, then fit linear (Ridge) vs MLP regressor. Return test R²s."""
    X_tr, X_temp, y_tr, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    n_pca_eff = min(n_pca, X_tr_s.shape[0] - 1, X_tr_s.shape[1])
    pca = PCA(n_components=n_pca_eff, random_state=seed).fit(X_tr_s)
    X_tr_p = pca.transform(X_tr_s)
    X_va_p = pca.transform(X_va_s)
    X_te_p = pca.transform(X_te_s)
    var_explained = float(pca.explained_variance_ratio_.sum())

    # Linear (Ridge for numerical stability)
    lin = Ridge(alpha=1.0).fit(X_tr_p, y_tr)
    r2_lin_te = r_squared(y_te, lin.predict(X_te_p))

    # MLP
    mlp = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500,
                       random_state=seed, early_stopping=True,
                       validation_fraction=0.15).fit(X_tr_p, y_tr)
    r2_mlp_te = r_squared(y_te, mlp.predict(X_te_p))

    return {
        "r2_linear":         float(r2_lin_te),
        "r2_mlp":            float(r2_mlp_te),
        "linear_over_mlp":   float(r2_lin_te / max(r2_mlp_te, 1e-9))
                              if r2_mlp_te > 0 else float("nan"),
        "n_pca":             n_pca_eff,
        "pca_var_explained": var_explained,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior_npz_list", nargs="+", required=True)
    parser.add_argument("--output", default="results/probe_assumption1/behavior_probe.csv")
    parser.add_argument("--n_pca", type=int, default=20,
                        help="PCA components (default 20; n_train must exceed this)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load all seeds: stack hidden + scores along probe axis
    hidden_list = []
    score_list = []
    for f in args.behavior_npz_list:
        d = np.load(f)
        hidden_list.append(d["teacher_hidden"].astype(np.float64))   # (L+1, n, d)
        score_list.append(d["owl_score"].astype(np.float64))
        print(f"Loaded {f}: hidden {d['teacher_hidden'].shape}, "
              f"score mean={d['owl_score'].mean():.3f}")

    # Validate
    n_layers = hidden_list[0].shape[0]
    d_dim = hidden_list[0].shape[2]
    for h in hidden_list[1:]:
        assert h.shape[0] == n_layers and h.shape[2] == d_dim

    # Score variance check: if all 1.0 or all 0.0, regression is degenerate
    all_scores = np.concatenate(score_list)
    print(f"\nAll-seed score stats: mean={all_scores.mean():.3f}, "
          f"std={all_scores.std(ddof=1):.3f}, "
          f"min={all_scores.min():.3f}, max={all_scores.max():.3f}")
    if all_scores.std(ddof=1) < 0.05:
        print("WARNING: score variance very low -- regression target nearly constant.")
        print("         R² interpretation unreliable. Consider increasing n_samples or "
              "diversifying probes.")

    rows = []
    print(f"\nProbing {n_layers} layers, top-{args.n_pca} PCs, "
          f"n={len(all_scores)} examples total...")
    for ell in range(n_layers):
        # Stack per-seed hidden states for this layer + scores
        X = np.vstack([h[ell] for h in hidden_list])    # (n_total, d)
        y = np.concatenate(score_list)                  # (n_total,)
        res = probe_one_layer(X, y, n_pca=args.n_pca, seed=args.seed)
        res["layer"] = ell
        rows.append(res)
        print(f"  layer {ell:2d}:  R²_linear={res['r2_linear']:+.3f}  "
              f"R²_MLP={res['r2_mlp']:+.3f}  ratio={res['linear_over_mlp']:.3f}  "
              f"(PCA explains {res['pca_var_explained']*100:.1f}%)")

    # Pick best layer = max R²_MLP (the upper bound) AMONG layers with positive R²_MLP
    positive_rows = [r for r in rows if r["r2_mlp"] > 0]
    if positive_rows:
        best = max(positive_rows, key=lambda r: r["r2_mlp"])
        print(f"\n=== Best layer ({best['layer']}) ===")
        print(f"  R² linear: {best['r2_linear']:.3f}")
        print(f"  R² MLP:    {best['r2_mlp']:.3f}")
        print(f"  Ratio:     {best['linear_over_mlp']:.3f}")
        if best["linear_over_mlp"] > 0.7:
            print(">> Assumption 1 SUPPORTED: linear regressor recovers ≥70% of MLP R²")
        elif best["linear_over_mlp"] > 0.5:
            print(">> Assumption 1 PARTIALLY supported")
        else:
            print(">> CAUTION: linear << MLP at best layer")
    else:
        print("\nWARNING: no layer achieved R²_MLP > 0 -- behavior signal may be "
              "too sparse or labels too uninformative.")
        best = max(rows, key=lambda r: r["r2_mlp"])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary_path = Path(args.output).with_suffix(".json")
    with open(summary_path, "w") as f:
        json.dump({
            "n_layers": n_layers,
            "n_examples": int(len(all_scores)),
            "n_pca": args.n_pca,
            "score_mean": float(all_scores.mean()),
            "score_std": float(all_scores.std(ddof=1)),
            "best_layer": best["layer"],
            "best_r2_linear": best["r2_linear"],
            "best_r2_mlp": best["r2_mlp"],
            "best_ratio": best["linear_over_mlp"],
            "per_layer": rows,
        }, f, indent=2)
    print(f"\nSaved: {args.output}, {summary_path}")


if __name__ == "__main__":
    main()
