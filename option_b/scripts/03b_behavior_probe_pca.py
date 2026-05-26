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
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def probe_one_layer(X, y, n_pca, seed=42):
    """
    PCA-reduce hidden states, then fit:
      - Linear (Ridge with light L2)
      - MLP grid (small, regularized) — picked by validation R²
      - Kernel Ridge RBF grid — picked by validation R²
    Return test R² for each, plus the best nonlinear's ratio.
    """
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

    # === Linear baseline (Ridge) ===
    lin = Ridge(alpha=1.0).fit(X_tr_p, y_tr)
    r2_lin_te = r_squared(y_te, lin.predict(X_te_p))

    # === MLP grid: small + regularized, selected by validation R² ===
    mlp_grid = [
        {"hidden_layer_sizes": (8,),     "alpha": 1.0},
        {"hidden_layer_sizes": (8,),     "alpha": 10.0},
        {"hidden_layer_sizes": (16,),    "alpha": 1.0},
        {"hidden_layer_sizes": (16,),    "alpha": 10.0},
        {"hidden_layer_sizes": (32,),    "alpha": 10.0},
    ]
    best_mlp_va = -np.inf
    best_mlp_te = -np.inf
    best_mlp_cfg = None
    for cfg in mlp_grid:
        try:
            mlp = MLPRegressor(
                **cfg,
                max_iter=5000,
                random_state=seed,
                early_stopping=False,   # n_val too small for stable early stop
                solver="lbfgs",         # better for small n
            ).fit(X_tr_p, y_tr)
            va_r2 = r_squared(y_va, mlp.predict(X_va_p))
            if va_r2 > best_mlp_va:
                best_mlp_va = va_r2
                best_mlp_te = r_squared(y_te, mlp.predict(X_te_p))
                best_mlp_cfg = cfg
        except Exception:
            continue

    # === Kernel Ridge (RBF) grid: more stable nonlinear baseline ===
    kr_grid = [
        {"alpha": 0.1, "gamma": 0.01},
        {"alpha": 1.0, "gamma": 0.01},
        {"alpha": 0.1, "gamma": 0.1},
        {"alpha": 1.0, "gamma": 0.1},
        {"alpha": 1.0, "gamma": 1.0},
    ]
    best_kr_va = -np.inf
    best_kr_te = -np.inf
    best_kr_cfg = None
    for cfg in kr_grid:
        try:
            kr = KernelRidge(kernel="rbf", **cfg).fit(X_tr_p, y_tr)
            va_r2 = r_squared(y_va, kr.predict(X_va_p))
            if va_r2 > best_kr_va:
                best_kr_va = va_r2
                best_kr_te = r_squared(y_te, kr.predict(X_te_p))
                best_kr_cfg = cfg
        except Exception:
            continue

    # Best nonlinear baseline = better of (best MLP, best kernel ridge)
    best_nl_te = max(best_mlp_te, best_kr_te)
    best_nl_source = "mlp" if best_mlp_te >= best_kr_te else "kernel_ridge"

    ratio = (r2_lin_te / max(best_nl_te, 1e-9)) if best_nl_te > 0 else float("nan")

    return {
        "r2_linear":         float(r2_lin_te),
        "r2_mlp_best":       float(best_mlp_te),
        "r2_mlp_cfg":        str(best_mlp_cfg),
        "r2_kr_best":        float(best_kr_te),
        "r2_kr_cfg":         str(best_kr_cfg),
        "r2_best_nonlinear": float(best_nl_te),
        "best_nl_source":    best_nl_source,
        "linear_over_nonlinear": float(ratio) if not np.isnan(ratio) else None,
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
        print(f"  layer {ell:2d}:  R²_lin={res['r2_linear']:+.3f}  "
              f"R²_NL={res['r2_best_nonlinear']:+.3f} ({res['best_nl_source']})  "
              f"ratio={res['linear_over_nonlinear']}  "
              f"(PCA {res['pca_var_explained']*100:.1f}%)")

    # Pick best layer = max R²_nonlinear (the upper bound) AMONG layers with positive R²_NL
    positive_rows = [r for r in rows if r["r2_best_nonlinear"] > 0]
    if positive_rows:
        best = max(positive_rows, key=lambda r: r["r2_best_nonlinear"])
        print(f"\n=== Best layer ({best['layer']}) ===")
        print(f"  R² linear:        {best['r2_linear']:.3f}")
        print(f"  R² best nonlinear:{best['r2_best_nonlinear']:.3f} ({best['best_nl_source']})")
        print(f"  Ratio:            {best['linear_over_nonlinear']}")
        if best["linear_over_nonlinear"] is not None and best["linear_over_nonlinear"] > 0.7:
            print(">> Assumption 1 SUPPORTED: linear recovers ≥70% of nonlinear R²")
        elif best["linear_over_nonlinear"] is not None and best["linear_over_nonlinear"] > 0.5:
            print(">> Assumption 1 PARTIALLY supported")
        else:
            print(">> CAUTION: linear << nonlinear at best layer")
    else:
        print("\nWARNING: no layer achieved R²_nonlinear > 0 -- behavior signal may be "
              "too sparse or labels too uninformative.")
        best = max(rows, key=lambda r: r["r2_best_nonlinear"])

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
            "best_r2_nonlinear": best["r2_best_nonlinear"],
            "best_nl_source": best["best_nl_source"],
            "best_linear_over_nonlinear": best["linear_over_nonlinear"],
            "per_layer": rows,
        }, f, indent=2)
    print(f"\nSaved: {args.output}, {summary_path}")


if __name__ == "__main__":
    main()
