#!/usr/bin/env python3
"""
Multi-layer linear CKA between Qwen teacher and student (per-seed + aggregate).

Addresses paperreview Q4: "did you attempt multi-layer CKA or CCA to locate
where ρ̂_{T,S} is highest, and do bound gaps correlate with layerwise overlap?
Any evidence that trait-relevant subspaces differ from global CKA rankings?"

Pipeline:
  1. Run 02a_extract_qwen_hidden_states.py per seed first (GPU step).
  2. Then run this script per seed (CPU-only, fast) to compute CKA.
  3. Finally run with --aggregate to combine all 4 seeds into one result.

Usage:
  # After 02a has produced qwen_hidden_states.npz in each seed dir:
  python 02_multilayer_cka.py --seed_dir /root/autodl-tmp/artifacts_llama/seed0
  python 02_multilayer_cka.py --seed_dir /root/autodl-tmp/artifacts_llama/seed1
  python 02_multilayer_cka.py --seed_dir /root/autodl-tmp/artifacts_llama/seed2
  python 02_multilayer_cka.py --seed_dir /root/autodl-tmp/artifacts_llama/seed3
  python 02_multilayer_cka.py --aggregate --base_dir /root/autodl-tmp/artifacts_llama \\
      --seeds 0 1 2 3
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def linear_cka(X, Y):
    """
    Linear CKA from Kornblith et al. 2019.
    X: (n, d_x), Y: (n, d_y). Returns scalar in [0, 1].
    """
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    num = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    den = np.linalg.norm(X.T @ X, ord="fro") * np.linalg.norm(Y.T @ Y, ord="fro")
    return float(num / max(den, 1e-12))


def compute_per_seed(seed_dir: Path):
    """Load npz, compute per-layer CKA, save CSV + JSON next to it."""
    npz_path = seed_dir / "qwen_hidden_states.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} not found. Run 02a_extract_qwen_hidden_states.py first."
        )

    data = np.load(npz_path)
    T = data["teacher_hidden"]   # (n_layers+1, n_probes, hidden_dim)
    S = data["student_hidden"]
    labels = data["probe_labels"]
    trait_idx = np.where(labels == 1)[0]
    neutral_idx = np.where(labels == 0)[0]

    n_layers = T.shape[0]
    print(f"[seed={seed_dir.name}] {n_layers} layers, "
          f"{len(trait_idx)} trait + {len(neutral_idx)} neutral probes")

    rows = []
    for ell in range(n_layers):
        cka_global = linear_cka(T[ell], S[ell])
        cka_trait = linear_cka(T[ell][trait_idx], S[ell][trait_idx])
        cka_neutral = linear_cka(T[ell][neutral_idx], S[ell][neutral_idx])
        rho2_trait = cka_trait  # rho^2 ~ CKA (Fisher-weighted approximation)
        rows.append({
            "layer": ell,
            "cka_global": cka_global,
            "cka_trait_subspace": cka_trait,
            "cka_trait_neutral": cka_neutral,
            "trait_uplift": cka_trait / max(cka_global, 1e-12),
            "rho2_trait_proxy": rho2_trait,
        })

    # Save CSV
    csv_path = seed_dir / "multilayer_cka.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Save JSON summary for easy aggregation
    json_path = seed_dir / "multilayer_cka_summary.json"
    max_global = max(rows, key=lambda r: r["cka_global"])
    max_trait = max(rows, key=lambda r: r["cka_trait_subspace"])
    summary = {
        "seed": seed_dir.name,
        "n_layers": n_layers,
        "max_global_cka_layer": max_global["layer"],
        "max_global_cka_value": max_global["cka_global"],
        "max_trait_cka_layer": max_trait["layer"],
        "max_trait_cka_value": max_trait["cka_trait_subspace"],
        "mean_global_cka": float(np.mean([r["cka_global"] for r in rows])),
        "mean_trait_cka": float(np.mean([r["cka_trait_subspace"] for r in rows])),
        "median_trait_uplift": float(np.median([r["trait_uplift"] for r in rows])),
        "rho2_trait_at_max_layer": max_trait["cka_trait_subspace"],
        "per_layer_rows": rows,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  max global CKA: layer {max_global['layer']}, {max_global['cka_global']:.3f}")
    print(f"  max trait CKA:  layer {max_trait['layer']}, {max_trait['cka_trait_subspace']:.3f}")
    print(f"  saved: {csv_path.name}, {json_path.name}")
    return summary


def aggregate(base_dir: Path, seeds: list):
    summaries = []
    for s in seeds:
        sjson = base_dir / f"seed{s}" / "multilayer_cka_summary.json"
        if not sjson.exists():
            print(f"  [skip] missing {sjson}")
            continue
        with open(sjson) as f:
            summaries.append(json.load(f))

    if not summaries:
        print("No per-seed summaries found. Run per-seed first.")
        return

    print(f"\n[aggregate over {len(summaries)} seeds]")
    fields = [
        "max_global_cka_value",
        "max_trait_cka_value",
        "mean_global_cka",
        "mean_trait_cka",
        "median_trait_uplift",
        "rho2_trait_at_max_layer",
    ]
    agg = {}
    for f in fields:
        vals = [s[f] for s in summaries]
        agg[f] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "values": vals,
        }
    # Per-layer mean across seeds
    n_layers = summaries[0]["n_layers"]
    per_layer_agg = []
    for ell in range(n_layers):
        gvals = [s["per_layer_rows"][ell]["cka_global"] for s in summaries]
        tvals = [s["per_layer_rows"][ell]["cka_trait_subspace"] for s in summaries]
        per_layer_agg.append({
            "layer": ell,
            "cka_global_mean": float(np.mean(gvals)),
            "cka_global_std": float(np.std(gvals, ddof=1)) if len(gvals) > 1 else 0.0,
            "cka_trait_mean": float(np.mean(tvals)),
            "cka_trait_std": float(np.std(tvals, ddof=1)) if len(tvals) > 1 else 0.0,
        })

    out = {
        "n_seeds": len(summaries),
        "seeds": [s["seed"] for s in summaries],
        "summary_stats": agg,
        "per_layer_aggregate": per_layer_agg,
    }
    out_path = base_dir / "multilayer_cka_aggregate.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # Pretty print key numbers for paper
    print(f"\n=== AGGREGATE (n={len(summaries)} seeds) ===")
    for f in fields:
        a = agg[f]
        print(f"  {f:32s}  {a['mean']:.4f}  (std {a['std']:.4f})")
    print(f"\n=== Per-layer trait-subspace CKA (rho^2 proxy) ===")
    print(f"  {'layer':>6} {'trait_mean':>12} {'trait_std':>11} {'global_mean':>13}")
    for r in per_layer_agg:
        print(f"  {r['layer']:>6d} {r['cka_trait_mean']:>12.4f} "
              f"{r['cka_trait_std']:>11.4f} {r['cka_global_mean']:>13.4f}")
    print(f"\nSaved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_dir", type=str, default=None,
                        help="Single seed directory (per-seed mode)")
    parser.add_argument("--aggregate", action="store_true",
                        help="Aggregate mode: combine per-seed results")
    parser.add_argument("--base_dir", type=str, default=None,
                        help="Base dir for aggregate (contains seed0/, seed1/, ...)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    args = parser.parse_args()

    if args.aggregate:
        assert args.base_dir, "--base_dir required for --aggregate"
        aggregate(Path(args.base_dir), args.seeds)
    else:
        assert args.seed_dir, "--seed_dir required for per-seed mode"
        compute_per_seed(Path(args.seed_dir))


if __name__ == "__main__":
    main()
