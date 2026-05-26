#!/usr/bin/env python3
"""
Geometric test of Assumption 1 via trait-shift PCA.

Computes delta_i = h_teacher_w[layer L][i] - h_base[layer L][i] for each
probe, averaged across seeds. Tests:
  1. Magnitude separation: ||delta||_2 on trait probes vs neutral probes
  2. PCA: variance explained by top components of delta matrix
  3. Cross-seed direction stability: cosine similarity of per-seed top-1 PCs

Output: results/probe_assumption1/shift_geometry.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from numpy.linalg import norm, svd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_npz", required=True)
    parser.add_argument("--teacher_npz_list", nargs="+", required=True)
    parser.add_argument("--output", default="results/probe_assumption1/shift_geometry.json")
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="Subset of layers to analyze (default: all)")
    args = parser.parse_args()

    base = np.load(args.base_npz)
    h_base = base["base_hidden"].astype(np.float64)   # (L+1, n_probes, d)
    labels = base["probe_labels"]
    trait_idx = np.where(labels == 1)[0]
    neutral_idx = np.where(labels == 0)[0]

    teacher_h_list = []
    for f in args.teacher_npz_list:
        td = np.load(f)
        teacher_h_list.append(td["teacher_hidden"].astype(np.float64))

    n_layers = h_base.shape[0]
    layers = args.layers or list(range(n_layers))

    per_layer = {}
    for ell in layers:
        # delta_i (per seed): h_teacher_seed[ell][i] - h_base[ell][i]
        deltas_per_seed = [th[ell] - h_base[ell] for th in teacher_h_list]
        # Seed-averaged shift:
        delta_avg = np.mean(deltas_per_seed, axis=0)   # (n_probes, d)

        # 1. Magnitude separation
        mag_trait = np.linalg.norm(delta_avg[trait_idx], axis=1)
        mag_neutral = np.linalg.norm(delta_avg[neutral_idx], axis=1)
        from scipy.stats import mannwhitneyu
        mwu_stat, mwu_p = mannwhitneyu(mag_trait, mag_neutral, alternative="greater")

        # 2. PCA on seed-averaged delta
        delta_centered = delta_avg - delta_avg.mean(axis=0, keepdims=True)
        U, S, Vt = svd(delta_centered, full_matrices=False)
        var_explained = (S ** 2) / (S ** 2).sum()
        cumvar = np.cumsum(var_explained)

        # 3. Cross-seed direction stability (top-1 PC per seed, compare)
        per_seed_top_pcs = []
        for d_seed in deltas_per_seed:
            d_c = d_seed - d_seed.mean(0, keepdims=True)
            _, _, Vt_s = svd(d_c, full_matrices=False)
            per_seed_top_pcs.append(Vt_s[0])
        # Pairwise abs cosine similarity
        sims = []
        for i in range(len(per_seed_top_pcs)):
            for j in range(i + 1, len(per_seed_top_pcs)):
                cs = abs(np.dot(per_seed_top_pcs[i], per_seed_top_pcs[j]))
                sims.append(float(cs))

        per_layer[ell] = {
            "magnitude_trait_mean":    float(np.mean(mag_trait)),
            "magnitude_trait_std":     float(np.std(mag_trait, ddof=1)),
            "magnitude_neutral_mean":  float(np.mean(mag_neutral)),
            "magnitude_neutral_std":   float(np.std(mag_neutral, ddof=1)),
            "magnitude_ratio":         float(np.mean(mag_trait) / max(np.mean(mag_neutral), 1e-12)),
            "magnitude_mwu_pvalue":    float(mwu_p),
            "var_explained_pc1":       float(var_explained[0]),
            "var_explained_top3":      float(cumvar[2]) if len(cumvar) >= 3 else float(cumvar[-1]),
            "var_explained_top5":      float(cumvar[4]) if len(cumvar) >= 5 else float(cumvar[-1]),
            "var_explained_top10":     float(cumvar[9]) if len(cumvar) >= 10 else float(cumvar[-1]),
            "cross_seed_pc1_cosine":   {"values": sims, "mean": float(np.mean(sims)) if sims else 0.0},
        }

        print(f"layer {ell:2d}:  ||δ||_trait/||δ||_neutral = {per_layer[ell]['magnitude_ratio']:.3f}  "
              f"(MWU p={mwu_p:.3e}) | "
              f"PC1 var = {var_explained[0]*100:.1f}% | "
              f"cross-seed cos = {per_layer[ell]['cross_seed_pc1_cosine']['mean']:.3f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(per_layer, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
