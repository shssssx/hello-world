# v1a scale/norm sanity

Re-ran the coarse all-heads token-table swap with a per-feature affine rescale of V_intervention to match real-V mean/std (eval-set stats). If the inverted-U (mid layers >> edge layers) survives rescaling, the v0 signal is not a LayerNorm-OOD scale artifact.

| layer | v0 delta | rescaled delta |
|---|---|---|
| L0 | +0.0000 | +0.0000 |
| L1 | +0.3209 | +0.2346 |
| L2 | +0.0878 | +0.0672 |
| L3 | +0.3426 | +0.2518 |
| L4 | +0.1255 | +0.0934 |
| L5 | +0.6230 | +0.5257 |
| L6 | +0.3236 | +0.5682 |
| L7 | +0.2651 | +0.4683 |
| L8 | +0.3667 | +0.4060 |
| L9 | +0.2248 | +0.2088 |
| L10 | +0.3811 | +0.4531 |
| L11 | +0.4536 | +0.3843 |
| L12 | +0.4071 | +0.2914 |
| L13 | +0.3866 | +0.2460 |
| L14 | +0.3348 | +0.1877 |
| L15 | +0.2702 | +0.1563 |
| L16 | +0.4191 | +0.2790 |
| L17 | +0.5728 | +0.1846 |
| L18 | +0.3215 | +0.1017 |
| L19 | +0.3774 | +0.1082 |
| L20 | +0.1679 | +0.0703 |
| L21 | +0.1740 | +0.0440 |
| L22 | +0.0942 | +0.0426 |
| L23 | +0.0775 | +0.0416 |

- mid (L5/11/17) mean delta:  v0=+0.5498  rescaled=+0.3649
- low (L20-23) mean delta:    v0=+0.1284  rescaled=+0.0496

**Verdict: inverted-U SURVIVES rescaling -> scale is NOT the main cause; v1a uses RAW token-table V (no rescale).**
