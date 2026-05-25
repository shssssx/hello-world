# v1a — single-layer low-rank V-path correction (summary)

Pythia-410m-deduped, fp16, single RTX 4090. Eval = the exact 1000 Pile-train
blocks (ctxlen 1024) used in v0; train = the *next* 1000 blocks (no overlap).
Baseline next-token CE recomputed each run = **2.1177** (v0: 2.1178, assert |Δ|<0.005 ok).

Goal: for layers where token-grounding V is costly, how much does a trainable
low-rank correction recover, and is a **shared per-layer** LoRA as efficient as a
**per-head** LoRA (Reading A: shared low-rank signal vs Reading B: heterogeneous)?

> Status: this is a **pilot on L5 and L11 only** (the v0 extremes). L17/L23 were
> intentionally NOT run — the probes below changed the question (see Q2). Treat
> recovery numbers as a characterization of the *parameterization*, not a final
> architectural ceiling.

## Correction

For target layer `l`, the intervened V = token-table V + low-rank correction:

```
shared : V_l(t)     = W_V·LN_l(E[x_t]) + (α/r)·LN_l(h_{l-1,t}) @ A_l @ B_l
per-head: V_{l,h}(t)= ...[h]           + (α/r)·LN_l(h_{l-1,t}) @ A_{l,h} @ B_{l,h}
```

`LN_l(h_{l-1,t})` is exactly the input to layer l's `query_key_value` (GPT-NeoX
applies input_layernorm before attention). Only A,B trainable; backbone frozen;
forward NOT under no_grad. α=r (scale=1). AdamW, wd=0, batch 8, ctxlen 1024.

## Deviations from the original v1a spec (and why)

1. **A init**: spec said `std=1/sqrt(r)`. For A:[d_model,r] the contraction dim
   is d_model, so that gives `Var(LN(h)@A)=d_model/r≈128` (std≈11 at r=8) and
   diverges. Corrected to fan-in init **`std=1/sqrt(d_model)`** (standard LoRA).
2. **lr**: spec said 1e-3 (constant). At corrected init, 1e-3 still **diverges on
   L11** (eval→3–8). The stable grid below uses **lr=3e-5** (still constant AdamW,
   no scheduler). Probes then varied lr/steps/clip/normcap as diagnostics.
3. **`--mode all` variant bug**: argparse default `--variant "shared"` made the
   grid loop run shared only; fixed to default "" so both variants run.
4. Dead-stuck guard: B=0 init makes `A.grad=0` at step 0 (legitimate); guard now
   checks A+B combined gradient.

## Scale / norm sanity (blocking pre-step)

Re-ran the v0 coarse all-heads swap with a per-feature affine rescale of the
token-table V to match real-V mean/std. The inverted-U **survives** (mid L5/11/17
rescaled mean 0.365 ≫ low L20-23 0.050) → the v0 signal is NOT a LayerNorm-OOD
scale artifact. **v1a uses raw token-table V, no rescale.** (see `scale_sanity.md`)

## Stable grid (lr=3e-5, 500 steps) — recovery ratio

```
L5  shared : r2 .070  r4 .113  r8 .179  r16 .200  r32 .232
L5  perhead: r2 .079  r4 .120  r8 .183  r16 .236  r32 .219
L11 shared : r2 .019  r4 .027  r8 .032  r16 .047  r32 -.061
L11 perhead: r2 .013  r4 .026  r8 .041  r16 .035  r32 -.048
```

- **L5**: recovery rises monotonically with rank to ~0.23; shared ≈ per-head at
  equal rank, but per-head costs ~8.5× the params.
- **L11**: recovery ≈0 (≤0.05); r32 even goes negative (high-rank instability).

## Stability probe — why L11 fails (undertraining vs landscape)

Per-50-step diagnostics (`||ΔV||/||V||`, held CE, grad/‖B‖) on L11 shared:

| config | final rec | mechanism |
|---|---|---|
| r8 lr3e-5 500 | +0.041 | stable, dV/V→0.11 |
| r8 lr3e-5 **2000** | −0.60 | dV/V→0.85, held **rises** while train CE falls |
| r8 lr1e-4 **+clip 1.0** | −0.22 | clip does not prevent it |
| r32 (all) | −0.09…−0.74 | worse |

**Not undertraining** (more steps → worse), **not fixable by lr or grad-clip**.
Train CE drops (2.7→2.0) while held/eval does not improve → **overfit /
unbounded-correction divergence**, see `probe_diagnostics.png`.

## Norm-capped probe — decisive (optimization vs architecture)

Hard-cap `||ΔV|| ≤ cap·||V||` removes the divergence entirely:

```
L11 r8  lr1e-4 cap.15 : rec +0.030      L5 r8  lr1e-4 cap.15 : rec +0.238
L11 r16 lr1e-4 cap.15 : rec +0.020      L5 r16 lr1e-4 cap.15 : rec +0.266
L11 r16 lr3e-4 cap.10 : rec +0.029
```

Capping stabilizes training (dV/V pinned at the cap, higher lr now safe). L5
recovers ~0.24–0.27; **L11 stays ~0.02–0.03 even when training is perfectly
stable.** So L11's near-zero recovery is **not** an optimization artifact — it is
a ceiling of the low-rank additive linear correction. (`cap_probe.png`)

## Answers

**Q1 — Reading A vs B (for recoverable layers, i.e. L5):**
shared ≈ per-head on *raw* recovery (r8: .179 vs .183; r16: .200 vs .236) while
per-head uses ~8.5× params. Param-normalized, **shared dominates → Reading A: the
recoverable V-contextualization is a shared low-rank signal**, not a per-head
heterogeneous one. (Tentative: one recoverable layer.)

**Q2 — is low-rank correction effective?** **Depth-dependent.**
- Mid layer (L5, v0 delta 0.62): **yes** — ~24–27% recovered at r≤16, shared,
  generalizes to held data, stable under capping.
- Deep layer (L11, v0 delta 0.45): **no** — ceiling ~3% even with stable, capped,
  higher-lr training. The missing V-contextualization at L11 is **not a low-rank
  (≤32) linear function of LN_l(h)**; it is higher-rank / not linearly read-out.

This mirrors v0's "distributed redundancy": deeper layers carry V-contextualization
in a higher-rank, more distributed form that a small shared linear adapter cannot
reconstruct.

## Caveats

- Only L5, L11 (v0 extremes); L17/L23 not run.
- Ranks ≤32 (≈1/32 of d_model); "not low-rank-recoverable" means at r≤32, not
  literally impossible at full rank.
- train and eval are both Pile-train (different blocks); recovery is a paired
  loss-delta so train/test distribution match is fine, but absolute losses are
  in-distribution.
- Stable-grid lr=3e-5 likely slightly undertrains L5 (cap probe at lr1e-4 gives
  L5 ~0.27 > grid 0.20), so L5 recovery here is a mild lower bound.

## Recommendation for v1b / redesign

- A simple low-rank additive linear correction is sufficient for mid layers but
  hits a representational wall at deep layers. Before extending to all 24 layers,
  the correction should be redesigned for deep layers: higher rank, a **gated**
  correction, or a norm-constrained / nonlinear read-out. Norm-capping (cap≈0.15)
  should be the default training stabilizer (it is a stability device, not an
  architecture trick) so higher lr can be used.
- The shared-vs-per-head verdict (Reading A) should be re-confirmed on a second
  recoverable layer once the deep-layer correction is fixed.

Artifacts: `recovery_curves.png`, `probe_diagnostics.png`, `cap_probe.png`,
`L{5,11}_{shared,perhead}_r*.json` (grid), `probe_*.json` (stability + cap),
`scale_sanity.md`.
