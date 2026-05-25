# v1a — single-layer low-rank V-path correction (summary)

> ## REFRAMING (anchor audit, supersedes the trained-corrector conclusions below)
>
> The trained-corrector chain (below) concluded L11 was ~unrecoverable. An anchor
> audit overturns the strong reading. A **fitted per-token-mean V table** A1(x) =
> E[V_l^real | token=x] (estimated on a sequence-disjoint calibration set, 97%
> eval coverage) recovers, vs the v0 token-table A0 = W_V·LN(E[x]):
>
> | layer | A0 | **A1 fitted token table** | oracle PCA r256 | genuine context residual (1−A1) |
> |---|---|---|---|---|
> | L5  | 0.00 | **0.87** | 0.98 | 0.13 |
> | L11 | 0.00 | **0.59** | 0.92 | 0.41 |
> | L17 | 0.00 | **0.87** | 0.95 | 0.13 |
> | L23 | 0.00 | **0.76** | 0.89 | 0.24 |
>
> **Three corrections to the earlier conclusions:**
> 1. **L11 is NOT incompressible.** A pure per-token table recovers 59% of its loss
>    (L11/L5 ratio 0.67 ≥ 0.6 pre-registered threshold ⇒ "weak anchor", not
>    "incompressible"). The trained low-rank corrector's ~3% was a *learning/
>    parameterization* failure (the oracle, using real V, reaches 0.92 at L11) —
>    not absence of recoverable signal.
> 2. **v0's "V-contextualization cost" is mostly a weak-anchor artifact, not
>    context.** A0 and A1 are *both* context-free per-token values, yet A0 recovers
>    0% and A1 recovers 76–88% (L5/17/23). So most of v0's coarse delta is just that
>    `W_V·LN(E[x])` is a poor stand-in for the per-token-average V — not genuine
>    contextualization.
> 3. **Genuine context-dependence (residual after A1) is small and depth-graded:**
>    ~13% at L5/L17, ~24% at L23, ~41% at L11. It rises with depth but is far below
>    v0's raw numbers, and the oracle shows it is largely low-rank.
>
> **v1b factorization:** stronger token anchor (fitted A1-style table) + a small
> correction for the residual context part — NOT more rank on the A0-anchored
> corrector. (`anchor_audit.png`, `anchor_audit.json`)
>
> Everything below is the diagnostic chain that led here; read it as method, with
> the L11 conclusions superseded by this box.


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
stable.** So L11's near-zero recovery is **not** an optimization artifact. (`cap_probe.png`)

## Is the L11 ceiling about linearity or rank? (both ruled out)

Two further capped/stable probes (`L11_ceiling.png`):

- **Nonlinear bottleneck (`mlp`: LN_h→W1→GELU→W2, equal param count):**
  L11 mlp k16/k32 → rec +0.033/+0.035 — identical to linear. **Not linearity.**
- **High-rank sweep (linear, capped):** L11 r64/r128/r256 → +0.023/+0.037/+0.033.
  Flat from r2 to r256 (r256 ≈ 1/4 of d_model). **Not rank/capacity.**

For comparison L5 plateaus at ~0.26 by r16 and stays there to r64. So **L11's
recovery is pinned at ~0.03 across rank 2–256, linear and nonlinear, under stable
capped training** — a robust wall, not tunable away by capacity, nonlinearity,
lr, steps, or clipping.

## Is it intrinsic rank? (no — SVD refutes the tempting explanation)

Covariance spectrum of `V_real − V_table` over 200K tokens (`diff_spectrum.png`,
`svd_diff.json`):

| | L5 | L11 |
|---|---|---|
| rank for 50% var | 56 | 57 |
| rank for 90% var | 334 | 326 |
| participation ratio | 73.6 | 77.9 |
| var(dV) | 553 | 1288 |

**The V-difference is high-rank at BOTH layers, and the spectra are nearly
identical.** So L11's recovery failure is *not* because its V-difference is
higher-rank than L5's — they have the same spectral structure, yet a rank-16
adapter recovers 26% at L5 and 3% at L11. The discriminator is therefore **not
the dimensionality of the V-difference** but whether its *loss-bearing* part is a
**learnable, generalizing low-rank function of LN_l(h)** — which it is at L5 and
is not at L11. (That L5 recovery saturates at ~0.26 by r16 while the V-difference
needs ~330 dims for 90% variance also shows most of the high-rank V-difference is
loss-irrelevant; only a low-rank, learnable slice matters — at L5.)

## Overfitting ruled out (train vs eval recovery, capped)

| config (capped, stable) | eval rec | train rec | gap |
|---|---|---|---|
| L5 shared r16 | 0.227 | 0.264 | 0.04 |
| L11 shared r16 | 0.026 | 0.065 | 0.04 |
| L11 shared r64 | 0.023 | 0.064 | 0.04 |

Under norm-cap the train–eval gap is small for **both** layers, and crucially
**L11 recovers only ~6% even on the training set.** So L11 is not an
overfitting/data problem — a reasonable-magnitude correction simply cannot *fit*
the loss-bearing V-contextualization at L11, even on train. (The large train-CE
drop seen in the *uncapped* probe was ΔV/V→0.85 memorization that does not
generalize.) L5, with the same small gap, fits and generalizes to ~0.23–0.26.

## Answers

**Q1 — Reading A vs B (for recoverable layers, i.e. L5):**
shared ≈ per-head on *raw* recovery (r8: .179 vs .183; r16: .200 vs .236) while
per-head uses ~8.5× params. Param-normalized, **shared dominates → Reading A: the
recoverable V-contextualization is a shared low-rank signal**, not a per-head
heterogeneous one. (Tentative: one recoverable layer.)

**Q2 — is low-rank correction effective?** **Depth-dependent.**
- Mid layer (L5, v0 delta 0.62): **yes** — ~24–27% recovered at r≤16, shared,
  generalizes to held data, stable under capping.
- Deep layer (L11, v0 delta 0.45): **no** — ceiling ~3% that is **not** moved by
  rank (2→256), nonlinearity (mlp), lr, steps, clipping, or norm-capping. The
  useful V-contextualization at L11 is essentially **not recoverable by any trained
  read-out of the current layer's post-LN hidden state LN_l(h)** under this setup.

The SVD (below) refutes the simplest explanation: the V-difference is high-rank at
BOTH layers (identical spectra), so it is **not** that L11's V-difference is higher
rank. The discriminator is whether the *loss-bearing* slice of the correction is a
learnable, generalizing low-rank function of LN_l(h). Candidate reasons (not
disambiguated here): at deep layers the needed signal is not compactly/learnably
present in the current post-LN hidden state, or the token-table is a fundamentally
lossy anchor deep in the stack.

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

- For **mid layers**, the shared low-rank correction works; v1b can extend it and
  re-confirm Reading A on more recoverable layers. Use **norm-cap (≈0.15) + lr 1e-4**
  as the default stable recipe (cap is a stability device, not an architecture trick).
- For **deep layers**, capacity/nonlinearity/optimization are NOT the bottleneck —
  do not just throw rank or gating at it. The informative next experiments are
  *diagnostic*, not bigger adapters: (a) feed the correction a richer input than
  LN_l(h) (e.g. earlier-layer states, or the actual residual stream), to test
  whether the needed signal is simply absent from LN_l(h); (b) measure the true
  rank of `V_real − V_table` at L11 directly (SVD of the per-token difference) to
  see if it is intrinsically high-rank; (c) check whether the loss is recoverable
  at all by *any* per-token V at L11 (oracle: train an unconstrained per-position V).
- Do **not** run the full 24-layer v1b grid with the current parameterization —
  it would mostly reproduce this L5-good / L11-bad gradient.

Artifacts: `recovery_curves.png`, `probe_diagnostics.png`, `cap_probe.png`,
`L{5,11}_{shared,perhead}_r*.json` (grid), `probe_*.json` (stability + cap),
`scale_sanity.md`.
