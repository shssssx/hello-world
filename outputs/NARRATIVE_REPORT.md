# The attention value path is mostly a token lookup, plus a context residual that SGD can't learn but ridge can

**Model:** EleutherAI/pythia-410m-deduped (GPT-NeoX, 24 layers, d_model=1024, 16 heads), fp16, single RTX 4090.
**Data:** monology/pile-uncopyrighted (train split). Eval = 1000×1024-token blocks (held fixed throughout); calibration/training use disjoint blocks. Baseline next-token CE = **2.1178**.

## Thesis

We decompose each layer's attention value V into a **token-determined** component and a
**context-dependent** component. A fitted per-token value anchor recovers most of the
value-path loss but leaves a **middle-layer context residual**. That residual is
**linearly recoverable from the very hidden state that produced V** — yet SGD-trained
low-rank adapters fail to find it. A closed-form ridge solution, truncated to low rank
and injected as a LoRA adapter, recovers it **with no training**, revealing an
**optimization pathology, not a representation bottleneck**.

The result is built as a falsification chain: each step refutes the lazy reading of the
previous one.

## 1. v0 — token-grounding V is costly, distributed across heads

Intervention: replace a layer's V with a token-only "table" value
`V_table(x_t) = W_V·LN_l(E[x_t])` (the value you'd get from the raw token embedding
through the layer's own pre-attention LayerNorm + value projection), leaving all other
layers intact. Measure next-token CE delta.

- Per **layer** (all 16 heads replaced): delta 0.08–0.62 nats, an inverted-U peaking
  mid-stack (median 0.32, max 0.62 at L5).
- Per **(layer, head)** (one head replaced): **97.7% of all 384 heads have |delta|<0.05**
  (median 0.012). No single head is load-bearing; the cost is distributed.

Lazy reading: *"V contextualization is broadly important, especially mid-stack."*

## 2. Anchor audit — most of that cost is a weak anchor, not context

`V_table` and a **fitted per-token mean** `A1(x)=E[V_l^real | token=x]` (estimated on a
sequence-disjoint calibration set, 97% eval coverage) are *both* context-free per-token
values — yet:

| layer | A0 (=V_table) recovery | A1 (fitted token table) recovery |
|---|---|---|
| L5  | 0.00 | 0.87 |
| L11 | 0.00 | 0.59 |
| L17 | 0.00 | 0.87 |

So **most of v0's cost is just that `W_V·LN(E[x])` is a poor approximation of the
optimal per-token V** — not contextualization. The genuine context-dependent residual
(1−A1) is **U-shaped in depth**: ~0.13 at early/late layers, but concentrated in mid
layers, **peaking at L6/L7 (A1≈0.22, residual ~0.78)**. The original 4-layer snapshot had
missed the L6/L7 trough.

Lazy reading refuted: V's loss contribution is mostly token-determined; genuine context
is small and concentrated mid-stack (L6/L7).

## 3. Trained corrector — fails on the context residual

A small low-rank/MLP adapter reading `LN_l(h)` (the layer's contextualized hidden state),
trained by SGD to correct V on top of A0 or A1: recovers **~3% (A0 base) / +0% over A1
(A1 base)** of the residual, across rank 2–256, linear and nonlinear, with/without grad
clip, more steps, and a 0.15 norm cap.

Lazy reading: *"the context residual is not recoverable from `LN_l(h)` — a representation
/ architectural ceiling."*

## 4. Closed-form ridge — refutes the ceiling; it's linearly recoverable

Fit a closed-form ridge map `W=(XᵀX+λI)⁻¹XᵀY`, X=`LN_l(h)`, Y=`V_real − A1`, inject
`V=A1 + X·W`, eval CE on held data:

| layer | A1 alone | full-rank ridge R_total | ridge **R_context** |
|---|---|---|---|
| L5  | 0.87 | 0.98 | 0.81 |
| L6  | 0.23 | 0.89 | 0.86 |
| L7  | 0.22 | 0.89 | 0.86 |
| L11 | 0.59 | 0.91 | 0.77 |

A linear map recovers **71–86% of the context residual at every layer, generalizing to
held data**. So the residual **is** a linear function of `LN_l(h)` — the SGD failure was
not representational. (An SVD spectrum of `V_real−A1` is high-rank and near-identical
across L5/L11, ruling out "deeper = higher rank" as the discriminator.)

## 5. The win is closed-form, and SGD genuinely can't reach it

Factor ridge W into LoRA factors `A=U_rS_r^½, B=S_r^½V_rᵀ` (so AB=W_r) and inject through
the real adapter path:

- **Zero-shot ridge-init reproduces ridge exactly** (r64 R_context: L6 .78, L7 .75,
  L11 .58) → deployable low-rank adapter, no path mismatch.
- **CE finetune adds ~0** over ridge-init (L6 .783→.784) → closed-form is already optimal.
- **Random-init + CE finetune fails even at a relaxed cap 0.5** (L6 .32, L7 .06, L11
  **−.34**) → SGD from random init cannot find the solution ridge gets in closed form.

The earlier failures were partly a too-tight norm cap (0.15; capped ridge only recovers
0.32–0.46 vs 0.71–0.86 uncapped) **and** a genuine loss-landscape pathology: even with the
cap relaxed to where the solution fits, SGD does not reach it.

## 6. Method

A **training-free** value-path correction: a fitted per-token anchor A1 plus a low-rank
**ridge readout of `LN_l(h)`**, deployed as a LoRA adapter. It recovers most of the
value-path loss the token table discards (deep mid-layers included), in closed form.
The contribution is as much **negative-about-SGD** as positive: the recoverable structure
exists and is low-rank and linear, but gradient training does not find it.

## Robustness — calibration-size scaling

Three-way disjoint split (eval=blocks[0:1000], validation=blocks[1000:2000] for λ
selection, calibration pool=blocks[2000:6000]). r64 R_context vs calibration size
(`ridge_scale.png`, `ridge_scale.json`):

| calib seqs | L5 | L6 | L7 | L11 |
|---|---|---|---|---|
| 250  | 0.44 | 0.73 | 0.69 | 0.52 |
| 500  | 0.48 | 0.76 | 0.72 | 0.54 |
| 1000 | 0.53 | 0.79 | 0.75 | 0.58 |
| 2000 | 0.56 | 0.80 | 0.76 | 0.60 |
| 4000 | 0.58 | 0.81 | 0.77 | 0.61 |

R_context **rises gently with calibration size and plateaus** (e.g. L6 gains +0.06 from
250→1000 but only +0.02 from 1000→4000); it is already **substantial at n=250** (L6 0.73),
far below the 1024×1024 ridge map's ~1M parameters. So the recovery is **not** a saturation
artifact of ~1M calibration tokens fitting ~1M parameters — it generalizes from a few
hundred sequences and saturates. λ is selected on a disjoint validation set, so it is not
tuned on the reported eval. (cap0.3 follows the same shape, ~0.06–0.07 lower.)

## Cross-scale replication on Pythia-160M

*(placeholder — to be filled from `v1b_160m/repro160.json`.) Minimal replication of the
three core claims on pythia-160m-deduped (12 layers, d=768), same Pile token blocks
(shared tokenizer), λ selected on a disjoint validation set:*
1. *A1 token anchor recovers much of the per-layer V-path loss;*
2. *a mid-layer context residual remains (depth profile of A1 recovery);*
3. *closed-form ridge (r64) recovers the residual from LN_l(h), reproduced zero-shot by a
   ridge-init LoRA adapter, while random-init SGD is much weaker.*

*Verdict to record: full replication (→ "observed across Pythia scales"), partial
(→ "decomposition reproduces, depth profile shifts with scale"), or failure
(→ scale-dependent emergence; downgrade to 410M case study).*

## Limitations

- Single model (pythia-410m), single corpus (Pile). No cross-model / cross-dataset check.
- A1 is a per-token table (vocab×d) — a real deployment cost; here it is a diagnostic /
  oracle-ish anchor, not necessarily the final deployable form.
- The ridge map is d×d; the deployable claim rests on its low-rank (r64) truncation, which
  retains most of the recovery (e.g. L6 r64 0.78 vs r256 0.84).

## Artifacts

`outputs/` (v0): `v_intervention.py`, `coarse_loss_delta.npy`, `fine_loss_delta.npy`,
`heatmap.png`, `summary.md`.
`outputs/v1a/`: `v1a_correction.py`, `summary.md`, `anchor_audit_full24.json`,
`depth_profile.png`, plots.
`outputs/v1b_ridge/`: `ridge_results.json`, `ridge_init_zeroshot.json`, `ridge_ft.json`,
`ridge_scale.json`, `ridge_summary.md`, `ridge_depth_probe.png`, `ridge_ft.png`.
