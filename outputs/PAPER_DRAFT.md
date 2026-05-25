# Most of the attention value path is a token lookup; its context residual is linearly present but SGD-unlearnable

*Working draft. Model: EleutherAI/pythia-{160m,410m}-deduped. Corpus: monology/pile-uncopyrighted.*

## Abstract

We study what the attention **value** (V) path contributes to a language model's
next-token loss, and whether that contribution is token-determined or genuinely
context-dependent. Replacing a layer's V with a token-only "table" value costs
0.08–0.62 nats per layer in Pythia-410M (an inverted-U over depth), which naively
suggests broad use of value-side contextualization. We show this reading is largely an
artifact of a **weak anchor**: a directly *fitted* per-token value table recovers
59–88% of that loss — both the table and the original token-grounded value are
context-free, so most of the apparent "contextualization" is just that the
embedding-projected table `W_V·LN(E[x])` poorly approximates the per-token mean value.
The genuinely context-dependent residual is **concentrated in mid-stack layers**
(peaking where the per-token table is weakest) and is much smaller than the raw numbers
imply. Strikingly, this residual is a **generalizing linear function of the same hidden
state that produced V** — a closed-form ridge readout recovers 71–86% of it — yet
**SGD-trained low-rank adapters fail to find it** (≈0% over the fitted anchor; random
init even hurts), across rank, nonlinearity, learning rate, gradient clipping and norm
budget. A closed-form ridge solution, truncated to rank 64 and injected as a LoRA
adapter, recovers the residual **with no training**. The phenomenon is therefore an
**optimization pathology, not a representation bottleneck**, and it **replicates across
two Pythia scales** (160M and 410M) with the depth profile shifting but not changing
character.

## 1. Introduction

Attention mixes information across positions through the value path: each position's
output is a weighted sum of value vectors `V_t = W_V·LN_l(h_{l-1,t})`. Because
`h_{l-1,t}` is itself contextualized by earlier layers, V carries cross-position
context. A natural diagnostic question is **how much** of a layer's value path is doing
genuine contextualization versus simply emitting a per-token value.

We answer this with a sequence of interventions, each falsifying the lazy reading of the
previous one, and arrive at a method-level claim plus a negative result about gradient
training. Our contributions:

1. **Decomposition.** The V-path loss decomposes into a large token-determined part
   (recovered by a fitted per-token value table) and a smaller, mid-stack-concentrated
   context residual. v0-style ablations that only swap in an embedding-projected table
   conflate the two and overstate "value contextualization."
2. **Linear recoverability vs. trainability gap.** The context residual is a
   generalizing low-rank linear function of the current post-LN hidden state (closed-form
   ridge recovers 71–86%), but SGD-trained low-rank/MLP adapters reading the same input
   do not find it — a clean optimization-vs-representation dissociation.
3. **Training-free correction.** Factoring the ridge solution into LoRA factors yields a
   deployable, zero-training value correction that reproduces the ridge recovery.
4. **Cross-scale replication.** All three reproduce on Pythia-160M and 410M.

## 2. Setup

**Models.** pythia-410m-deduped (24 layers, d=1024, 16 heads) and pythia-160m-deduped
(12 layers, d=768, 12 heads), GPT-NeoX, fp16, single GPU. **Data.** Pile (train split),
tokenized into 1024-token blocks. **Three-way disjoint split:** eval = blocks[0:1000]
(fixed; metric = mean next-token CE), validation = blocks[1000:2000] (hyperparameter /
λ selection), calibration = blocks[2000:6000] (anchor + ridge fitting). Baseline CE:
410M 2.118, 160M 2.994.

**Intervention mechanism.** A forward hook on a layer's fused `query_key_value` rewrites
only the V slice (Q, K untouched; layout verified against GPT-NeoX's per-head
`[3·head_dim]` interleaving), leaving all other layers intact. We measure the CE delta
vs. the unmodified baseline.

**Anchors and corrections.**
- `A0` (token table): `W_V·LN_l(E[x_t])` — value from the raw token embedding.
- `A1` (fitted token table): `E[V^real_l | token=x]`, estimated on calibration.
- correction: `V = anchor + (α/r)·LN_l(h)·A·B`, A,B low-rank; or a closed-form ridge
  map `W=(XᵀX+λI)⁻¹XᵀY`, X=LN_l(h), Y=V_real−anchor.

Recovery metrics: `R_total = 1 − Δ_method/Δ_A0`; `R_context = (Δ_A1 − Δ_method)/Δ_A1`
(recovery of the residual that remains after the fitted anchor).

## 3. The value path is mostly a token lookup

**v0 (410M).** Replacing a whole layer's V with A0 costs 0.08–0.62 nats (inverted-U,
peak L5). Per single head, 97.7% of 384 heads have |delta|<0.05 — the cost is
distributed, no single head is load-bearing.

**Anchor audit.** A0 and A1 are both context-free per-token values, yet A0 recovers 0%
while A1 recovers 0.59–0.87 across layers. Most of the v0 cost is therefore a poor anchor,
not contextualization. The genuine residual (1−A1) is U-shaped in depth — ~0.13 at
early/late layers, peaking mid-stack (410M: L6/L7, A1≈0.22; 160M: L4/L5, A1 0.39/0.67).
An SVD of `V_real−A1` is high-rank and near-identical across layers, so depth-varying
rank does not explain the residual differences.

## 4. Linearly present, but SGD cannot learn it

**Trained corrector fails.** A low-rank or MLP adapter reading LN_l(h), trained by SGD to
correct V on top of A0 or A1, recovers ~3% (A0) / +0% (A1) of the residual — invariant to
rank (2–256), nonlinearity, lr, grad-clip, step count, and a 0.15 norm cap. This looks
like a representation ceiling.

**Closed-form ridge refutes it.** A ridge map of the same input recovers 71–86% of the
residual on held data (410M; full-rank R_total 0.89–0.98). The information is linearly
present in LN_l(h) and generalizes; SGD simply did not find it.

**Two culprits.** (i) The 0.15 norm cap is far too tight — capped ridge recovers only
0.32–0.46 vs 0.71–0.86 uncapped; the required correction norm exceeds it. (ii) Even with
the cap relaxed to where the solution fits (0.5), **random-init SGD still fails** (R_context
L6 0.32, L7 0.06, L11 −0.34) — a genuine loss-landscape pathology, not just a budget issue.

## 5. A training-free deployable correction

Factoring the ridge W into LoRA factors `A=U_rS_r^{1/2}, B=S_r^{1/2}V_rᵀ` and injecting
through the real adapter path **reproduces ridge zero-shot** (410M r64 R_context L6 .78 /
L7 .75 / L11 .58, matching the offline ridge to 3 dp — no path mismatch). CE finetuning
on top adds ~0 (closed-form is already near-optimal). Robustness: R_context rises gently
with calibration size and plateaus, and is already substantial at n=250 sequences — the
1024×1024 ridge map is not merely saturating ~1M calibration tokens.

## 6. Cross-scale replication (Pythia-160M)

All three claims reproduce at 160M (12 layers): A1 recovers 0.76–0.92 at most layers with
a mid-stack dip (L4 0.39, L5 0.67); r64 ridge recovers R_context 0.84 at the most
context-bound layer L4; random-init SGD there gives −0.17 vs. ridge-init 0.73. The depth
profile shifts in index (mid-stack at L4/L5 vs. L6/L7) but not in character. Verdict:
**observed across Pythia scales.**

## 7. Discussion

The headline is a dissociation: for deep value contextualization, **the recoverable
structure exists, is low-rank, and is linear in the obvious input**, yet gradient descent
from a random initialization does not reach it — and a too-conservative norm constraint,
adopted precisely to keep SGD stable, structurally precludes it. This cautions against
reading "a trained adapter fails to recover X" as "X is not represented in the input."
Methodologically, a closed-form anchor+ridge factorization is a cheap, training-free
value correction and a tight diagnostic upper bound for trainable variants.

## 8. Limitations

Two model scales, one corpus; no >410M check. The fitted per-token anchor A1 is a
vocab×d table — a deployment cost we treat as diagnostic, not necessarily the final form.
The deployable claim rests on the rank-64 truncation of a d×d ridge map (retains most of
the recovery). We did not isolate *why* SGD fails (sharpness, conditioning, init);
characterizing that landscape is future work.

## 9. Conclusion

Most of the attention value path's loss contribution is a token lookup; the genuine
context residual is mid-stack-concentrated, linearly present in the current hidden state,
and recoverable in closed form — but not by SGD. The result is a method (training-free
ridge-init value correction) and a cautionary negative result (optimization, not
representation, was the bottleneck), replicated across two model scales.

*Artifacts: `outputs/` (v0), `outputs/v1a/` (anchor/ridge diagnostics, depth profiles),
`outputs/v1b_ridge/` (ridge / zero-shot / finetune / calibration scaling),
`outputs/v1b_160m/` (cross-scale), `outputs/NARRATIVE_REPORT.md` (dense findings).*
