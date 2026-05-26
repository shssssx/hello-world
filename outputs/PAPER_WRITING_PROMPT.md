# Prompt for Claude Opus 4.7 — write the paper

Copy everything below the line into a fresh Claude Opus 4.7 session that has access to
this repository (branch `claude/zen-allen-7Y8Bx`). It is self-contained: the model should
read the artifacts, not invent them.

---

You are an experienced ML researcher and writer. Write a complete, submission-quality
paper from the materials in this repository. A draft already exists at `outputs/PAPER.md`
and a dense findings log at `outputs/NARRATIVE_REPORT.md`; treat them as ground truth for
the narrative and numbers, but rewrite for clarity, rigor, and flow rather than lightly
editing. **Do not invent results, numbers, or citations.** Every quantitative claim must
match the artifacts (`outputs/**/*.json`, `*.npy`, `summary.md`); if a number isn't in the
artifacts, don't state it.

## The work (so you don't rediscover it)

Topic: what the attention **value (V)** path contributes to next-token loss in Pythia
(GPT-NeoX), and whether its context content is learnable. The result is a falsification
chain ending in a method + a negative result. The single most important framing: **each
step refutes the lazy reading of the previous one.**

1. Ablating a layer's V to a token-only embedding-projected table `A0 = W_V·LN(E[x])`
   costs 0.08–0.62 nats/layer (inverted-U, 410M); per-head it's nearly free (97.7% of heads
   <0.05). Lazy reading: "value contextualization is broadly important."
2. A *fitted* per-token table `A1 = E[V_real|token]` recovers 0.59–0.87. A0 and A1 are
   **both context-free**, so most of the v0 cost is a **weak anchor**, not contextualization.
   The genuine context residual is small and **mid-stack-concentrated** (410M L6/L7; 160M
   L4/L5). This reframing is the conceptual core — do not bury it.
3. An SGD-trained low-rank/MLP adapter on `LN_l(h)` recovers ~3% (over A0) / +0% (over A1)
   of the residual, across rank/nonlinearity/lr/clip/steps/norm-cap. Lazy reading: "the
   residual isn't recoverable from LN(h) — a representation ceiling."
4. A **closed-form ridge** of the same input recovers 71–86% of the residual on held data
   (generalizing). So it IS linearly present; SGD just didn't find it. **This dissociation
   (linearly present but SGD-unlearnable) is the headline.**
5. The ridge solution, factored into LoRA `A=U_rS^½, B=S^½Vᵀ`, reproduces ridge zero-shot
   (training-free, deployable); CE-finetune adds ~0; random-init SGD fails even at a relaxed
   norm cap (sometimes negative). Culprits: a 0.15 norm cap that's structurally too tight
   (capped ridge 0.32–0.46 vs 0.71–0.86), plus a genuine loss-landscape pathology.
6. Robustness: ridge recovery plateaus in calibration size (substantial at n=250). Replicates
   on Pythia-160M and 410M (depth profile shifts in index, not character).

## What to produce

A single polished paper (`outputs/PAPER_v2.md`, leave the existing `PAPER.md` intact):
abstract, introduction (motivation + contributions), related work, setup, method (the
falsification chain), results (with the existing tables and the 10 figures embedded via
relative paths — figures already exist under `outputs/`), discussion, limitations,
conclusion, reproduction appendix. Target a strong workshop/short-paper length; tighten,
don't pad.

## Rules

- **Lead with the dissociation and the weak-anchor reframing.** A reader should grasp both
  in the abstract.
- **Do not soften the negative result.** "SGD does not find a solution that exists and is
  linearly present" is the point, not a footnote. Frame it as a methodological caution:
  "trained probe fails" ≠ "not represented."
- **Honesty over hype.** Keep the limitations real: two scales, one corpus, A1 is a
  vocab×d table (a deployment cost, treated as diagnostic), rank-64 truncation, we localize
  *that* SGD fails but not *why*. Don't claim generality beyond 160M/410M/Pile.
- **Citations:** the draft marks real-but-unverified works with `[TODO]` (LoRA, Pythia,
  GPT-NeoX, the Pile, mechanistic-interpretability decompositions, activation patching,
  optimization-difficulty literature). Either fill these with correct bibliographic detail
  you are confident about, or keep the `[TODO]` markers — **never fabricate authors, years,
  or venues.** Flag every citation you couldn't verify.
- **Numbers:** cross-check against `outputs/v1a/anchor_audit_full24.json`,
  `outputs/v1b_ridge/{ridge_results,ridge_init_zeroshot,ridge_ft,ridge_scale}.json`,
  `outputs/v1b_160m/repro160.json`, `outputs/baseline_loss.json`. Quote them exactly.
- **Figures:** embed the existing PNGs (heatmap, depth_profile, anchor_audit, diff_spectrum,
  probe_diagnostics, L11_ceiling, ridge_depth_probe, ridge_ft, ridge_scale, repro160_profile)
  with informative captions; don't regenerate or alter them.
- **Tone:** precise, declarative, no filler ("In this paper we...", "It is important to
  note..."). Define notation once. One claim per sentence where it matters.

## Before you write

Read `outputs/PAPER.md`, `outputs/NARRATIVE_REPORT.md`, `outputs/v1a/summary.md`,
`outputs/v1b_ridge/ridge_summary.md`, and the JSONs above. Then write. If something in the
draft and an artifact disagree, the artifact wins — and note the discrepancy at the top of
your output so a human can check.
