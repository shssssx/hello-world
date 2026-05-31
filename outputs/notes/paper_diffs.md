# PAPER.md / paper.tex — by-section status + pending diffs

Paper draft lives in `outputs/notes/paper_draft.tex` (single-author, AAAI 2027
target). Final `paper.tex` is committed at top-level once §5.5 / §5.7 numbers
land. PAPER.md is the prior working-draft; new prose goes in paper_draft.tex.

## By-section status

Updated when blocking experiments land or sections are drafted.

| section          | blocking | drafted | notes |
|------------------|----------|---------|-------|
| Title + Abstract | seq=50 ridge_init_ft outcome | partial (in user message) | "SGD-Unreachable" may weaken to "SGD-Hard" if seq=50 random-init breaks; abstract numbers update with §5.5 |
| §1 Intro         | none     | partial (in user message) | "across ... learning rate" removed from invariance list pending LR probe |
| §2 Related work  | none     | partial (in user message) | TODO bibkeys flagged |
| §3 Setup         | none     | partial (in user message) | §3.4 metrics: append $R_\text{context} \in (-\infty,1]$ clarification |
| §4 Method        | none     | user writing | |
| §5.1 v0 ablation | none     | user writing | numbers stable from §5.1 of PAPER.md |
| §5.2 Anchor      | none     | user writing | numbers stable; sparse-sampling caveat present |
| §5.3 Trained corrector fails | seq=50 outcome | user writing | "~3%/+0%" may change if cap fix shifts trained-corrector behavior; verify post seq=50 |
| §5.4 Closed-form ridge | implicit (seq=50 ridge_init_ft @ L11 ≈ 0.56 confirms ridge_ft.json invariant) | user writing | full-rank uncapped (Fig 7, Table 2) cap=0.15 column: unchanged; cap=0.5 column: predicted unchanged but seq=50 is the verification |
| §5.5 SGD pressure test | **BLOCKED on seq=50** | placeholder | see Diff 1-4 below; ridge_init_ft outcome decides scenario branch |
| §5.6 Calib scaling | none | user writing | numbers stable |
| §5.7 Cross-scale | **BLOCKED on Llama-1B run** | skeleton + placeholder | 160M already in Fig 2 bottom row; Llama-3.2-1B (GQA, 32Q/8KV, d=2048, 16L) hook in `outputs/v1c_llama1b/` (in prep) |
| §6 Discussion    | partial (first paragraph + induction-head section may shift with §5.5) | user writing | most content stable |
| §7 Limitations   | none | user writing | refresh against PAPER.md §7 — half resolved |
| §8 Conclusion    | seq=50 outcome | user writing | last sentence may shift on framing |
| Appendix A       | none | user writing | |
| Appendix B       | none | user writing | |
| Appendix C (LR probe, conditional) | seq=50 outcome → maybe seq=51 | not written | 5-run appendix probe if 60% scenario; 12-run main if 15% scenario |

## seq=50 outcome → scenario branches

Three cases for ridge_init_ft (L6, L7, L11) at lr=1e-4 5000-step with
differentiable cap. Prior 60/25/15.

- **Clean (60%, ~0.5-0.78)**: cap was single root cause.
  - Title stays "SGD-Unreachable" (random-init still ≤0.1 expected)
  - §5.5 Scenario A narrative (in Diff 3 below)
  - 5-run LR probe at L7 random-init optional, drop into appendix
  - ridge_ft.json stays invariant (implicit verification)
- **Partial (25%, ~0.2-0.4)**: cap was main but not sole; lr=1e-4 still bad for ridge basin retention.
  - Title weaken to "SGD-Hard" or "SGD-Unreliable"
  - §5.5 narrative needs custom paragraph: "even ridge-init AdamW drifts at default lr"
  - 5-run LR probe at L7 becomes main-text figure, not appendix
- **No recovery (15%, still <0)**: cap not the root cause.
  - 12-run LR probe (3 lr × 4 layer) is main result, replaces Fig 11
  - Title rewrite: framing flip to "lr-conditional" or weaker
  - Re-read this file before applying any §5.5 diff

---

# Pending §5.5 prose diffs (apply when seq=50 lands + scenario chosen)

Originally drafted assuming seq=49 (lr=1e-4 + soft-cap) numbers; now to be
applied to seq=50 (lr=1e-4 + differentiable cap) numbers. Variant naming
matches `outputs/v1b_ridge/sgd_pressure.json` (post seq=50 push).

## Diff 1 — §5.5 first paragraph: two-ceiling disambiguation

**Purpose.** §5.4 quotes "ridge recovers 71–86 % of R_context" using full-rank
uncapped ridge. §5.5 / Table 3 / Fig 11 use the rank-64 cap-0.5 closed-form
ridge (the `ridge_init_ft` control) as the SGD ceiling. A reader who lands on
Table 3 first will see numerically different "ridge" numbers and suspect
inconsistency. This paragraph names the distinction explicitly, before the
results.

Insert at the end of the first paragraph of §5.5 (right after the
"to pre-empt a natural objection" sentence, before the variant list):

```latex
We compare against the ridge solution under the \emph{same} architecture and
norm budget as the SGD variants ($r{=}64$, $\mathrm{cap}{=}0.5$, A1 base),
obtained by initializing the LoRA factors from the ridge solution and finetuning
(variant \texttt{ridge\_init\_ft}; the finetune barely perturbs the closed-form
weights). This is the apples-to-apples ceiling for the SGD comparison and is
necessarily below the unconstrained ridge ceiling reported in
\S\ref{sec:ridge} (Table~\ref{tab:ridge}, Figure~\ref{fig:ridge-depth}),
which uses full-rank uncapped ridge to test whether the residual is
\emph{representable} at all. Here we test whether SGD can \emph{reach} it.
```

## Diff 2 — §5.5 budget justification

**Purpose.** `--steps 5000` for the extended variants is $2.5\times$ the
baseline budget but still relatively short for a "pressure test". Reviewers
will ask: why not $20{,}000$? Answer pre-emptively with a loss-plateau
argument anchored in already-shown Fig 5 (probe diagnostics curves).

Insert immediately after the variant list, before Table 3 reference:

```latex
We use 5{,}000 training steps for the extended variants ($2.5\times$ the
baseline budget). This is supported by the training-loss curves in
Figure~\ref{fig:sgd-diverges} (\S\ref{sec:probe}): SGD loss plateaus by
$\sim$step $3{,}000$ at all ranks tested, so the additional budget is
sufficient to expose any non-asymptotic improvement.
```

If pilot 10k+ step curves exist (they don't, AFAIK), upgrade to the stronger
version pointing at an Appendix figure of those curves. Current Fig 5 already
shows the plateau, so this version is honest and sufficient.

## Diff 3 — §5.5 result narrative (TWO SCENARIOS, choose one)

### Scenario A — all five SGD variants R_context ≤ 0.30 (expected most likely)

```latex
Table~\ref{tab:sgd-pressure} reports $R_\mathrm{context}$ for each variant at
L6, L7, L11. No random-init recipe recovers more than $X.XX$ of the context
residual that the rank-matched closed-form ridge captures (ceiling
$0.XX$--$0.XX$). Extending the budget ($2{,}000 \to 5{,}000$ steps) yields a
gain of at most $X.XX$; switching optimizer (AdamW $\to$ Lion), adding
linear warmup, and ramping the norm cap on a curriculum all leave the gap
substantially open. The ridge-initialized control reaches the ceiling, as
expected. We conclude that the gap is a property of optimization from
random init, not of representation, rank, or norm-cap calibration.
```

### Scenario B — one variant (most likely `curriculum_cap`) reaches R_context ≥ 0.50

```latex
Table~\ref{tab:sgd-pressure} reports $R_\mathrm{context}$ for each variant at
L6, L7, L11. Most random-init recipes recover less than $0.XX$
($R_\mathrm{context}$ for baseline / long AdamW / warmup / Lion); a single
exception is the curriculum norm-cap schedule, which reaches
$0.XX$--$0.XX$ -- a substantial gain over baseline, but still short of the
$0.XX$--$0.XX$ ridge ceiling at matched rank. The closed-form solution
therefore remains the practical recommendation: it requires no schedule
tuning, no optimizer choice, and matches or exceeds the best SGD recipe in
both accuracy and cost.
```

## Diff 4 — abstract (mirror Scenario A or B)

### Scenario A version

In the abstract's existing §5.5 clause, replace:

> "...the residual is a generalizing linear function of the same hidden state
> that produced V — a closed-form ridge readout recovers 71–86% of it on held
> data — yet random-init SGD-trained low-rank adapters reading that input
> recover ≤ 10% under default settings, ..."

with:

> "...the residual is a generalizing linear function of the same hidden state
> that produced V — a closed-form ridge readout recovers 71–86 % of it on
> held data — yet SGD-trained low-rank adapters reading that input recover
> $\leq X.XX$ even under $2.5\times$ extended training, alternative optimizers
> (Lion), warmup schedules, and curriculum norm budgets
> (§5.5, Table 3), while ridge-initialized adapters match the closed-form
> recovery zero-shot."

### Scenario B version (only if curriculum_cap or similar breaks ≥ 0.50)

> "...closed-form ridge readout recovers 71–86 %; gradient training from
> random initialization fails under default settings ($\leq 0.10$), and even a
> curriculum norm-cap schedule reaches only $0.XX$--$0.XX$ — the closed-form
> solution remains both stricter and substantially cheaper to obtain (§5.5)."

## Application checklist (when JSON lands)

1. `cat outputs/v1b_ridge/sgd_pressure.json | jq .` — verify flat schema as
   expected (per layer dict, no wrappers).
2. `python3 outputs/figs/fig11_sgd_pressure.py` — render fig11.
3. Choose Scenario A vs B by inspecting max non-control variant `R_context`.
4. Fill Table 3 in PAPER.md (6 cells × 3 layers + ceiling row).
5. Apply Diffs 1, 2, 3 above to PAPER.md §5.5.
6. Apply Diff 4 (matching scenario) to abstract.
7. Single commit + push: `paper: §5.5 SGD pressure test + Fig 11 + abstract`.
8. Run `bash outputs/figs/regenerate_all.sh` to verify no regressions in
   other figs (and that the `require_data` skip-logic for any other missing
   inputs behaves cleanly).
