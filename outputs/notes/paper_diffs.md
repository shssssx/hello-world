# Pending PAPER.md diffs — apply when seq=49 sgd_pressure.json lands

These are paper edits that depend on the SGD pressure-test results numerically,
but whose *prose* can be locked now. Both go into §5.5. Hold until JSON lands;
apply together with Table 3 + abstract update in a single commit.

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
