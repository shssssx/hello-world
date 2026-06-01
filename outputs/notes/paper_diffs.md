# PAPER.md / paper.tex — by-section status + pending diffs

## Framing flip log (chronological, most recent first)

### Flip #2 — Cap fix (seq=51 LR probe) is net negative; reverted, now running LR probe (seq=51)

- seq=49 (lr=1e-4, soft_nograd cap): ridge_init_ft = (0.30, -0.23, -0.94) — bad
- seq=51 LR probe (lr=1e-4, differentiable cap, my "fix"): ridge_init_ft = (0.02, -0.55, **-2.00**) — WORSE in every cell, by 2-5x at L7/L11
- Root cause of fix failure: differentiable cap injects a "shrink ||c|| toward 0"
  gradient signal even when cap doesn't bind; on ridge-init this pulls A,B
  AWAY from the closed-form optimum. Mental run on already-converged init
  was not done before pushing the change.
- ridge_ft.json (lr=3e-5, soft_nograd cap): ridge_init_ft = (0.78, 0.74, 0.56) — clean.
  Difference vs seq=49 is lr (3e-5 vs 1e-4); difference vs seq=51 LR probe is also cap formulation.
- Resolution: lr is the dominant variable, cap form is secondary; differentiable cap
  reverted. Now exposed as --cap_mode flag (default soft_nograd) for honest
  reproducibility / Appendix B.
- seq=51 runs `mode_sgd_lr_probe`: 2 inits × 4 lrs × 3 layers = 24 cells. Apples-to-apples
  with ridge_ft at lr=3e-5 (sanity check), and 100× lr range total.

### Flip #1 — sgd_pressure ridge_init_ft control collapsed (seq=49 results read)

- Expected: ridge_init_ft as ceiling ≈ ridge_ft.json values (0.78/0.74/0.56)
- Actual at seq=49 (lr=1e-4): (0.30, -0.23, -0.94)
- Diagnosis: lr=1e-4 + 5000-step AdamW drifts A,B out of ridge basin
- Initial hypothesis (proven wrong): soft_nograd cap = "known pathology"
- Action taken: cap "fix" pushed as seq=51 LR probe — turned out to be Flip #2
- True diagnosis: lr=1e-4 is too high (LoRA default but bad for this objective);
  ridge_ft used lr=3e-5 which is the sweet spot.

---



Paper draft lives in `outputs/notes/paper_draft.tex` (single-author, AAAI 2027
target). Final `paper.tex` is committed at top-level once §5.5 / §5.7 numbers
land. PAPER.md is the prior working-draft; new prose goes in paper_draft.tex.

## By-section status

Updated when blocking experiments land or sections are drafted.

| section          | blocking | drafted | notes |
|------------------|----------|---------|-------|
| Title + Abstract | seq=51 LR probe ridge_init_ft outcome | partial (in user message) | "SGD-Unreachable" may weaken to "SGD-Hard" if seq=51 LR probe random-init breaks; abstract numbers update with §5.5 |
| §1 Intro         | none     | partial (in user message) | "across ... learning rate" removed from invariance list pending LR probe |
| §2 Related work  | none     | partial (in user message) | TODO bibkeys flagged |
| §3 Setup         | none     | partial (in user message) | §3.4 metrics: append $R_\text{context} \in (-\infty,1]$ clarification |
| §4 Method        | none     | user writing | |
| §5.1 v0 ablation | none     | user writing | numbers stable from §5.1 of PAPER.md |
| §5.2 Anchor      | none     | user writing | numbers stable; sparse-sampling caveat present |
| §5.3 Trained corrector fails | seq=51 LR probe outcome | user writing | "~3%/+0%" may change if cap fix shifts trained-corrector behavior; verify post seq=51 LR probe |
| §5.4 Closed-form ridge | implicit (seq=51 LR probe ridge_init_ft @ L11 ≈ 0.56 confirms ridge_ft.json invariant) | user writing | full-rank uncapped (Fig 7, Table 2) cap=0.15 column: unchanged; cap=0.5 column: predicted unchanged but seq=51 LR probe is the verification |
| §5.5 SGD pressure test | **BLOCKED on seq=51 LR probe** | placeholder | see Diff 1-4 below; ridge_init_ft outcome decides scenario branch |
| §5.6 Calib scaling | none | user writing | numbers stable |
| §5.7 Cross-scale | **BLOCKED on Llama-1B run** | skeleton + placeholder | 160M already in Fig 2 bottom row; Llama-3.2-1B (GQA, 32Q/8KV, d=2048, 16L) hook in `outputs/v1c_llama1b/` (in prep) |
| §6 Discussion    | partial (first paragraph + induction-head section may shift with §5.5) | user writing | most content stable |
| §7 Limitations   | none | user writing | refresh against PAPER.md §7 — half resolved |
| §8 Conclusion    | seq=51 LR probe outcome | user writing | last sentence may shift on framing |
| Appendix A       | none | user writing | |
| Appendix B       | none | user writing | |
| Appendix C (LR probe, conditional) | seq=51 LR probe outcome → maybe seq=51 | not written | 5-run appendix probe if 60% scenario; 12-run main if 15% scenario |

## seq=51 outcome → scenario branches (replaces obsolete seq=51 LR probe branches)

24-cell LR probe (2 inits × 4 lrs × 3 layers). Prior 60/25/15.

- **Outcome 1 (60%, ridge_init @ lr=3e-5 ≈ 0.78/0.74/0.56)**: lr was the dominant
  cause; SGD-from-random fails at all 4 lrs while ridge-init is stable at 3e-5.
  - Title: "SGD-Unreachable" (random-init never recovers across 100× lr range)
  - Fig 11: LR sensitivity line plot (x=lr log scale, y=R_context, 2 lines per
    layer for ridge_init vs random_init, 3 subplots per layer)
  - §5.5 narrative: "across lr ∈ [1e-5, 3e-4] (100× range, including the
    LoRA-default lr=1e-4 [Hu+22]), no random-init AdamW configuration reaches
    R_context > X.XX. Ridge-init is stable at lr=3e-5 and drifts away above lr=1e-4.
    The optimization landscape is unfriendly to random-init at every lr we tested,
    and unfriendly to ridge-init at the very lr commonly used for LoRA finetuning."
  - LoRA-default-lr punch: §5.5 closing sentence "Practitioners attempting to deploy
    LoRA value corrections at LoRA-default hyperparameters would observe degradation,
    not improvement."

- **Outcome 2 (25%, some lr makes random_init > 0.3)**: lr-conditional;
  random-init works at some sweet spot.
  - Main claim flip; SGD is "lr-sensitive" not "unreachable"
  - Title: "SGD-Hard: Closed-Form Solves Without Hyperparameter Search"
  - §5.5 narrative: "Random-init AdamW recovers R_context = 0.X at lr=X.Xe-X,
    but degrades catastrophically at adjacent lrs. The closed-form ridge is
    obtained without hyperparameter search and reaches a strictly higher
    recovery on the same setup."
  - Paper still publishable but story shifts from "SGD cannot" to "ridge wins
    without tuning"

- **Outcome 3 (15%, ridge_init never recovers ≥ 0.5 at any lr)**:
  ridge_ft.json's 0.78 is irreproducible or due to a step-count / data-order
  difference we haven't isolated.
  - Pause §5.5 / abstract changes
  - Diff this run's calib data order, batch RNG seed, ridge_calibrate code path
    against ridge_ft's commit
  - Possible Appendix C: reproducibility of ridge_init zero-shot recovery

---

# Pending §5.5 prose diffs (apply when seq=51 LR probe lands + scenario chosen)

Originally drafted assuming seq=49 (lr=1e-4 + soft-cap) numbers; now to be
applied to seq=51 LR probe (lr=1e-4 + differentiable cap) numbers. Variant naming
matches `outputs/v1b_ridge/sgd_pressure.json` (post seq=51 LR probe push).

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
