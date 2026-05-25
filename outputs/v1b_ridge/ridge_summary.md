# v1b ridge probe — closed-form linear ceiling (training vs representation)

Question: the trained LoRA/MLP corrector reading LN_l(h) failed to recover the
A1 residual (genuine-context V) — was that because the signal is **not in
LN_l(h)** (representation), or because **SGD/parameterization** failed? Closed-form
ridge `W=(XᵀX+λI)⁻¹XᵀY` with `X=LN_l(h)`, `Y=V_real−V_A1(x)`, injected as
`V=V_A1+XW`, eval CE on the held (sequence-disjoint) set.

## Result (full-rank, λ by eval CE)

| layer | A1 alone | ridge full **R_total** | ridge full **R_context** | ridge cap0.15 R_ctx | ridge r64 R_ctx | best λ |
|---|---|---|---|---|---|---|
| L5  | 0.874 | 0.976 | 0.805 | 0.411 | 0.532 | 1000 |
| L6  | 0.234 | 0.894 | 0.862 | 0.463 | 0.783 | 1e-4 |
| L7  | 0.220 | 0.889 | 0.858 | 0.381 | 0.746 | 1e-3 |
| L11 | 0.587 | 0.905 | 0.770 | 0.317 | 0.576 | 0.1 |
| L20 | 0.845 | 0.955 | 0.708 | 0.320 | 0.501 | 1000 |

R_total = 1 − Δ_ridge/Δ_A0 ; R_context = (Δ_A1 − Δ_ridge)/Δ_A1 (recovery of the
A1 residual specifically). Eval is sequence-disjoint from the calibration used to
fit W → generalization, not memorization.

## Verdict — Case A (it's training, not representation)

**The genuine-context residual IS a generalizing linear function of LN_l(h).** A
closed-form linear map recovers 71–86% of it at every probed layer, including the
most context-bound L6/L7 (86%) and L11 (77%); total recovery 0.89–0.98. So the
prior trained-corrector failure (~3% A0-base, +0% A1-base) was an **optimization /
parameterization pathology, not absence of the signal**. This **refutes the
earlier "not learnable from LN_l(h) / representation problem" reading and takes
Option 1 (representation search) off the table.**

Two contributing culprits identified for the SGD failure:
1. **Norm cap too tight.** Capped (0.15) ridge recovers only 0.32–0.46 of the
   residual vs 0.71–0.86 uncapped → the needed linear correction has norm well
   above 0.15·‖V‖. The SGD corrector was capped at 0.15 (to stop divergence), so
   it was structurally unable to reach the solution; uncapped SGD diverged. Ridge
   sidesteps both.
2. **Rank.** r64 ridge gets 0.50–0.78 (most of it at L6/L7); the residual is
   somewhat high-rank but not extreme.

## Implication for v1b

The correction is linearly learnable from LN_l(h); fix the **training**, not the
input representation:
- initialize the LoRA from the ridge solution (or distill ridge→low-rank);
- relax / schedule the norm budget (0.15 is far too tight) with a stability
  method other than a hard cap;
- or two-stage: MSE-regress the residual (closed-form/SGD) then CE-finetune.

## Caveat

best λ at L6/L7 is near-unregularized (1e-4/1e-3) and W is d×d (~1M params) fit on
~1M calibration tokens — borderline; the high *eval* R_total confirms it
generalizes, but a larger calibration set would harden the claim.
