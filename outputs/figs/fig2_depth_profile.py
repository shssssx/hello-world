"""Fig 2 (double, 2x2) - core figure. Top row: Pythia-410M. Bottom row: Pythia-160M.
Left col: A1 recovery vs depth (U-shape, mid-stack trough shaded).
Right col: per-layer v0 cost split into A1-recoverable vs genuine-context.
Data: outputs/v1a/anchor_audit_full24.json + outputs/v1b_160m/repro160.json."""
import os
import matplotlib.pyplot as plt
from plotting_lib import save, COLOR, load_json, tidy, HERE


def panel_pair(axL, axR, Ls, a1, trough_lo, trough_hi, annotate_layer,
               annotate_xy_text, title):
    ctx = [1 - x for x in a1]
    axL.axvspan(trough_lo, trough_hi, color=COLOR["gray"], alpha=0.15, lw=0)
    axL.plot(Ls, a1, color=COLOR["A1"], lw=1.2)
    axL.annotate(f"context-bound\nregion (L{annotate_layer})",
                 xy=(annotate_layer, a1[Ls.index(annotate_layer)]),
                 xytext=annotate_xy_text, fontsize=7, color=COLOR["highlight"],
                 arrowprops=dict(arrowstyle="->", lw=0.6, color=COLOR["highlight"]))
    axL.set_xlabel("Layer")
    axL.set_ylabel(r"$R_{\mathrm{total}}$ (A1 anchor)")
    axL.set_ylim(0, 1.02)
    axL.set_title(title, fontsize=9)
    tidy(axL)
    axR.bar(Ls, a1, color=COLOR["A1"], label="token-determined (A1)", width=0.8)
    axR.bar(Ls, ctx, bottom=a1, color=COLOR["highlight"],
            label="genuine context (1-A1)", width=0.8)
    axR.set_xlabel("Layer")
    axR.set_ylabel("Fraction of v0 cost")
    axR.set_ylim(0, 1.0)
    axR.set_title(title, fontsize=9)
    axR.legend(ncol=1, loc="lower center", fontsize=7)
    tidy(axR)


# ---- 410M ----
d4 = load_json("v1a/anchor_audit_full24.json")
Ls4 = sorted(int(L) for L in d4 if d4[L]["v0_delta"] > 0.05)
a1_4 = [d4[str(L)]["A1"]["recovery"] for L in Ls4]

# ---- 160M ----
d1 = load_json("v1b_160m/repro160.json")["profile"]
Ls1 = sorted(int(L) for L in d1 if d1[L]["coarse_delta"] > 0.05
             and d1[L]["A1_recovery"] is not None
             and d1[L]["A1_recovery"] == d1[L]["A1_recovery"])  # NaN filter
a1_1 = [d1[str(L)]["A1_recovery"] for L in Ls1]

fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.5 * 0.78),
                         constrained_layout=True, sharex=False)
panel_pair(axes[0, 0], axes[0, 1], Ls4, a1_4, 5.5, 7.5, 7, (11, 0.42),
           "Pythia-410M (24 layers)")
panel_pair(axes[1, 0], axes[1, 1], Ls1, a1_1, 3.5, 5.5, 5, (7, 0.42),
           "Pythia-160M (12 layers)")
save(fig, "fig2")

# Caption:
# Most layer-wise value-ablation cost is token-determined (A1); genuine
# contextualization concentrates mid-stack, across two model scales. Top:
# Pythia-410M (24 layers), A1 recovery U-shaped with sharp trough at L6/L7.
# Bottom: Pythia-160M (12 layers), the same U-shape with a trough at L4/L5
# (proportional depth). Right column: each layer's v0 cost split into
# A1-recoverable (blue) and genuine context residual (red). The depth profile
# shifts in index but not character across scales.
