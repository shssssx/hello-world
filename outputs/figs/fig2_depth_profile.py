"""Fig 2 (double, 2-panel) — core figure. Left: A1 recovery vs depth (U-shape,
mid-stack trough shaded). Right: per-layer v0 cost split into A1-recoverable vs
genuine-context. Data: outputs/v1a/anchor_audit_full24.json."""
import numpy as np
from plotting_lib import fig_double_2panel, save, COLOR, load_json, tidy

d = load_json("v1a/anchor_audit_full24.json")
Ls = sorted(int(L) for L in d if d[L]["v0_delta"] > 0.05)
a1 = [d[str(L)]["A1"]["recovery"] for L in Ls]
ctx = [1 - x for x in a1]

fig, (axL, axR) = fig_double_2panel(h_ratio=0.42, sharex=True)
# Left: A1 recovery U-shape
axL.axvspan(5.5, 7.5, color=COLOR["gray"], alpha=0.15, lw=0)
axL.plot(Ls, a1, color=COLOR["A1"], marker="o", markersize=3, lw=1.2)
tr = min(Ls, key=lambda L: d[str(L)]["A1"]["recovery"])
axL.annotate("context-bound\nregion (L6/L7)", xy=(7, d["7"]["A1"]["recovery"]),
             xytext=(11, 0.42), fontsize=7, color=COLOR["highlight"],
             arrowprops=dict(arrowstyle="->", lw=0.6, color=COLOR["highlight"]))
axL.set_xlabel("Layer"); axL.set_ylabel(r"$R_{\mathrm{total}}$ (A1 anchor)")
axL.set_ylim(0, 1.02); tidy(axL)
# Right: stacked decomposition (A1-recoverable below, context above)
axR.bar(Ls, a1, color=COLOR["A1"], label="token-determined (A1)", width=0.8)
axR.bar(Ls, ctx, bottom=a1, color=COLOR["highlight"], label="genuine context (1-A1)", width=0.8)
axR.set_xlabel("Layer"); axR.set_ylabel("Fraction of v0 cost"); axR.set_ylim(0, 1.0)
axR.legend(ncol=1, loc="lower center", fontsize=7)
tidy(axR)
save(fig, "fig2")

# Caption:
# Most layer-wise value-ablation cost is token-determined (A1); genuine
# contextualization concentrates mid-stack. Left: A1 token-anchor recovery is
# U-shaped over depth with a sharp trough at L6/L7 (A1 ~ 0.22; shaded). Right:
# each layer's v0 cost split into A1-recoverable (blue) and genuine context
# residual (red, =1-A1); the residual peaks where A1 is weakest.
