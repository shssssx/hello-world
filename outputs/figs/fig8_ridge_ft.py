"""Fig 8 (single): closed-form ridge is reached only from ridge init; random-init
SGD does not find it (sometimes worsens the anchor). Grouped bars, hatching as 2nd
channel. Data: outputs/v1b_ridge/ridge_ft.json."""
import numpy as np
from plotting_lib import fig_single, save, COLOR, PALETTE, load_json, tidy

d = load_json("v1b_ridge/ridge_ft.json")
Ls = ["6", "7", "11"]
conds = [("ridge_init_notrain", "ridge-init (no train)", PALETTE["red"], None),
         ("ridge_init_ft", "ridge-init + finetune", PALETTE["orange"], "///"),
         ("random_init_ft", "random-init + finetune", PALETTE["gray"], "---")]
x = np.arange(len(Ls)); w = 0.26
fig, ax = fig_single(h_ratio=0.72)
for i, (k, lab, c, hatch) in enumerate(conds):
    ax.bar(x + (i - 1) * w, [d[L][k] for L in Ls], w, label=lab,
           color=c, hatch=hatch, edgecolor="white", linewidth=0.3)
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(x); ax.set_xticklabels([f"L{L}" for L in Ls])
ax.set_ylabel(r"$R_{\mathrm{context}}$")
ax.legend(ncol=1, loc="upper right", fontsize=7); tidy(ax)
save(fig, "fig8")

# Caption:
# The closed-form ridge solution is reached only from ridge initialization; SGD
# from random init does not find it, and at L11 worsens the anchor (negative
# recovery). Bars: recovery of the A1 residual at the three most context-bound
# layers, for ridge-init (no training), ridge-init + CE finetune, and random-init +
# CE finetune (relaxed cap 0.5).
