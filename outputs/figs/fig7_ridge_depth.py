"""Fig 7 (double, 2-panel): closed-form ridge recovery over layers. Left R_total,
right R_context; A1 alone vs ridge cap0.15 vs ridge uncapped. The residual is a
generalizing linear function of LN_l(h). Data: outputs/v1b_ridge/ridge_results.json."""
from plotting_lib import fig_double_2panel, save, COLOR, PALETTE, load_json, tidy

d = load_json("v1b_ridge/ridge_results.json")
Ls = sorted((int(L) for L in d), key=int)
xs = list(Ls)
def col(metric, key):
    return [d[str(L)]["variants"][key][metric] for L in Ls]
A1 = [d[str(L)]["A1_recovery"] for L in Ls]

fig, (axL, axR) = fig_double_2panel(h_ratio=0.42)
# Left: R_total
axL.plot(xs, A1, color=COLOR["A1"], marker="o", markersize=3, lw=1.2, label="A1 alone")
axL.plot(xs, col("R_total", "full_cap0.15"), color=PALETTE["orange"], ls=(0, (4, 2)), lw=1.2, label="ridge cap0.15")
axL.plot(xs, col("R_total", "full_uncapped"), color=COLOR["ridge"], lw=1.2, marker="s", markersize=3, label="ridge uncapped")
axL.set_xlabel("Layer"); axL.set_ylabel(r"$R_{\mathrm{total}}$"); axL.set_ylim(0, 1.02)
axL.set_xticks(xs); axL.legend(loc="lower left", fontsize=7); tidy(axL)
# Right: R_context
axR.plot(xs, col("R_context", "full_cap0.15"), color=PALETTE["orange"], ls=(0, (4, 2)), lw=1.2, label="ridge cap0.15")
axR.plot(xs, col("R_context", "full_uncapped"), color=COLOR["ridge"], lw=1.2, marker="s", markersize=3, label="ridge uncapped")
axR.set_xlabel("Layer"); axR.set_ylabel(r"$R_{\mathrm{context}}$"); axR.set_ylim(0, 1.0)
axR.set_xticks(xs); axR.legend(loc="center right", fontsize=7); tidy(axR)
axR.annotate("ridge fills residual\nat every layer", xy=(7, d["7"]["variants"]["full_uncapped"]["R_context"]),
             xytext=(11, 0.45), fontsize=7, color=COLOR["ridge"],
             arrowprops=dict(arrowstyle="->", lw=0.6, color=COLOR["ridge"]))
save(fig, "fig7")

# Caption:
# The post-anchor residual is a generalizing linear function of LN_l(h). Left:
# total recovery. Right: context recovery (fraction of the post-A1 residual). Ridge
# (uncapped) recovers 0.71-0.86 at every layer; the 0.15 norm cap structurally
# under-budgets the correction.
