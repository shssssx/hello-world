"""Fig 6 (single): ridge context-recovery vs rank across layers; full-rank as the
ceiling (dashed). Low-rank ridge already captures most of the residual that the
SGD-trained corrector (Fig 8) cannot reach. Data: outputs/v1b_ridge/ridge_results.json."""
from plotting_lib import fig_single, save, PALETTE, load_json, tidy

d = load_json("v1b_ridge/ridge_results.json")
ranks = [16, 64, 256]
cols = {"5": PALETTE["blue"], "6": PALETTE["orange"], "7": PALETTE["green"], "11": PALETTE["red"]}
fig, ax = fig_single(h_ratio=0.7)
for L in ["5", "6", "7", "11"]:
    if L not in d:
        continue
    y = [d[L]["variants"][f"r{r}_uncapped"]["R_context"] for r in ranks]
    ax.plot(ranks, y, color=cols[L], marker="o", markersize=3, lw=1.2, label=f"L{L}")
    full = d[L]["variants"]["full_uncapped"]["R_context"]
    ax.axhline(full, color=cols[L], lw=0.5, ls=(0, (1, 1.5)))
ax.set_xscale("log", base=2); ax.set_xticks(ranks); ax.set_xticklabels(ranks)
ax.set_xlabel("Ridge rank"); ax.set_ylabel(r"$R_{\mathrm{context}}$"); ax.set_ylim(0, 1.0)
ax.legend(ncol=2, loc="lower right", fontsize=7); tidy(ax)
save(fig, "fig6")

# Caption:
# Low-rank closed-form ridge recovers most of the context residual; dashed lines
# mark each layer's full-rank ceiling. Even rank 64 captures 0.50-0.78 of the
# residual at L6/L7 - a regime SGD-trained adapters never reach (Fig 8).
