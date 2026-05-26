"""Fig 9 (single): r64 ridge context-recovery vs calibration size - rises and
plateaus, substantial already at n=250. Data: outputs/v1b_ridge/ridge_scale.json."""
from plotting_lib import fig_single, save, PALETTE, load_json, tidy

d = load_json("v1b_ridge/ridge_scale.json")
sizes = [250, 500, 1000, 2000, 4000]
cols = {"5": PALETTE["blue"], "6": PALETTE["orange"], "7": PALETTE["green"], "11": PALETTE["red"]}
fig, ax = fig_single(h_ratio=0.7)
for L in ["5", "6", "7", "11"]:
    if L not in d:
        continue
    y = [d[L][str(n)]["r64_uncapped"] for n in sizes]
    ax.plot(sizes, y, color=cols[L], marker="o", markersize=3, lw=1.2, label=f"L{L}")
    ax.axhline(y[-1], color=cols[L], lw=0.4, ls=(0, (1, 2)))
ax.set_xscale("log"); ax.set_xticks(sizes); ax.set_xticklabels(sizes)
ax.set_xlabel("Calibration sequences"); ax.set_ylabel(r"$R_{\mathrm{context}}$ (r64)")
ax.set_ylim(0, 0.9); ax.legend(ncol=2, loc="lower right", fontsize=7); tidy(ax)
save(fig, "fig9")

# Caption:
# r64 ridge context-recovery rises gently with calibration size and plateaus
# (dashed = each layer's n=4000 value); it is already substantial at n=250
# sequences, far below the 1024x1024 ridge map's parameter count - not a
# saturation artifact. lambda selected on a disjoint validation set.
