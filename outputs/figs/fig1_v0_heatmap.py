"""Fig 1 (double): v0 value-path ablation. Left = per-layer coarse delta with
mid-stack peaks annotated; right = per-(layer,head) heatmap, near-zero everywhere.
Data: outputs/coarse_loss_delta.npy, outputs/fine_loss_delta.npy."""
import numpy as np
from plotting_lib import fig_double_2panel, save, COLOR, OUT, tidy
import os

coarse = np.load(os.path.join(OUT, "coarse_loss_delta.npy"))
fine = np.load(os.path.join(OUT, "fine_loss_delta.npy"))
L, H = fine.shape
fig, (axL, axR) = fig_double_2panel(h_ratio=0.42)

layers = np.arange(L)
axL.plot(layers, coarse, color=COLOR["A1"], marker="o", markersize=3, lw=1.2)
for pk in (5, 11, 17):
    axL.annotate(f"{coarse[pk]:.2f}", xy=(pk, coarse[pk]), xytext=(pk, coarse[pk] + 0.07),
                 ha="center", fontsize=7, color=COLOR["highlight"],
                 arrowprops=dict(arrowstyle="-", lw=0.6, color=COLOR["highlight"]))
axL.set_xlabel("Layer"); axL.set_ylabel(r"$\Delta$ CE (nats)")
axL.set_xticks(range(0, L, 4)); tidy(axL); axL.set_ylim(0, coarse.max() * 1.18)

im = axR.imshow(fine, aspect="auto", cmap="viridis", vmin=0, vmax=np.nanmax(fine))
axR.set_xlabel("Head"); axR.set_ylabel("Layer")
axR.set_xticks(range(0, H, 3)); axR.set_yticks(range(0, L, 4))
axR.text(0.5, 1.0, r"97.7% of heads $|\Delta|<0.05$", transform=axR.transAxes,
         ha="center", va="bottom", fontsize=7)
cb = fig.colorbar(im, ax=axR, fraction=0.046, pad=0.02)
cb.set_label(r"$\Delta$ CE (nats)", fontsize=8); cb.ax.tick_params(labelsize=7)
save(fig, "fig1")

# Caption:
# Value-path ablation: per-layer cost is mid-stack-concentrated, per-head cost is
# near-zero everywhere. Left: replacing a whole layer's V with the token table A0
# costs 0.08-0.62 nats, peaking mid-stack (L5/L11/L17 annotated). Right: replacing a
# single head's V costs almost nothing; 97.7% of 384 heads have |delta CE| < 0.05.
