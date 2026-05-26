"""Fig 4 (single): cumulative variance of (V_real - A1) for L5 vs L11 - nearly
identical, high-rank. Data: outputs/v1a/diff_eig_L{5,11}.npy."""
import numpy as np, os
from plotting_lib import fig_single, save, COLOR, OUT, tidy

fig, ax = fig_single(h_ratio=0.7)
for L, c, ls in [(5, COLOR["A1"], "-"), (11, COLOR["highlight"], (0, (4, 2)))]:
    ev = np.load(os.path.join(OUT, f"v1a/diff_eig_L{L}.npy"))
    ev = np.sort(ev)[::-1].clip(0)
    cum = np.cumsum(ev) / ev.sum()
    ax.plot(np.arange(1, len(cum) + 1), cum, color=c, ls=ls, lw=1.3, label=f"L{L}")
    for frac in (0.5, 0.9):
        k = int((cum < frac).sum()) + 1
        ax.plot([k, k], [0, frac], color=c, lw=0.5, ls=":")
ax.axhline(0.9, color=COLOR["gray"], lw=0.5, ls=":")
ax.set_xscale("log"); ax.set_xlim(1, 1024)
ax.set_xlabel("Component (rank)"); ax.set_ylabel("Cumulative variance"); ax.set_ylim(0, 1.02)
ax.legend(loc="lower right", fontsize=8); tidy(ax)
ax.text(0.04, 0.9, "90% var", transform=ax.transAxes, fontsize=7, va="bottom", color=COLOR["gray"])
save(fig, "fig4")

# Caption:
# L5 and L11 have nearly identical residual spectra - high-rank in both (~330
# components for 90% variance). Depth-varying intrinsic rank therefore does not
# explain why the residual is recoverable at L5 but the trained corrector fails at
# L11; the difference is learnability, not dimensionality.
