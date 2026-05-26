"""Fig 3 (single): A0 (~0) vs A1 (fitted) vs oracle PCA recovery over depth.
Data: outputs/v1a/anchor_audit_full24.json."""
from plotting_lib import fig_single, save, COLOR, LINESTYLE, load_json, tidy

d = load_json("v1a/anchor_audit_full24.json")
Ls = sorted(int(L) for L in d if d[L]["v0_delta"] > 0.05)
A0 = [d[str(L)]["A0"]["recovery"] for L in Ls]
A1 = [d[str(L)]["A1"]["recovery"] for L in Ls]
ORA = [d[str(L)]["oracle_r256"]["recovery"] for L in Ls]

fig, ax = fig_single(h_ratio=0.7)
ax.plot(Ls, ORA, color=COLOR["oracle"], ls=LINESTYLE["oracle"], lw=1.2, label="oracle PCA (r256)")
ax.plot(Ls, A1, color=COLOR["A1"], ls=LINESTYLE["A1"], marker="o", markersize=3, lw=1.2, label="A1 fitted table")
ax.plot(Ls, A0, color=COLOR["A0"], ls=LINESTYLE["A0"], lw=1.0, label="A0 embed-proj table")
# highlight L6/L7 A1 trough
for L in (6, 7):
    ax.scatter([L], [d[str(L)]["A1"]["recovery"]], s=34, facecolors="none",
               edgecolors=COLOR["highlight"], linewidths=1.0, zorder=5)
ax.text(0.40, 0.06, "A0: ~0 at all layers", transform=ax.transAxes, fontsize=7, color=COLOR["A0"])
ax.set_xlabel("Layer"); ax.set_ylabel("Recovery"); ax.set_ylim(-0.03, 1.03)
ax.legend(ncol=1, loc="center right", fontsize=7); tidy(ax)
save(fig, "fig3")

# Caption:
# Both A0 and A1 are context-free per-token values, yet A0 recovers ~0 and A1
# recovers 0.59-0.87. Most of the v0 ablation cost is a weak anchor, not
# contextualization. Circled: the L6/L7 A1 trough (most context-bound). The oracle
# PCA of the residual (using real V) is high everywhere.
