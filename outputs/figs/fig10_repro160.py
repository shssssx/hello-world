"""Fig 10 (double, 2-panel): Pythia-160M replication. Left A1 recovery (U-shape,
trough L4/L5) + selected-layer ridge R_context; right cost decomposition.
Data: outputs/v1b_160m/repro160.json."""
from plotting_lib import fig_double_2panel, save, COLOR, load_json, tidy

d = load_json("v1b_160m/repro160.json")
prof = d["profile"]
Ls = sorted((int(L) for L in prof if prof[L]["coarse_delta"] > 0.05))
a1 = [prof[str(L)]["A1_recovery"] for L in Ls]
ctx = [1 - x for x in a1]
sel = [int(x) for x in d["selected"]]
rx = sorted(sel); ry = [d["layers"][str(L)]["ridge_r64_unc_Rcontext"] for L in rx]

fig, (axL, axR) = fig_double_2panel(h_ratio=0.42, sharex=True)
axL.axvspan(3.5, 5.5, color=COLOR["gray"], alpha=0.15, lw=0)
axL.plot(Ls, a1, color=COLOR["A1"], marker="o", markersize=3, lw=1.2, label="A1 recovery")
axL.plot(rx, ry, color=COLOR["ridge"], ls=(0, (4, 2)), marker="s", markersize=3, lw=1.2,
         label=r"ridge r64 $R_{\mathrm{context}}$")
axL.set_xlabel("Layer"); axL.set_ylabel("Recovery"); axL.set_ylim(0, 1.02)
axL.legend(loc="lower center", fontsize=7); tidy(axL)
axR.bar(Ls, a1, color=COLOR["A1"], width=0.8, label="token-determined (A1)")
axR.bar(Ls, ctx, bottom=a1, color=COLOR["highlight"], width=0.8, label="genuine context")
axR.set_xlabel("Layer"); axR.set_ylabel("Fraction of v0 cost"); axR.set_ylim(0, 1.0)
axR.legend(loc="lower center", fontsize=7); tidy(axR)
fig.suptitle("Pythia-160M (12 layers)", fontsize=9)
save(fig, "fig10")

# Caption:
# Cross-scale replication on Pythia-160M. Left: A1 recovery is U-shaped with a
# mid-stack trough at L4/L5 (as in 410M's L6/L7); selected-layer ridge (dashed)
# fills the residual (L4 = 0.84). Right: cost decomposition, genuine context
# concentrated mid-stack. The decomposition reproduces; the depth profile shifts
# in index but not character.
