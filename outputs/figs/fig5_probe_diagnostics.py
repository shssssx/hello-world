"""Fig 5 (double, 2-panel): SGD corrector at L11 diverges, not undertrains.
Left held CE vs step; right ||dV||/||V|| vs step. Data: outputs/v1a/probe_L11_*.json."""
from plotting_lib import fig_double_2panel, save, COLOR, PALETTE, load_json, tidy

BASE = 2.674  # 64-seq held baseline used during the probe
cfgs = [
    ("probe_L11_shared_r8_lr3e-05_s500_c0.json",  "lr 3e-5, 500 step",  PALETTE["blue"]),
    ("probe_L11_shared_r8_lr3e-05_s2000_c0.json", "lr 3e-5, 2000 step", PALETTE["orange"]),
    ("probe_L11_shared_r8_lr0.0001_s500_c1.json", "lr 1e-4 + clip",     PALETTE["red"]),
]
fig, (axL, axR) = fig_double_2panel(h_ratio=0.42)
for fn, lab, c in cfgs:
    s = [x for x in load_json(f"v1a/{fn}")["series"] if "held_ce" in x]
    st = [x["step"] for x in s]
    axL.plot(st, [x["held_ce"] for x in s], color=c, lw=1.2, label=lab)
    axR.plot(st, [x["dv_ratio"] for x in s], color=c, lw=1.2, label=lab)
axL.axhline(BASE, color=COLOR["baseline"], lw=0.6, ls=(0, (2, 2)))
axL.text(0.02, BASE, "no-correction baseline", transform=axL.get_yaxis_transform(),
         fontsize=6.5, va="bottom")
axL.set_xlabel("Training step"); axL.set_ylabel("Held-out CE"); tidy(axL)
axL.legend(loc="lower right", fontsize=6.5)
axR.set_xlabel("Training step"); axR.set_ylabel(r"$\|\Delta V\|/\|V\|$"); tidy(axR)
save(fig, "fig5")

# Caption:
# SGD-trained corrector at L11 diverges, not undertrains. Left: held-out CE rises
# above the no-correction baseline as training continues (2000 steps is worse than
# 500). Right: the correction norm grows unbounded. More steps and gradient
# clipping do not change the pattern.
