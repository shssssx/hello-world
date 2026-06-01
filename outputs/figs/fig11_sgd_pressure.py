"""Fig 11 (LR sensitivity, replaces old 6-variant sgd_pressure bar chart).
Three subplots (L6, L7, L11), x = lr (log scale), y = R_context.
Two main lines per subplot: ridge_init (solid) + random_init (dashed) from
outputs/v1b_ridge/lr_probe.json. Horizontal dashed reference: closed-form
ridge ceiling from outputs/v1b_ridge/ridge_ft.json (ridge_init_notrain, the
SAME rank-64 cap=0.5 ridge solution as plotted, evaluated WITHOUT subsequent
SGD training -- the 'training-free deployable' value).

Overlay (scatter markers, no extra GPU): the seq=49 6-variant data at
lr=1e-4 (Lion at lr=3e-5) for visual confirmation that variant tweaks
within fixed-lr SGD do not escape the envelope. Numbers transcribed from
the seq=49 run (pre cap-fix, same soft_nograd cap as the LR probe).
"""
import numpy as np
from plotting_lib import fig_double, save, COLOR, PALETTE, load_json

probe = load_json("v1b_ridge/lr_probe.json")
ridge_ft = load_json("v1b_ridge/ridge_ft.json")

LAYERS = probe["layers"]                          # [6, 7, 11]
LRS = probe["lrs"]                                # [1e-5, 3e-5, 1e-4, 3e-4]
A1 = probe["A1_recovery"]                         # {"6": .., "7": .., "11": ..}

# seq=49 6-variant transcript data (soft_nograd cap, lr=1e-4 except Lion at 3e-5)
SEQ49 = {
    "6":  {"baseline_2k":  (1e-4, 0.1182),
           "long_adamw":   (1e-4, -0.2406),
           "warmup_adamw": (1e-4, 0.1624),
           "curriculum_cap": (1e-4, -0.0499),
           "lion":         (3e-5, 0.1349),
           "ridge_init_ft": (1e-4, 0.2950)},
    "7":  {"baseline_2k":  (1e-4, -0.5492),
           "long_adamw":   (1e-4, -0.2667),
           "warmup_adamw": (1e-4, -0.8085),
           "curriculum_cap": (1e-4, -0.3103),
           "lion":         (3e-5, -0.2022),
           "ridge_init_ft": (1e-4, -0.2271)},
    "11": {"baseline_2k":  (1e-4, -1.2954),
           "long_adamw":   (1e-4, -0.9807),
           "warmup_adamw": (1e-4, -1.0787),
           "curriculum_cap": (1e-4, -0.7258),
           "lion":         (3e-5, -0.9393),
           "ridge_init_ft": (1e-4, -0.9399)},
}
RANDOM_VARS = ("baseline_2k", "long_adamw", "warmup_adamw", "curriculum_cap", "lion")
RIDGE_VARS = ("ridge_init_ft",)


import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(6.5, 6.5 * 0.36),
                        constrained_layout=True, sharey=False)

for ax, L in zip(axes, LAYERS):
    cell = probe["data"][str(L)]
    ri = [cell["ridge_init"][f"{lr:.0e}"] for lr in LRS]
    rd = [cell["random_init"][f"{lr:.0e}"] for lr in LRS]

    # main lines
    ax.plot(LRS, ri, color=COLOR["ridge"], lw=1.4, marker="o", markersize=3,
            label="ridge-init")
    ax.plot(LRS, rd, color=COLOR["A1"], lw=1.4, ls=(0, (4, 2)), marker="s",
            markersize=3, label="random-init")

    # closed-form ridge ceiling (zero-shot, no SGD) from ridge_ft.json
    if str(L) in ridge_ft:
        ceil = ridge_ft[str(L)].get("ridge_init_notrain")
        if ceil is not None:
            ax.axhline(ceil, color=COLOR["ridge"], ls=":", lw=0.8, alpha=0.7)
            ax.text(LRS[0] * 0.85, ceil + 0.025, f"ridge zero-shot = {ceil:.2f}",
                    fontsize=6.5, color=COLOR["ridge"], ha="left", va="bottom")

    # zero baseline (= A1 anchor; below this means actively damaging)
    ax.axhline(0, color="black", lw=0.4, alpha=0.5)

    # seq=49 6-variant overlay (markers only, no line)
    for v in RANDOM_VARS:
        lr_v, val = SEQ49[str(L)][v]
        ax.scatter([lr_v], [val], marker="x", s=18, color=COLOR["A1"],
                   alpha=0.55, zorder=2, linewidths=0.7)
    for v in RIDGE_VARS:
        lr_v, val = SEQ49[str(L)][v]
        ax.scatter([lr_v], [val], marker="+", s=24, color=COLOR["ridge"],
                   alpha=0.55, zorder=2, linewidths=0.8)

    ax.set_xscale("log")
    ax.set_xticks(LRS)
    ax.set_xticklabels([f"{lr:.0e}" for lr in LRS], fontsize=7)
    ax.set_xlabel("learning rate (AdamW)")
    if L == LAYERS[0]:
        ax.set_ylabel(r"$R_{\mathrm{context}}$")
    ax.set_title(f"L{L}", fontsize=9)
    ax.set_ylim(-1.4, 1.0)
    ax.tick_params(axis="y", labelsize=7)

# single shared legend, top-right of last panel
axes[-1].legend(loc="lower left", fontsize=7, frameon=False)
save(fig, "fig11")

# Caption:
# LR sensitivity at the three most context-bound layers of Pythia-410M.
# Solid red: ridge-init AdamW (5000 step, r=64, cap=0.5). Dashed blue:
# random-init AdamW (same setup, fan-in initialization). Dotted horizontal
# red: the closed-form ridge solution evaluated WITHOUT subsequent SGD
# training (the deployable form). Random-init never reaches the ridge
# ceiling across 100x lr range (max 0.27 at L6, 0.01 at L7, negative at L11).
# Ridge-init degrades monotonically with lr above 1e-5, indicating that the
# closed-form solution is not a CE-loss local minimum. Markers ('x' random,
# '+' ridge-init) overlay the seq=49 six-variant sweep (Lion, warmup,
# curriculum cap, 2k/5k step AdamW) at lr=1e-4 (lr=3e-5 for Lion),
# confirming that within-fixed-lr SGD variants do not escape the envelope.
