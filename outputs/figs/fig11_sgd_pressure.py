"""Fig 11. SGD pressure test (§5.5). Extended training, alternative optimizers,
warmup, and curriculum norm-cap all fail to reach the closed-form ridge ceiling
at the three most context-bound Pythia-410M layers.

Input:  outputs/v1b_ridge/sgd_pressure.json
Schema (from mode_sgd_pressure in v1a/v1a_correction.py, --steps 5000):
  { "6":  {"A1_recovery": .., "baseline_2k": .., "long_adamw": ..,
           "warmup_adamw": .., "lion": .., "curriculum_cap": ..,
           "ridge_init_ft": ..},
    "7":  {...},  "11": {...} }
`ridge_init_ft` is the closed-form rank-64 cap=0.5 ridge ceiling used to
initialize the control variant; we plot it as the per-layer ceiling line.
The other five entries are SGD-from-random recipes.
"""
import numpy as np
from plotting_lib import fig_double, save, COLOR, PALETTE, load_json

d = load_json("v1b_ridge/sgd_pressure.json")
layers = sorted(int(k) for k in d.keys() if k.isdigit())

ORDER = [
    ("baseline_2k",    "Baseline (2k step)",     PALETTE["gray"]),
    ("long_adamw",     "AdamW long (5k)",        PALETTE["blue"]),
    ("warmup_adamw",   "AdamW + warmup (5k)",    PALETTE["skyblue"]),
    ("lion",           "Lion (5k)",              PALETTE["orange"]),
    ("curriculum_cap", "Curriculum cap 0.1->0.5", PALETTE["purple"]),
]
CEIL_KEY = "ridge_init_ft"


def _f(v):
    if isinstance(v, (int, float)):
        return float(v)
    return float("nan")  # "NaN" string or missing -> NaN bar


fig, ax = fig_double(h_ratio=0.46)
x = np.arange(len(layers))
n = len(ORDER)
bw = 0.78 / n
for i, (key, label, color) in enumerate(ORDER):
    vals = [_f(d[str(L)].get(key)) for L in layers]
    offset = (i - (n - 1) / 2) * bw
    ax.bar(x + offset, vals, bw, label=label, color=color,
           edgecolor="white", linewidth=0.4, zorder=3)

for j, L in enumerate(layers):
    ceil = _f(d[str(L)].get(CEIL_KEY))
    ax.hlines(ceil, x[j] - 0.42, x[j] + 0.42,
              colors=COLOR["ridge"], linestyles=(0, (4, 2)),
              linewidth=1.0, zorder=4)
    ax.text(x[j] + 0.43, ceil, f"  ridge={ceil:.2f}",
            color=COLOR["ridge"], fontsize=7, va="center", ha="left")

ax.axhline(0, color="black", linewidth=0.5, zorder=2)
ax.set_xticks(x)
ax.set_xticklabels([f"L{L}" for L in layers])
ax.set_xlabel("Layer")
ax.set_ylabel(r"$R_{\mathrm{context}}$")
ax.set_ylim(-0.4, 1.0)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.34),
          ncol=3, frameon=False, fontsize=7)
save(fig, "fig11")

# Caption:
# At Pythia-410M's three most context-bound layers (L6/L7/L11) no random-init
# SGD recipe reaches the closed-form ridge ceiling. Bars: 5 SGD variants of
# increasing aggressiveness. Dashed red: `ridge_init_ft`, the rank-64 cap=0.5
# closed-form ridge used to initialize the control - the apples-to-apples
# ceiling. The gap is an optimization-from-random-init phenomenon, not a
# representation limit.
