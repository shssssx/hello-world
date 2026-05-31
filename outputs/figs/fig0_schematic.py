"""Fig 0 (double, schematic): four-panel V-derivation pipeline.
matplotlib fallback for primary TikZ version embedded in main.tex.

Panels:
  (a) V-real:    V = W_V * LN(h_{l-1,t})
  (b) A0 anchor: V_hat = W_V * LN(E[x_t])      (token-grounded, embed-projected)
  (c) A1 anchor: V_hat = E[V_real | token=x]    (token-grounded, fitted average)
  (d) Ridge:     V_hat = A1 + W_ridge * LN(h)   (context residual, closed-form)
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from plotting_lib import save, COLOR, PALETTE


def block(ax, xy, w, h, text, fc, ec=None, fontsize=8):
    ec = ec or fc
    box = FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.04",
                         linewidth=0.7, edgecolor=ec, facecolor=fc, alpha=0.18)
    ax.add_patch(box)
    box2 = FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h,
                          boxstyle="round,pad=0.02,rounding_size=0.04",
                          linewidth=0.8, edgecolor=ec, facecolor="none")
    ax.add_patch(box2)
    ax.text(xy[0], xy[1], text, ha="center", va="center",
            fontsize=fontsize, color="#222")


def arrow(ax, p0, p1, label=None, dy=0.06):
    a = FancyArrowPatch(p0, p1, arrowstyle="->", mutation_scale=8,
                        lw=0.7, color="#444")
    ax.add_patch(a)
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + dy
        ax.text(mx, my, label, ha="center", va="bottom",
                fontsize=7, color="#444")


def panel(ax, title, equation, blocks, edges):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=9, loc="left", pad=4)
    for (xy, w, h, txt, fc) in blocks:
        block(ax, xy, w, h, txt, fc)
    for (p0, p1, lab) in edges:
        arrow(ax, p0, p1, lab)
    ax.text(0.5, 0.04, equation, ha="center", va="center",
            fontsize=8.5, color=COLOR["highlight"])


fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.5 * 0.62),
                        constrained_layout=True)

# (a) V-real path
panel(axes[0, 0], r"(a) V-real (ground truth)",
      r"$V = W_V \cdot \mathrm{LN}(h_{l-1,t})$",
      blocks=[
          ((0.12, 0.65), 0.20, 0.22, r"$h_{l-1,t}$", PALETTE["blue"]),
          ((0.45, 0.65), 0.20, 0.22, r"$\mathrm{LN}$", PALETTE["gray"]),
          ((0.78, 0.65), 0.22, 0.22, r"$W_V$", PALETTE["green"]),
      ],
      edges=[((0.22, 0.65), (0.35, 0.65), None),
             ((0.55, 0.65), (0.67, 0.65), None),
             ((0.89, 0.65), (0.96, 0.50), r"$V$")])

# (b) A0 anchor
panel(axes[0, 1], r"(b) A0 anchor (embed-proj)",
      r"$\widehat{V}_{A0} = W_V \cdot \mathrm{LN}(\mathbb{E}[x_t])$",
      blocks=[
          ((0.12, 0.65), 0.20, 0.22, r"$x_t$", PALETTE["orange"]),
          ((0.45, 0.65), 0.20, 0.22, r"$\mathbb{E}$", PALETTE["gray"]),
          ((0.78, 0.65), 0.22, 0.22, r"$W_V\,\mathrm{LN}$", PALETTE["green"]),
      ],
      edges=[((0.22, 0.65), (0.35, 0.65), None),
             ((0.55, 0.65), (0.67, 0.65), None),
             ((0.89, 0.65), (0.96, 0.50), r"$\widehat{V}$")])

# (c) A1 anchor
panel(axes[1, 0], r"(c) A1 anchor (fitted table)",
      r"$\widehat{V}_{A1} = \mathbb{E}[V_{\mathrm{real}} \mid \mathrm{token}=x]$",
      blocks=[
          ((0.18, 0.65), 0.28, 0.22, r"calib. corpus", PALETTE["orange"]),
          ((0.55, 0.65), 0.22, 0.22, r"$V$-table", PALETTE["purple"]),
          ((0.85, 0.65), 0.18, 0.22, r"$x_t$", PALETTE["blue"]),
      ],
      edges=[((0.32, 0.65), (0.44, 0.65), r"avg"),
             ((0.85, 0.55), (0.65, 0.55), r"lookup")])

# (d) Ridge
panel(axes[1, 1], r"(d) Ridge (context residual)",
      r"$\widehat{V} = A_1 + W_{\mathrm{ridge}}\,\mathrm{LN}(h_{l-1,t})$",
      blocks=[
          ((0.13, 0.65), 0.22, 0.22, r"A1", PALETTE["purple"]),
          ((0.45, 0.65), 0.20, 0.22, r"$+$", PALETTE["gray"]),
          ((0.78, 0.65), 0.26, 0.22, r"$W_{\mathrm{ridge}}\mathrm{LN}(h)$",
           PALETTE["red"]),
      ],
      edges=[((0.24, 0.65), (0.35, 0.65), None),
             ((0.55, 0.65), (0.65, 0.65), None),
             ((0.91, 0.65), (0.96, 0.50), r"$\widehat{V}$")])

# Suppress sanity warnings: schematic has no axes labels by design.
for ax in axes.flatten():
    ax.set_xlabel(" "); ax.set_ylabel(" ")
save(fig, "fig0")

# Caption:
# Four definitions of the V-vector used in this paper. (a) the ground-truth V
# from real attention; (b) A0, a per-token embed-projected anchor (no
# corpus); (c) A1, a per-token average of real V over a calibration corpus -
# token-determined but uses the real W_V; (d) closed-form ridge on top of A1
# captures the context residual. (a)-(c) are introduced for diagnostic
# decomposition; (d) is the recommended deployable predictor.
