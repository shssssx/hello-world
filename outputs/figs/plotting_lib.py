"""Conference-grade matplotlib style + helpers (EMNLP/ACL/NeurIPS/ICLR).

Wong (Nature 2011) colorblind-safe palette; serif/Computer-Modern math; Tufte
spines; embedded TrueType fonts (pdf.fonttype=42); single/double-column sizing.
All figure scripts import from here.
"""
import json
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ---- Typography ----
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.titlesize": 10,
})
# ---- Lines & ticks ----
mpl.rcParams.update({
    "lines.linewidth": 1.2, "lines.markersize": 4, "lines.markeredgewidth": 0.8,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.minor.visible": False, "ytick.minor.visible": False,
})
# ---- Layout / output ----
mpl.rcParams.update({
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.borderaxespad": 0.4,
    "figure.dpi": 150, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

PALETTE = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
    "purple": "#CC79A7", "yellow": "#F0E442", "skyblue": "#56B4E9",
    "black": "#000000", "gray": "#7F7F7F",
}
COLOR = {
    "A0": PALETTE["gray"], "A1": PALETTE["blue"], "ridge": PALETTE["red"],
    "oracle": PALETTE["green"], "sgd": PALETTE["orange"], "lora_r16": PALETTE["purple"],
    "baseline": PALETTE["black"], "highlight": PALETTE["red"], "gray": PALETTE["gray"],
}
LINESTYLE = {
    "A0": (0, (2, 2)), "A1": "-", "ridge": "-",
    "oracle": (0, (5, 1, 1, 1)), "sgd": (0, (1, 1.5)),
}
MARKER = {"A0": "x", "A1": "o", "ridge": "s", "oracle": "^", "sgd": "v", "lora_r16": "D"}

HERE = os.path.dirname(os.path.abspath(__file__))           # outputs/figs
OUT = os.path.dirname(HERE)                                 # outputs


def load_json(rel):
    with open(os.path.join(OUT, rel)) as f:
        return json.load(f)


def fig_single(h_ratio=0.62):
    return plt.subplots(figsize=(3.15, 3.15 * h_ratio), constrained_layout=True)


def fig_double(h_ratio=0.40):
    return plt.subplots(figsize=(6.5, 6.5 * h_ratio), constrained_layout=True)


def fig_double_2panel(h_ratio=0.40, sharex=False, sharey=False):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 6.5 * h_ratio),
                             constrained_layout=True, sharex=sharex, sharey=sharey)
    return fig, axes


def tidy(ax, nbins_y=5):
    ax.yaxis.set_major_locator(MaxNLocator(nbins_y))


def sanity_check_figure(fig, max_width_in=6.5):
    w, h = fig.get_size_inches()
    if w > max_width_in + 1e-6:
        print(f"  WARN width {w:.2f}in > {max_width_in}in")
    for ax in fig.axes:
        if ax.get_label() == "<colorbar>" or ax.get_images():
            continue  # skip colorbar / heatmap axes
        if not ax.get_xlabel():
            print("  WARN axis missing xlabel")
        if not ax.get_ylabel():
            print("  WARN axis missing ylabel")
        if len(ax.get_yticks()) > 7:
            print(f"  WARN {len(ax.get_yticks())} y-ticks (>7)")


def save(fig, name):
    sanity_check_figure(fig)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  saved {name}.pdf/.png")
