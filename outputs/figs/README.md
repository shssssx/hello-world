# Conference-grade figures

Regenerate all: `bash regenerate_all.sh` (from this dir). Each `figN_*.py` is
self-contained: reads data from `outputs/**`, writes `figN.pdf` (vector, primary) +
`figN.png` (preview) here. Style/colors/sizing live in `plotting_lib.py` (Wong
colorblind-safe palette, serif + CM math, Tufte spines, `pdf.fonttype=42` for font
embedding, single-column 3.15in / double-column 6.5in).

| fig | paper section | type | data |
|---|---|---|---|
| fig1  | §5.1 v0 ablation | double, 2-panel | `coarse_loss_delta.npy`, `fine_loss_delta.npy` |
| fig2  | §5.2 (core) decomposition | double, 2-panel | `v1a/anchor_audit_full24.json` |
| fig3  | §5.2 anchor audit | single | `v1a/anchor_audit_full24.json` |
| fig4  | §5.2 residual spectrum | single | `v1a/diff_eig_L{5,11}.npy` |
| fig5  | §5.3 SGD diverges | double, 2-panel | `v1a/probe_L11_*.json` |
| fig6  | §5.4 ridge vs rank | single | `v1b_ridge/ridge_results.json` |
| fig7  | §5.4 ridge over depth | double, 2-panel | `v1b_ridge/ridge_results.json` |
| fig8  | §5.5 ridge vs SGD finetune | single | `v1b_ridge/ridge_ft.json` |
| fig9  | §5.6 calibration scaling | single | `v1b_ridge/ridge_scale.json` |
| fig10 | §5.7 160M replication | double, 2-panel | `v1b_160m/repro160.json` |

Self-contained captions are in each script's trailing comment (copy into the paper).
Original exploratory PNGs are kept under `outputs/v1a/`, `outputs/v1b_ridge/`,
`outputs/v1b_160m/` for history. Fonts fall back to DejaVu Serif if Times/Nimbus are
absent; install `texlive`/`msttcorefonts` for exact Times rendering.

Pending: a §5.5 SGD-pressure-test panel can be added once `sgd_pressure.json` (seq=49)
lands — overlay the long-AdamW/Lion/warmup/curriculum recovery as a flat band against
the ridge ceiling.
