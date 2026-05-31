#!/usr/bin/env bash
# Regenerate all conference-grade figures (pdf + png). Run from outputs/figs/.
# Missing-input handling: every figure has a require_data guard naming the
# files it depends on; if any is absent the figure is skipped with a clean
# "skip" message rather than a Python traceback, so a partial repo still
# regenerates the figures that ARE ready.
#
# fig0_schematic.py is a matplotlib FALLBACK only. Primary Fig 0 source is
# fig0_schematic.tex (TikZ); compile that with pdflatex when LaTeX is
# available. Regenerate the fallback explicitly with: python3 fig0_schematic.py
set -u
cd "$(dirname "$0")"

# require_data <fig_label> <path1> [path2 ...]
# Returns 0 (success, => proceed) iff every path exists.
require_data() {
  local label="$1"; shift
  for p in "$@"; do
    if [ ! -e "$p" ]; then
      echo "skip $label: missing $p"
      return 1
    fi
  done
  return 0
}

run_fig() {
  echo "== $1 =="
  python3 "$1.py"
}

# fig1: v0 ablation heatmap
require_data fig1 ../coarse_loss_delta.npy ../fine_loss_delta.npy \
  && run_fig fig1_v0_heatmap

# fig2: cross-scale depth profile (410M + 160M)
require_data fig2 ../v1a/anchor_audit_full24.json ../v1b_160m/repro160.json \
  && run_fig fig2_depth_profile

# fig3: anchor audit (A0 vs A1 vs oracle)
require_data fig3 ../v1a/anchor_audit_full24.json \
  && run_fig fig3_anchor_audit

# fig4: residual eigenvalue spectrum
require_data fig4 ../v1a/diff_eig_L5.npy ../v1a/diff_eig_L11.npy \
  && run_fig fig4_diff_spectrum

# fig5: probe / SGD-diverges diagnostics (three specific runs)
require_data fig5 \
  ../v1a/probe_L11_shared_r8_lr3e-05_s500_c0.json \
  ../v1a/probe_L11_shared_r8_lr3e-05_s2000_c0.json \
  ../v1a/probe_L11_shared_r8_lr0.0001_s500_c1.json \
  && run_fig fig5_probe_diagnostics

# fig6: ridge vs rank
require_data fig6 ../v1b_ridge/ridge_results.json \
  && run_fig fig6_ridge_rank

# fig7: ridge over depth
require_data fig7 ../v1b_ridge/ridge_results.json \
  && run_fig fig7_ridge_depth

# fig8: ridge vs SGD finetune
require_data fig8 ../v1b_ridge/ridge_ft.json \
  && run_fig fig8_ridge_ft

# fig9: ridge calibration scaling
require_data fig9 ../v1b_ridge/ridge_scale.json \
  && run_fig fig9_ridge_scale

# fig11: SGD pressure test (#1)
require_data fig11 ../v1b_ridge/sgd_pressure.json \
  && run_fig fig11_sgd_pressure

echo "done."
