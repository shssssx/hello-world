#!/usr/bin/env bash
# Regenerate all conference-grade figures (pdf + png). Run from outputs/figs/.
set -e
cd "$(dirname "$0")"
for f in fig1_v0_heatmap fig2_depth_profile fig3_anchor_audit fig4_diff_spectrum \
         fig5_probe_diagnostics fig6_ridge_rank fig7_ridge_depth fig8_ridge_ft \
         fig9_ridge_scale fig10_repro160; do
  echo "== $f =="; python3 "$f.py"
done
echo "done."
