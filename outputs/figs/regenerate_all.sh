#!/usr/bin/env bash
# Regenerate all conference-grade figures (pdf + png). Run from outputs/figs/.
set -e
cd "$(dirname "$0")"
# fig0_schematic.py is a matplotlib FALLBACK only. Primary Fig 0 source is
# fig0_schematic.tex (TikZ); compile that with pdflatex when LaTeX is available.
# Regenerate the fallback explicitly with: python3 fig0_schematic.py
for f in fig1_v0_heatmap fig2_depth_profile fig3_anchor_audit \
         fig4_diff_spectrum fig5_probe_diagnostics fig6_ridge_rank fig7_ridge_depth \
         fig8_ridge_ft fig9_ridge_scale; do
  echo "== $f =="; python3 "$f.py"
done
# fig11 depends on outputs/v1b_ridge/sgd_pressure.json from the #1 SGD pressure
# test. Skipped here unless that file exists, so a partial repo still renders.
if [ -f ../v1b_ridge/sgd_pressure.json ]; then
  echo "== fig11_sgd_pressure =="; python3 fig11_sgd_pressure.py
else
  echo "skip fig11_sgd_pressure (sgd_pressure.json not yet present)"
fi
echo "done."
