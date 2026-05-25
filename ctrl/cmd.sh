# seq=30: nonlinear bottleneck (mlp) correction probe -- does going beyond linear lift L11?
rm -f outputs/v1a/mlpprobe.done
python outputs/v1a/v1a_correction.py --mode probe --mlp_matrix --batch_size 8
echo done > outputs/v1a/mlpprobe.done
git add outputs/v1a
git commit -m "v1a mlp probe results" || true
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  git pull --rebase -q origin claude/zen-allen-7Y8Bx 2>/dev/null || true
  git push -q origin claude/zen-allen-7Y8Bx 2>/dev/null && { echo SELFPUSH_OK; break; }
  sleep $((i*2))
done
