# seq=33: overfit test -- train-recovery vs eval-recovery (data/generalization vs architecture)
rm -f outputs/v1a/overfit.done
python outputs/v1a/v1a_correction.py --mode probe --overfit_matrix --batch_size 8
echo done > outputs/v1a/overfit.done
git add outputs/v1a
git commit -m "v1a overfit test results" || true
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  git pull --rebase -q origin claude/zen-allen-7Y8Bx 2>/dev/null || true
  git push -q origin claude/zen-allen-7Y8Bx 2>/dev/null && { echo SELFPUSH_OK; break; }
  sleep $((i*2))
done
