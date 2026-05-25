# seq=39: closed-form ridge residual readout (L5,6,7,11,20) -- training-vs-representation confound
rm -f outputs/v1a/ridge.done
python outputs/v1a/v1a_correction.py --mode ridge --layers 5,6,7,11,20 --batch_size 16
echo "===== ridge_results.json ====="; cat outputs/v1b_ridge/ridge_results.json
echo done > outputs/v1a/ridge.done
git add outputs/v1a outputs/v1b_ridge
git commit -m "v1b ridge results" || true
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  git pull --rebase -q origin claude/zen-allen-7Y8Bx 2>/dev/null || true
  git push -q origin claude/zen-allen-7Y8Bx 2>/dev/null && { echo SELFPUSH_OK; break; }
  sleep $((i*2))
done
