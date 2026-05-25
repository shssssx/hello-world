# seq=32: SVD/covariance spectrum of (V_real - V_table) at L5 vs L11 -- intrinsic rank
rm -f outputs/v1a/svd.done
python outputs/v1a/v1a_correction.py --mode svd_diff --layers 5,11 --batch_size 16
echo "===== svd_diff.json ====="; cat outputs/v1a/svd_diff.json
echo done > outputs/v1a/svd.done
git add outputs/v1a
git commit -m "v1a svd_diff results" || true
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  git pull --rebase -q origin claude/zen-allen-7Y8Bx 2>/dev/null || true
  git push -q origin claude/zen-allen-7Y8Bx 2>/dev/null && { echo SELFPUSH_OK; break; }
  sleep $((i*2))
done
