# seq=36: v1b pilot -- fitted A1 anchor + small trained correction (L11, L5)
rm -f outputs/v1a/v1b.done
python outputs/v1a/v1a_correction.py --mode v1b --layers 11,5 --ranks 16,64 --lr 1e-4 --steps 500 --batch_size 8
echo "===== v1b_anchor_plus_corr.json ====="; cat outputs/v1a/v1b_anchor_plus_corr.json
echo done > outputs/v1a/v1b.done
git add outputs/v1a
git commit -m "v1b pilot results" || true
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  git pull --rebase -q origin claude/zen-allen-7Y8Bx 2>/dev/null || true
  git push -q origin claude/zen-allen-7Y8Bx 2>/dev/null && { echo SELFPUSH_OK; break; }
  sleep $((i*2))
done
