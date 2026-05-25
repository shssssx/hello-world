# seq=35: anchor audit + oracle (rerun after dtype fix)
rm -f outputs/v1a/anchor.done
python outputs/v1a/v1a_correction.py --mode anchor --layers 5,11,17,23 --batch_size 16
echo "===== anchor_audit.json ====="; cat outputs/v1a/anchor_audit.json
echo done > outputs/v1a/anchor.done
git add outputs/v1a
git commit -m "v1a anchor audit + oracle results" || true
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  git pull --rebase -q origin claude/zen-allen-7Y8Bx 2>/dev/null || true
  git push -q origin claude/zen-allen-7Y8Bx 2>/dev/null && { echo SELFPUSH_OK; break; }
  sleep $((i*2))
done
