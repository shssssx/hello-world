# seq=28: v1a stability probe -- 8 configs (L11 shared, lr/steps/clip matrix) with full diagnostics
rm -f outputs/v1a/probe.done
python outputs/v1a/v1a_correction.py --mode probe --batch_size 8
echo done > outputs/v1a/probe.done
git add outputs/v1a
git commit -m "v1a stability probe results" || true
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  git pull --rebase -q origin claude/zen-allen-7Y8Bx 2>/dev/null || true
  git push -q origin claude/zen-allen-7Y8Bx 2>/dev/null && { echo SELFPUSH_OK; break; }
  sleep $((i*2))
done
