# seq=46: 160m with hook diagnostics
rm -f outputs/v1b_160m/repro.done
python outputs/v1b_160m/repro160.py 2>&1 | tail -40
echo done > outputs/v1b_160m/repro.done
git add outputs/v1b_160m
git commit -m "v1b 160m results (diag)" || true
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  git pull --rebase -q origin claude/zen-allen-7Y8Bx 2>/dev/null || true
  git push -q origin claude/zen-allen-7Y8Bx 2>/dev/null && { echo SELFPUSH_OK; break; }
  sleep $((i*2))
done
