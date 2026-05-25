# seq=45: Pythia-160m replication (rerun with hook hygiene)
rm -f outputs/v1b_160m/repro.done
python outputs/v1b_160m/repro160.py
echo "===== repro160.json ====="; cat outputs/v1b_160m/repro160.json
echo done > outputs/v1b_160m/repro.done
git add outputs/v1b_160m
git commit -m "v1b 160m replication results" || true
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  git pull --rebase -q origin claude/zen-allen-7Y8Bx 2>/dev/null || true
  git push -q origin claude/zen-allen-7Y8Bx 2>/dev/null && { echo SELFPUSH_OK; break; }
  sleep $((i*2))
done
