# seq=47: 160m -- FORCE-sync latest scripts first (runner reset was stale), then run
rm -f outputs/v1b_160m/repro.done
for i in 1 2 3 4 5; do git fetch -q origin claude/zen-allen-7Y8Bx && break || sleep 3; done
git checkout -q origin/claude/zen-allen-7Y8Bx -- outputs/v1b_160m/repro160.py outputs/v1a/v1a_correction.py
echo "## repro160 line 65:"; sed -n '65p' outputs/v1b_160m/repro160.py
python outputs/v1b_160m/repro160.py > /tmp/r160.log 2>&1; tail -5 /tmp/r160.log
echo "## key lines:"; grep -E "\[diag\]|\[160m\]|Error|Traceback" /tmp/r160.log | tail -45
echo done > outputs/v1b_160m/repro.done
git add outputs/v1b_160m
git commit -m "v1b 160m results (forced sync)" || true
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  git pull --rebase -q origin claude/zen-allen-7Y8Bx 2>/dev/null || true
  git push -q origin claude/zen-allen-7Y8Bx 2>/dev/null && { echo SELFPUSH_OK; break; }
  sleep $((i*2))
done
