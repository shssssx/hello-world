# seq=53: 1.4B Step 5 completion (random_vs_ridge at L7) with BS=4.
# seq=52 OOM'd silently at this step with BS=8 backward through 1.4B.
# Output: patches outputs/v1b_1_4b/repro14b.json random_vs_ridge_L7 entry.
# ~15-20 min on 4090 at fp16.
rm -f outputs/v1b_1_4b/step5_l7.done
for i in 1 2 3 4 5; do git fetch -q origin claude/zen-allen-7Y8Bx && break || sleep 3; done
git checkout -q origin/claude/zen-allen-7Y8Bx -- outputs/v1a/v1a_correction.py outputs/v1b_1_4b/step5_l7.py outputs/v1b_1_4b/repro14b.json
python outputs/v1b_1_4b/step5_l7.py > /tmp/step5_l7.log 2>&1
RC=$?
echo "===== python rc=$RC ====="
echo "===== repro14b.json random_vs_ridge_L7 ====="; python -c "import json; d=json.load(open('outputs/v1b_1_4b/repro14b.json')); print(json.dumps(d.get('random_vs_ridge_L7'), indent=2))"
echo "----- [step5] lines -----"; grep "\[step5\]" /tmp/step5_l7.log | tail -20
echo "----- log tail (incl any traceback) -----"; tail -30 /tmp/step5_l7.log
echo done > outputs/v1b_1_4b/step5_l7.done
git add outputs/v1b_1_4b
git commit -m "v1b 1.4b step 5 results (seq=53): random_vs_ridge_L7 at BS=4" || true
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  git pull --rebase -q origin claude/zen-allen-7Y8Bx 2>/dev/null || true
  git push -q origin claude/zen-allen-7Y8Bx 2>/dev/null && { echo SELFPUSH_OK; break; }
  sleep $((i*2))
done
