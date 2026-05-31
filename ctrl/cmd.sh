# seq=50: #1 SGD pressure test RE-RUN with differentiable dv_cap (v1a_correction.py
# lines 194-198). Same matrix as seq=49 (6 variants x 3 layers x 5000 step). ~3h.
# Predicted: ridge_init_ft L6/7/11 climbs from (0.30,-0.23,-0.94) toward
# (~0.78,~0.74,~0.56) if cap was single root cause; partial recovery if lr=1e-4
# is also independently bad. See outputs/notes/paper_diffs.md for narrative branches.
rm -f outputs/v1b_ridge/sgd.done
for i in 1 2 3 4 5; do git fetch -q origin claude/zen-allen-7Y8Bx && break || sleep 3; done
git checkout -q origin/claude/zen-allen-7Y8Bx -- outputs/v1a/v1a_correction.py
python outputs/v1a/v1a_correction.py --mode sgd_pressure --layers 6,7,11 --steps 5000 --batch_size 8 > /tmp/sgd.log 2>&1
echo "===== sgd_pressure.json ====="; cat outputs/v1b_ridge/sgd_pressure.json
echo "----- [sgd] lines -----"; grep "\[sgd\]" /tmp/sgd.log | tail -30
echo done > outputs/v1b_ridge/sgd.done
git add outputs/v1b_ridge
git commit -m "v1b #1 sgd pressure results" || true
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  git pull --rebase -q origin claude/zen-allen-7Y8Bx 2>/dev/null || true
  git push -q origin claude/zen-allen-7Y8Bx 2>/dev/null && { echo SELFPUSH_OK; break; }
  sleep $((i*2))
done
