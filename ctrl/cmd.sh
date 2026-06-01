# seq=52: Pythia-1.4b-deduped cross-scale replication (third scale point for §5.8).
# Mirrors v1b_160m/repro160.py. Adds 3rd point to scale curve (160M / 410M / 1.4B).
# ~30-40 min on a 4090 at fp16, BS=8. Output: outputs/v1b_1_4b/repro14b.json.
#
# Predicted (high confidence given 160M / 410M replication):
#   - A1 anchor recovers most V-path ablation cost across depth
#   - U-shaped depth profile, trough mid-stack (likely L6-L9 area at 24-layer model)
#   - ridge r64 R_context at most context-bound layer ~0.6-0.8
#   - random-init CE finetune at that layer < 0.0 vs ridge-init ~0.6
rm -f outputs/v1b_1_4b/repro14b.done
for i in 1 2 3 4 5; do git fetch -q origin claude/zen-allen-7Y8Bx && break || sleep 3; done
git checkout -q origin/claude/zen-allen-7Y8Bx -- outputs/v1a/v1a_correction.py outputs/v1b_1_4b/repro14b.py
python outputs/v1b_1_4b/repro14b.py > /tmp/repro14b.log 2>&1
echo "===== repro14b.json ====="; cat outputs/v1b_1_4b/repro14b.json
echo "----- [1.4b] lines -----"; grep "\[1.4b\]" /tmp/repro14b.log | tail -40
echo done > outputs/v1b_1_4b/repro14b.done
git add outputs/v1b_1_4b
git commit -m "v1b 1.4b cross-scale results (seq=52)" || true
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  git pull --rebase -q origin claude/zen-allen-7Y8Bx 2>/dev/null || true
  git push -q origin claude/zen-allen-7Y8Bx 2>/dev/null && { echo SELFPUSH_OK; break; }
  sleep $((i*2))
done
