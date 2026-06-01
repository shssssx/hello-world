# seq=51: LR sensitivity probe (framing flip #2 follow-up).
# Cap fix from seq=50 was empirically WORSE (ridge_init_ft collapsed deeper).
# v1a_correction.py reverted to soft_nograd cap (with --cap_mode flag).
# This probe: 2 inits (ridge / random) x 4 lrs (1e-5, 3e-5, 1e-4, 3e-4) x
# 3 layers (L6, L7, L11) = 24 cells, 5000 step each, r=64, cap=0.5,
# soft_nograd cap. ~2-3h on 4090. Output: outputs/v1b_ridge/lr_probe.json.
#
# Predicted (outcome 1, ~60%):
#   ridge_init @ lr=3e-5  -> ~0.78/0.74/0.56 (matches ridge_ft.json sanity)
#   random_init across all 4 lrs -> all < 0.1 (apples-to-apples failure)
#   => paper claim "across 100x lr range, SGD reach not the ridge" is clean
#
# Outcome 2 (~25%): some lr makes random_init recover -> framing flip
# Outcome 3 (~15%): ridge_init never recovers -> deeper debug needed
rm -f outputs/v1b_ridge/lr_probe.done
for i in 1 2 3 4 5; do git fetch -q origin claude/zen-allen-7Y8Bx && break || sleep 3; done
git checkout -q origin/claude/zen-allen-7Y8Bx -- outputs/v1a/v1a_correction.py
python outputs/v1a/v1a_correction.py --mode sgd_lr_probe --layers 6,7,11 --batch_size 8 > /tmp/lrprobe.log 2>&1
echo "===== lr_probe.json ====="; cat outputs/v1b_ridge/lr_probe.json
echo "----- [lrprobe] lines -----"; grep "\[lrprobe\]" /tmp/lrprobe.log | tail -40
echo done > outputs/v1b_ridge/lr_probe.done
git add outputs/v1b_ridge
git commit -m "v1b lr sensitivity probe results (seq=51)" || true
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  git pull --rebase -q origin claude/zen-allen-7Y8Bx 2>/dev/null || true
  git push -q origin claude/zen-allen-7Y8Bx 2>/dev/null && { echo SELFPUSH_OK; break; }
  sleep $((i*2))
done
