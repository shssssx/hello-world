# seq=40: ridge-init ZERO-SHOT deployable correction probe (no SGD) -- L5,6,7,11 x ranks x caps
rm -f outputs/v1a/ridgeinit.done
python outputs/v1a/v1a_correction.py --mode ridge_init --layers 5,6,7,11 --batch_size 16
echo "===== ridge_init_zeroshot.json ====="; cat outputs/v1b_ridge/ridge_init_zeroshot.json
echo done > outputs/v1a/ridgeinit.done
git add outputs/v1a outputs/v1b_ridge
git commit -m "v1b ridge-init zero-shot results" || true
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  git pull --rebase -q origin claude/zen-allen-7Y8Bx 2>/dev/null || true
  git push -q origin claude/zen-allen-7Y8Bx 2>/dev/null && { echo SELFPUSH_OK; break; }
  sleep $((i*2))
done
