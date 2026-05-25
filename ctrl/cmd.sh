# seq=41: tiny CE finetune -- ridge-init(notrain) vs ridge-init+ft vs random-init+ft (L6,7,11, r64, cap0.5)
rm -f outputs/v1a/ridgeft.done
python outputs/v1a/v1a_correction.py --mode ridge_ft --layers 6,7,11 --lr 3e-5 --steps 500 --batch_size 8
echo "===== ridge_ft.json ====="; cat outputs/v1b_ridge/ridge_ft.json
echo done > outputs/v1a/ridgeft.done
git add outputs/v1a outputs/v1b_ridge
git commit -m "v1b ridge finetune results" || true
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  git pull --rebase -q origin claude/zen-allen-7Y8Bx 2>/dev/null || true
  git push -q origin claude/zen-allen-7Y8Bx 2>/dev/null && { echo SELFPUSH_OK; break; }
  sleep $((i*2))
done
