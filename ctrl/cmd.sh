# seq=26: lr stability probe on the worst diverger (L11) to pick a stable constant lr
LOG=outputs/v1a/lr_probe.txt; : > "$LOG"
for lr in 3e-4 1e-4 3e-5; do
  for cfg in "11 perhead 16" "11 shared 8" "5 shared 8"; do
    set -- $cfg
    echo "##### lr=$lr layer=$1 variant=$2 rank=$3" | tee -a "$LOG"
    python outputs/v1a/v1a_correction.py --mode train --layer $1 --variant $2 --rank $3 --lr "$lr" --batch_size 8 --force 2>&1 \
      | grep -E "\[run\]|step   0 |step 250 |step 499 " | tee -a "$LOG"
  done
done
git add outputs/v1a/lr_probe.txt
git commit -m "v1a lr probe log" || true
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  git pull --rebase -q origin claude/zen-allen-7Y8Bx 2>/dev/null || true
  git push -q origin claude/zen-allen-7Y8Bx 2>/dev/null && { echo SELFPUSH_OK; break; }
  sleep $((i*2))
done
