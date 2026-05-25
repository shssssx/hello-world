# seq=29: norm-capped dV probe -- does bounding ||dV|| let L11 learn & generalize? (vs L5 control)
rm -f outputs/v1a/capprobe.done
python outputs/v1a/v1a_correction.py --mode probe --cap_matrix --batch_size 8
echo done > outputs/v1a/capprobe.done
git add outputs/v1a
git commit -m "v1a norm-capped probe results" || true
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  git pull --rebase -q origin claude/zen-allen-7Y8Bx 2>/dev/null || true
  git push -q origin claude/zen-allen-7Y8Bx 2>/dev/null && { echo SELFPUSH_OK; break; }
  sleep $((i*2))
done
