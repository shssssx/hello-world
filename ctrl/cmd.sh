# seq=27: v1a grid L5,L11 at stable lr=3e-5 (force overwrite diverged lr=1e-3 runs)
rm -f outputs/v1a/chunk_5_11.done
python outputs/v1a/v1a_correction.py --mode all --layers 5,11 --lr 3e-5 --force --batch_size 8
echo done > outputs/v1a/chunk_5_11.done
git add outputs/v1a
git commit -m "v1a results L5,L11 @lr3e-5" || true
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  git pull --rebase -q origin claude/zen-allen-7Y8Bx 2>/dev/null || true
  git push -q origin claude/zen-allen-7Y8Bx 2>/dev/null && { echo SELFPUSH_OK; break; }
  sleep $((i*2))
done
python - <<'PY'
import json,glob
for f in sorted(glob.glob("outputs/v1a/L{5,11}_*.json".replace("{5,11}","[15]*"))):
    d=json.load(open(f)); print(f"{f.split('/')[-1]:22s} lr={d.get('lr')} rec={d['recovery_ratio']:+.3f} resid={d['residual_delta']:+.3f}")
PY
