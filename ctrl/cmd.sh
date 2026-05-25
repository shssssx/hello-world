# seq=25: v1a grid layers 5,11 (both variants now) + robust self-push of results
python outputs/v1a/v1a_correction.py --mode all --layers 5,11 --batch_size 8
git add outputs/v1a
git commit -m "v1a results L5,L11 (self-push)" || true
ok=0
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  git pull --rebase -q origin claude/zen-allen-7Y8Bx 2>/dev/null || true
  if git push -q origin claude/zen-allen-7Y8Bx 2>/dev/null; then ok=1; echo "SELFPUSH_OK i=$i"; break; fi
  sleep $((i*2))
done
[ "$ok" = 1 ] || echo "SELFPUSH_FAILED"
python - <<'PY'
import json,glob
for f in sorted(glob.glob("outputs/v1a/L*_*.json")):
    d=json.load(open(f)); print(f"{f.split('/')[-1]:24s} rec={d['recovery_ratio']:+.3f} resid={d['residual_delta']:+.3f} pM={d['param_count_M']:.4f}")
PY
