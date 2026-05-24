# seq=23: v1a full grid -- layer 5 (2 variants x 5 ranks; skips the 2 smoke runs)
python outputs/v1a/v1a_correction.py --mode all --layers 5 --batch_size 8
echo "===== L5 jsons so far ====="
for f in outputs/v1a/L5_*.json; do python - "$f" <<'PY'
import json,sys; d=json.load(open(sys.argv[1]))
print(f"{d['variant']:7s} r{d['rank']:<2d} resid={d['residual_delta']:+.3f} rec={d['recovery_ratio']:+.3f} pM={d['param_count_M']:.4f} rec/M={d['recovery_per_M']:+.2f}")
PY
done
