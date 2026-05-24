# seq=24: v1a full grid -- layer 11
python outputs/v1a/v1a_correction.py --mode all --layers 11 --batch_size 8
python - <<'PY'
import json,glob,os
for f in sorted(glob.glob("outputs/v1a/L11_*.json")):
    d=json.load(open(f))
    print(f"{d['variant']:7s} r{d['rank']:<2d} resid={d['residual_delta']:+.3f} rec={d['recovery_ratio']:+.3f} pM={d['param_count_M']:.4f} rec/M={d['recovery_per_M']:+.2f}")
PY
