# seq=22: v1a smoke v2 -- shared AND per-head at L5 r8 after init fix (expect recovery>0)
python outputs/v1a/v1a_correction.py --mode train --layer 5 --variant shared  --rank 8 --batch_size 8 --force
python outputs/v1a/v1a_correction.py --mode train --layer 5 --variant perhead --rank 8 --batch_size 8 --force
for v in shared perhead; do
  echo "=== L5 $v r8 ==="
  python - "$v" <<'PY'
import json,sys
v=sys.argv[1]
d=json.load(open(f"outputs/v1a/L5_{v}_r8.json"))
print({k:round(d[k],4) for k in ['eval_loss','residual_delta','recovery_ratio','param_count_M','recovery_per_M']})
print("curve:",[round(x[1],3) for x in d['train_curve']])
PY
done
