# seq=14: Q3 path B 5-split CV (mean±std ratio) + round-trip response JSONs.
export OMP_NUM_THREADS=8
REPO="$(pwd)"
APP=/root/autodl-tmp/option_b_analyses
DST="$APP/scripts"
RESDIR="$APP/results/probe_assumption1"
BASE=/root/autodl-tmp/artifacts_llama
mkdir -p "$RESDIR"

echo "=== run 03b over 5 splits (seed 0..4) ==="
for sd in 0 1 2 3 4; do
  python "$DST/03b_behavior_probe_pca.py" \
    --behavior_npz_list "$BASE/seed1/trait_behavior.npz" "$BASE/seed2/trait_behavior.npz" "$BASE/seed3/trait_behavior.npz" \
    --output "$RESDIR/bp_split${sd}.csv" --n_pca 20 --seed $sd \
    > "$RESDIR/bp_split${sd}.log" 2>&1
  echo "  split seed=$sd done (rc=$?)"
done

echo "=== aggregate across splits (mean +/- std) ==="
python - <<'PY'
import json, glob, numpy as np
files = sorted(glob.glob('/root/autodl-tmp/option_b_analyses/results/probe_assumption1/bp_split*.json'))
print(f"loaded {len(files)} splits")
per = [json.load(open(f))['per_layer'] for f in files]
nL = len(per[0])
lin = np.array([[p[l]['r2_linear'] for l in range(nL)] for p in per])        # (5, nL)
nl  = np.array([[p[l]['r2_best_nonlinear'] for l in range(nL)] for p in per]) # (5, nL)
print(f"{'layer':>5} {'lin_mean':>9} {'lin_std':>8} {'nl_mean':>9} {'nl_std':>8} {'ratio(meanlin/meannl)':>22}")
agg=[]
for l in range(nL):
    lm, ls = lin[:,l].mean(), lin[:,l].std(ddof=1)
    nm, ns = nl[:,l].mean(),  nl[:,l].std(ddof=1)
    ratio = lm/nm if nm>0 else float('nan')
    agg.append({'layer':l,'lin_mean':float(lm),'lin_std':float(ls),
                'nl_mean':float(nm),'nl_std':float(ns),'ratio':float(ratio)})
    print(f"{l:>5} {lm:>9.3f} {ls:>8.3f} {nm:>9.3f} {ns:>8.3f} {ratio:>22.3f}")

# headline 1: layer with max mean linear R2
best_lin = max(agg, key=lambda r: r['lin_mean'])
# deep trait band 15-28
deep = [r for r in agg if 15 <= r['layer'] <= 28 and r['nl_mean']>0]
deep_ratio = np.array([r['ratio'] for r in deep])
print("\n=== HEADLINE ===")
print(f"max-linear layer = {best_lin['layer']}: "
      f"lin={best_lin['lin_mean']:.3f}+/-{best_lin['lin_std']:.3f}  "
      f"nl={best_lin['nl_mean']:.3f}+/-{best_lin['nl_std']:.3f}  ratio={best_lin['ratio']:.3f}")
print(f"deep band (15-28) ratio: mean={deep_ratio.mean():.3f}  "
      f"median={np.median(deep_ratio):.3f}  min={deep_ratio.min():.3f}  max={deep_ratio.max():.3f}")
json.dump({'n_splits':len(files),'per_layer':agg,
           'max_linear_layer':best_lin,
           'deep_band_ratio_mean':float(deep_ratio.mean()),
           'deep_band_ratio_median':float(np.median(deep_ratio))},
          open('/root/autodl-tmp/option_b_analyses/results/probe_assumption1/behavior_probe_cv.json','w'), indent=2)
print("saved behavior_probe_cv.json")
PY

echo "=== round-trip CV outputs + response JSONs into repo ==="
RES="$REPO/option_b/results/probe_assumption1"
mkdir -p "$RES" "$RES/responses"
cp -v "$RESDIR/"bp_split*.csv "$RESDIR/"bp_split*.log "$RESDIR/"behavior_probe_cv.json "$RES"/ 2>&1
for s in 1 2 3; do
  cp -v "$BASE/seed$s/trait_behavior_responses.json" "$RES/responses/seed${s}_trait_behavior_responses.json" 2>&1
done
ls -la "$RES" "$RES/responses"
