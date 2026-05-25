# seq=13: Q3 path B v2 — redeliver 03b (MLP grid + KernelRidge RBF), re-run Step2 only.
export OMP_NUM_THREADS=8
REPO="$(pwd)"
SRC="$REPO/option_b/scripts"
APP=/root/autodl-tmp/option_b_analyses
DST="$APP/scripts"
RESDIR="$APP/results/probe_assumption1"
BASE=/root/autodl-tmp/artifacts_llama
mkdir -p "$DST" "$RESDIR"

echo "=== deliver 03b (v2) ==="
cp -v "$SRC/03b_behavior_probe_pca.py" "$DST"/
python -c "import py_compile; py_compile.compile('$DST/03b_behavior_probe_pca.py', doraise=True); print('OK')" 2>&1 || exit 1

echo "=== Step2 v2: behavior probe (MLP grid + KernelRidge RBF) ==="
python "$DST/03b_behavior_probe_pca.py" \
    --behavior_npz_list "$BASE/seed1/trait_behavior.npz" "$BASE/seed2/trait_behavior.npz" "$BASE/seed3/trait_behavior.npz" \
    --output "$RESDIR/behavior_probe_v2.csv" --n_pca 20 \
    2>&1 | tee "$RESDIR/behavior_probe_v2.log"

echo "=== round-trip v2 into repo ==="
RES="$REPO/option_b/results/probe_assumption1"
mkdir -p "$RES"
cp -v "$RESDIR/"behavior_probe_v2.* "$RES"/ 2>&1
ls -la "$RES"
