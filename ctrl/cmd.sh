# seq=11: Q3 path A — trait-shift geometry test (CPU, uses existing npz). Deliver + run + round-trip.
export OMP_NUM_THREADS=8
REPO="$(pwd)"
APP=/root/autodl-tmp/option_b_analyses
DST="$APP/scripts"
RESDIR="$APP/results/probe_assumption1"
mkdir -p "$DST" "$RESDIR"

echo "=== deliver 04a ==="
cp -v "$REPO/option_b/scripts/04a_trait_shift_geometry.py" "$DST"/
python -c "import py_compile; py_compile.compile('$DST/04a_trait_shift_geometry.py', doraise=True); print('OK')" 2>&1 || exit 1

echo "=== run trait-shift geometry ==="
python "$DST/04a_trait_shift_geometry.py" \
    --base_npz /root/autodl-tmp/artifacts_llama/base_qwen_hidden_states.npz \
    --teacher_npz_list \
        /root/autodl-tmp/artifacts_llama/seed1/qwen_hidden_states.npz \
        /root/autodl-tmp/artifacts_llama/seed2/qwen_hidden_states.npz \
        /root/autodl-tmp/artifacts_llama/seed3/qwen_hidden_states.npz \
    --output "$RESDIR/shift_geometry.json" \
    2>&1 | tee "$RESDIR/shift_geometry.log"

echo "=== round-trip into repo ==="
RES="$REPO/option_b/results/probe_assumption1"
mkdir -p "$RES"
cp -v "$RESDIR/shift_geometry.json" "$RESDIR/shift_geometry.log" "$RES"/ 2>&1
ls -la "$RES"
