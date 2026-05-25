# seq=10: install sklearn, re-run Q3 linear probe (base npz already extracted), round-trip.
export OMP_NUM_THREADS=8
REPO="$(pwd)"
APP=/root/autodl-tmp/option_b_analyses
DST="$APP/scripts"
RESDIR="$APP/results/probe_assumption1"
mkdir -p "$RESDIR"

echo "=== install scikit-learn ==="
pip install --quiet scikit-learn 2>&1 | tail -3
python -c "import sklearn; print('sklearn', sklearn.__version__)" 2>&1

echo "=== Step2: linear probe (base vs teacher_w pooled, seeds 1/2/3) ==="
python "$DST/03_linear_probe_assumption1.py" \
    --base_npz /root/autodl-tmp/artifacts_llama/base_qwen_hidden_states.npz \
    --teacher_npz_list \
        /root/autodl-tmp/artifacts_llama/seed1/qwen_hidden_states.npz \
        /root/autodl-tmp/artifacts_llama/seed2/qwen_hidden_states.npz \
        /root/autodl-tmp/artifacts_llama/seed3/qwen_hidden_states.npz \
    --output "$RESDIR/probe.csv" \
    2>&1 | tee "$RESDIR/probe.log"

echo "=== round-trip results into repo ==="
RES="$REPO/option_b/results/probe_assumption1"
mkdir -p "$RES"
cp -v "$RESDIR/"* "$RES"/ 2>&1
ls -la "$RES"
