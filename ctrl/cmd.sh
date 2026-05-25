# seq=9: Q3 — deliver 02b/03, extract base Qwen hidden states (GPU0), run linear probe,
# round-trip results into the repo.
export OMP_NUM_THREADS=8
REPO="$(pwd)"
SRC="$REPO/option_b/scripts"
APP=/root/autodl-tmp/option_b_analyses
DST="$APP/scripts"
RESDIR="$APP/results/probe_assumption1"
mkdir -p "$DST" "$RESDIR"

echo "=== deliver updated scripts ==="
cp -v "$SRC/02b_extract_base_qwen.py" "$SRC/03_linear_probe_assumption1.py" "$DST"/
echo "=== syntax check ==="
for f in "$DST/02b_extract_base_qwen.py" "$DST/03_linear_probe_assumption1.py"; do
  python -c "import py_compile; py_compile.compile('$f', doraise=True); print('OK', '$f')" 2>&1 || { echo "SYNTAX FAIL $f"; exit 1; }
done

echo "=== Step1: extract base Qwen hidden states (GPU0) ==="
CUDA_VISIBLE_DEVICES=0 python "$DST/02b_extract_base_qwen.py" \
    --base_model_path /root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct \
    --output_npz /root/autodl-tmp/artifacts_llama/base_qwen_hidden_states.npz \
    2>&1 | tee "$RESDIR/extract_base.log"

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
