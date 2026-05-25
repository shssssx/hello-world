# seq=12: Q3 path B — generate+score teacher_w (3 GPU parallel) then PCA behavior probe.
export OMP_NUM_THREADS=8
REPO="$(pwd)"
SRC="$REPO/option_b/scripts"
APP=/root/autodl-tmp/option_b_analyses
DST="$APP/scripts"
RESDIR="$APP/results/probe_assumption1"
BASE=/root/autodl-tmp/artifacts_llama
mkdir -p "$DST" "$RESDIR"

echo "=== deliver 02c + 03b ==="
cp -v "$SRC/02c_generate_score_teacher.py" "$SRC/03b_behavior_probe_pca.py" "$DST"/
for f in "$DST/02c_generate_score_teacher.py" "$DST/03b_behavior_probe_pca.py"; do
  python -c "import py_compile; py_compile.compile('$f', doraise=True); print('OK', '$f')" 2>&1 || { echo "SYNTAX FAIL $f"; exit 1; }
done

echo "=== Step1: generate+score on seed1/2/3 (GPU 0/1/2 parallel) ==="
CUDA_VISIBLE_DEVICES=0 nohup python "$DST/02c_generate_score_teacher.py" --seed_dir "$BASE/seed1" --n_samples 5 --max_new_tokens 25 > "$RESDIR/gen_seed1.log" 2>&1 & P1=$!
CUDA_VISIBLE_DEVICES=1 nohup python "$DST/02c_generate_score_teacher.py" --seed_dir "$BASE/seed2" --n_samples 5 --max_new_tokens 25 > "$RESDIR/gen_seed2.log" 2>&1 & P2=$!
CUDA_VISIBLE_DEVICES=2 nohup python "$DST/02c_generate_score_teacher.py" --seed_dir "$BASE/seed3" --n_samples 5 --max_new_tokens 25 > "$RESDIR/gen_seed3.log" 2>&1 & P3=$!
wait $P1; R1=$?
wait $P2; R2=$?
wait $P3; R3=$?
echo "gen rc: seed1=$R1 seed2=$R2 seed3=$R3"
for s in 1 2 3; do echo "--- tail gen_seed$s.log ---"; tail -10 "$RESDIR/gen_seed$s.log"; done
echo "=== npz present? ==="
ls -la "$BASE"/seed1/trait_behavior.npz "$BASE"/seed2/trait_behavior.npz "$BASE"/seed3/trait_behavior.npz 2>&1

if [ "$R1" = 0 ] && [ "$R2" = 0 ] && [ "$R3" = 0 ]; then
  echo "=== Step2: PCA behavior probe ==="
  python "$DST/03b_behavior_probe_pca.py" \
    --behavior_npz_list "$BASE/seed1/trait_behavior.npz" "$BASE/seed2/trait_behavior.npz" "$BASE/seed3/trait_behavior.npz" \
    --output "$RESDIR/behavior_probe.csv" --n_pca 20 \
    2>&1 | tee "$RESDIR/behavior_probe.log"
else
  echo "!! generation failed on a seed; skipping probe."
fi

echo "=== round-trip into repo ==="
RES="$REPO/option_b/results/probe_assumption1"
mkdir -p "$RES"
cp -v "$RESDIR/"gen_seed*.log "$RESDIR/"behavior_probe.* "$RES"/ 2>&1
ls -la "$RES"
