# seq=5: CKA pipeline (Q4) — extract hidden states (02a) on seeds 1/2/3 across
# GPU 0/1/2 in parallel, then per-seed CKA (02) + aggregate. seed0 excluded (no models).
export OMP_NUM_THREADS=8
cd /root/autodl-tmp/option_b_analyses/scripts || { echo "scripts dir missing"; exit 1; }
BASE=/root/autodl-tmp/artifacts_llama

echo "=== launch 02a extraction: seed1->gpu0, seed2->gpu1, seed3->gpu2 ==="
CUDA_VISIBLE_DEVICES=0 nohup python 02a_extract_qwen_hidden_states.py --seed_dir "$BASE/seed1" > "$BASE/extract1.log" 2>&1 & P1=$!
CUDA_VISIBLE_DEVICES=1 nohup python 02a_extract_qwen_hidden_states.py --seed_dir "$BASE/seed2" > "$BASE/extract2.log" 2>&1 & P2=$!
CUDA_VISIBLE_DEVICES=2 nohup python 02a_extract_qwen_hidden_states.py --seed_dir "$BASE/seed3" > "$BASE/extract3.log" 2>&1 & P3=$!
echo "pids: $P1 $P2 $P3 ; waiting..."
wait $P1; R1=$?
wait $P2; R2=$?
wait $P3; R3=$?
echo "extract rc: seed1=$R1 seed2=$R2 seed3=$R3"

for s in 1 2 3; do
  echo "--- tail extract$s.log ---"; tail -12 "$BASE/extract$s.log"
done

echo "=== npz produced? ==="
ls -la "$BASE"/seed1/qwen_hidden_states.npz "$BASE"/seed2/qwen_hidden_states.npz "$BASE"/seed3/qwen_hidden_states.npz 2>&1

if [ "$R1" = 0 ] && [ "$R2" = 0 ] && [ "$R3" = 0 ]; then
  echo "=== per-seed CKA ==="
  for s in 1 2 3; do python 02_multilayer_cka.py --seed_dir "$BASE/seed$s"; done
  echo "=== aggregate (seeds 1 2 3) ==="
  python 02_multilayer_cka.py --aggregate --base_dir "$BASE" --seeds 1 2 3
  echo "=== aggregate json dump ==="
  cat "$BASE/multilayer_cka_aggregate.json"
else
  echo "!! extraction failed on at least one seed; skipping CKA. See logs above."
fi
