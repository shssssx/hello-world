# seq=3: inspect each seed dir structure (read-only)
export OMP_NUM_THREADS=8
BASE=/root/autodl-tmp/artifacts_llama
for s in 0 1 2 3; do
  echo "========== seed$s =========="
  ls -la "$BASE/seed$s" 2>&1
  echo "-- subdir check --"
  for sub in teacher_model student_model; do
    if [ -d "$BASE/seed$s/$sub" ]; then
      echo "[OK] $sub/ present:"; ls "$BASE/seed$s/$sub" 2>&1 | head -20
    else
      echo "[MISSING] $sub/"
    fi
  done
  echo "-- step json files --"
  ls "$BASE/seed$s"/step*.json 2>&1
  echo "-- du --"
  du -sh "$BASE/seed$s" 2>&1
done
