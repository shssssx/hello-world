# seq=4: deliver Option B scripts to /root/autodl-tmp + re-inspect seed dirs
export OMP_NUM_THREADS=8
set -e
SRC="$(pwd)/option_b/scripts"
DST=/root/autodl-tmp/option_b_analyses/scripts
mkdir -p "$DST"
cp -v "$SRC"/*.py "$DST"/
echo "--- delivered scripts ---"
ls -la "$DST"
echo "--- python syntax check (compile only) ---"
for f in "$DST"/*.py; do
  python -c "import py_compile,sys; py_compile.compile('$f', doraise=True); print('OK', '$f')" 2>&1 || echo "SYNTAX FAIL $f"
done
set +e
echo "--- re-inspect seed dirs ---"
BASE=/root/autodl-tmp/artifacts_llama
for s in 0 1 2 3; do
  echo "========== seed$s =========="
  ls -la "$BASE/seed$s" 2>&1
  for sub in teacher_model student_model; do
    if [ -d "$BASE/seed$s/$sub" ]; then echo "[OK] $sub/"; else echo "[MISSING] $sub/"; fi
  done
  ls "$BASE/seed$s"/step*.json 2>&1
  du -sh "$BASE/seed$s" 2>&1
done
