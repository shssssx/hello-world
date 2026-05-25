# seq=6: copy CKA result files back into the repo so they round-trip to remote.
# (npz are 37MB each — NOT copied; only the small json/csv the paper needs.)
export OMP_NUM_THREADS=8
BASE=/root/autodl-tmp/artifacts_llama
DST="$(pwd)/option_b/results/cka"
mkdir -p "$DST"
cp -v "$BASE/multilayer_cka_aggregate.json" "$DST"/
for s in 1 2 3; do
  cp -v "$BASE/seed$s/multilayer_cka_summary.json" "$DST/seed${s}_multilayer_cka_summary.json"
  cp -v "$BASE/seed$s/multilayer_cka.csv"          "$DST/seed${s}_multilayer_cka.csv"
done
echo "--- delivered back ---"
ls -la "$DST"
