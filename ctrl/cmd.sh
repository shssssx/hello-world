# seq=20: v1a SMOKE TEST -- single run, verify training is healthy before the full 40
python outputs/v1a/v1a_correction.py --mode train --layer 5 --variant shared --rank 8 --batch_size 8
echo "===== L5_shared_r8.json ====="
cat outputs/v1a/L5_shared_r8.json
