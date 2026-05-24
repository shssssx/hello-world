# seq=21: v1a smoke test retry (L5 shared r8) after guard fix
python outputs/v1a/v1a_correction.py --mode train --layer 5 --variant shared --rank 8 --batch_size 8
echo "===== L5_shared_r8.json ====="
cat outputs/v1a/L5_shared_r8.json
