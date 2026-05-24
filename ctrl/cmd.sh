# seq=19: v1a blocking scale/norm sanity -- V stats (eval + train) then rescaled coarse
python outputs/v1a/v1a_correction.py --mode scale_stats --split eval  --batch_size 16
python outputs/v1a/v1a_correction.py --mode scale_stats --split train --batch_size 16
python outputs/v1a/v1a_correction.py --mode scale_coarse --batch_size 16
echo "===== scale_sanity.md ====="
cat outputs/v1a/scale_sanity.md
