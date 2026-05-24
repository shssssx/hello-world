# seq=6: per-head fine scan on a depth-stratified set of layers (low/mid/high)
python outputs/v_intervention.py --mode fine --layers 2,5,8,11,14,17,20,23 --num_seq 1000 --ctxlen 1024 --batch_size 16
