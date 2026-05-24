# seq=1: baseline eval (downloads model + streams dataset on first run)
python outputs/v_intervention.py --mode baseline --num_seq 1000 --ctxlen 1024 --batch_size 16
