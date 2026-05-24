# seq=4: layer-0 sanity (expect ~0), then coarse per-layer scan (24x)
echo "########## SANITY ##########"
python outputs/v_intervention.py --mode sanity --num_seq 1000 --ctxlen 1024 --batch_size 16
echo "########## COARSE ##########"
python outputs/v_intervention.py --mode coarse --num_seq 1000 --ctxlen 1024 --batch_size 16
