# seq=5: sanity (expect layer-0 ~0) + coarse scan, with hook reshape fixed
echo "########## SANITY ##########"
python outputs/v_intervention.py --mode sanity --num_seq 1000 --ctxlen 1024 --batch_size 16
echo "########## COARSE ##########"
python outputs/v_intervention.py --mode coarse --num_seq 1000 --ctxlen 1024 --batch_size 16
