# seq=7: build heatmap.png + summary.md from coarse/fine arrays
python outputs/v_intervention.py --mode summary --num_seq 1000 --ctxlen 1024
echo "########## summary.md ##########"
cat outputs/summary.md
