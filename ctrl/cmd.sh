# seq=18: complete the fine scan over the remaining 16 layers -> full 24-layer heatmap
python outputs/v_intervention.py --mode fine --layers 0,1,3,4,6,7,9,10,12,13,15,16,18,19,21,22 --num_seq 1000 --ctxlen 1024 --batch_size 16
echo "## fine coverage check"
python -c "import numpy as np; a=np.load('outputs/fine_loss_delta.npy'); print('measured rows:', int((~np.isnan(a)).any(1).sum()), '/ 24'); print('measured cells:', int((~np.isnan(a)).sum()), '/ 384')"
