# seq=2: enable pile (zstd), reset stale 246-seq cache, re-run baseline at 1000 seqs
pip install -q zstandard 2>&1 | tail -1
rm -f outputs/losses_per_seq.pt outputs/baseline_loss.json outputs/coarse_loss_delta.npy outputs/fine_loss_delta.npy
python outputs/v_intervention.py --mode baseline --num_seq 1000 --ctxlen 1024 --batch_size 16
