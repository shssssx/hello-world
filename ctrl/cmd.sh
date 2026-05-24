# seq=3: re-run baseline with fixed data loader (pile train + wikitext train fallback, cached)
rm -f outputs/eval_seqs_*.pt outputs/losses_per_seq.pt outputs/baseline_loss.json outputs/coarse_loss_delta.npy outputs/fine_loss_delta.npy
python outputs/v_intervention.py --mode baseline --num_seq 1000 --ctxlen 1024 --batch_size 16
