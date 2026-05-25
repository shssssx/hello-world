# seq=38: full-depth anchor audit, layers 1-23 (L0 trivial/zero-delta skipped)
rm -f outputs/v1a/anchor_full.done
python outputs/v1a/v1a_correction.py --mode anchor --layers 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 --batch_size 16
cp outputs/v1a/anchor_audit.json outputs/v1a/anchor_audit_full24.json
echo done > outputs/v1a/anchor_full.done
git add outputs/v1a
git commit -m "v1a anchor audit layers 1-23" || true
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  git pull --rebase -q origin claude/zen-allen-7Y8Bx 2>/dev/null || true
  git push -q origin claude/zen-allen-7Y8Bx 2>/dev/null && { echo SELFPUSH_OK; break; }
  sleep $((i*2))
done
