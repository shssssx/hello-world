echo "PROBE alive $(date -u +%T)"; nvidia-smi -L 2>&1 | head -1 || echo no-gpu
