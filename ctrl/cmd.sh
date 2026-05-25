# handshake seq=1: confirm runner is alive and probe the 3-GPU environment
echo "[handshake] runner is alive"
echo "host: $(hostname)"
echo "cwd:  $(pwd)"
echo "date: $(date -u +%FT%TZ)"
echo "python: $(python --version 2>&1)"
echo "--- torch / cuda ---"
python - <<'PY' 2>&1
try:
    import torch
    print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(),
          "device_count", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print("  gpu", i, torch.cuda.get_device_name(i))
except Exception as e:
    print("torch import failed:", e)
PY
echo "--- transformers / datasets ---"
python -c "import transformers, datasets; print('transformers', transformers.__version__, '| datasets', datasets.__version__)" 2>&1 || echo "deps not installed"
echo "--- nvidia-smi ---"
nvidia-smi 2>&1 || echo "no nvidia-smi"
echo "--- disk (cwd + /root/autodl-tmp) ---"
df -h . 2>&1
df -h /root/autodl-tmp 2>&1 || echo "no /root/autodl-tmp"
echo "--- look for option_b scripts + artifacts on this machine ---"
echo "[search scripts]"; find / -maxdepth 6 \( -iname '0[1-5]*_*.py' -o -iname '*option_b*' \) 2>/dev/null | grep -vi proc | head -40
echo "[search seed dirs]"; find / -maxdepth 6 -type d -iname 'seed[0-9]' 2>/dev/null | grep -vi proc | head -40
echo "[artifacts_llama]"; ls -la /root/autodl-tmp/artifacts_llama 2>&1 | head -40
