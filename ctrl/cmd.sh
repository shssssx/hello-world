# handshake: confirm the runner can execute and report the environment
echo "[handshake] runner is alive"
echo "cwd: $(pwd)"
echo "python: $(python --version 2>&1)"
echo "torch: $(python -c 'import torch;print(torch.__version__, torch.cuda.is_available())' 2>&1)"
echo "--- nvidia-smi ---"
nvidia-smi -L 2>&1 || echo "no nvidia-smi"
echo "--- pip check (transformers/datasets) ---"
python -c "import transformers, datasets; print('transformers', transformers.__version__, '| datasets', datasets.__version__)" 2>&1 || echo "deps not yet installed"
