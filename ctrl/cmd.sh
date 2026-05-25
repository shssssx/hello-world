# seq=7: reconnaissance for Q3 — is base Qwen2.5-7B-Instruct cached on AutoDL?
export OMP_NUM_THREADS=8
echo "=== ls /root/autodl-tmp/models ==="
ls -la /root/autodl-tmp/models/ 2>&1
echo "=== ls /root/autodl-tmp/models/Qwen ==="
ls -la /root/autodl-tmp/models/Qwen/ 2>&1
echo "=== find Qwen2.5-7B-Instruct dirs ==="
find /root/autodl-tmp / -maxdepth 6 -type d -iname "*Qwen2.5-7B-Instruct*" 2>/dev/null | grep -vi proc | head -20
echo "=== HF hub cache ==="
ls -la /root/.cache/huggingface/hub/ 2>&1 | head -30
find /root/.cache/huggingface -maxdepth 4 -type d -iname "*Qwen2.5-7B*" 2>/dev/null | head -20
echo "=== teacher_model config (what base were these FT'd from?) ==="
head -40 /root/autodl-tmp/artifacts_llama/seed1/teacher_model/config.json 2>&1
echo "=== sizes / disk ==="
du -sh /root/autodl-tmp/models/ 2>/dev/null || echo "no models dir"
df -h /root/autodl-tmp 2>&1
