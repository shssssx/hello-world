# seq=8: test whether AutoDL can reach OpenAI + whether key is visible to relay shell
echo "=== AutoDL -> api.openai.com:443 connectivity ==="
python - <<'PY'
import socket
try:
    s = socket.create_connection(("api.openai.com", 443), timeout=10)
    print("RESULT: AutoDL -> api.openai.com:443 OPEN")
    s.close()
except Exception as e:
    print("RESULT: AutoDL -> OpenAI FAILED:", repr(e))
PY
echo "=== is OPENAI_API_KEY visible to the (non-interactive) relay shell? ==="
if [ -n "${OPENAI_API_KEY:-}" ]; then echo "RESULT: OPENAI_API_KEY present (len=${#OPENAI_API_KEY})"; else echo "RESULT: OPENAI_API_KEY NOT visible to relay shell"; fi
