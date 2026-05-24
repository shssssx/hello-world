# seq=14: diagnose server reachability to Anthropic / OpenAI + any proxy env
echo "## proxy env"; env | grep -iE "proxy" || echo "(no proxy env)"
echo "## anthropic"; curl -s -o /dev/null -w "api.anthropic.com -> %{http_code} (%{time_total}s)\n" --max-time 12 https://api.anthropic.com/v1/models 2>&1 || echo "anthropic unreachable"
echo "## openai"; curl -s -o /dev/null -w "api.openai.com -> %{http_code} (%{time_total}s)\n" --max-time 12 https://api.openai.com/v1/models 2>&1 || echo "openai unreachable"
echo "## auth.openai"; curl -s -o /dev/null -w "auth.openai.com -> %{http_code} (%{time_total}s)\n" --max-time 12 https://auth.openai.com 2>&1 || echo "auth.openai unreachable"
echo "## google (control)"; curl -s -o /dev/null -w "google -> %{http_code} (%{time_total}s)\n" --max-time 12 https://www.google.com 2>&1 || echo "google unreachable"
echo "## hf (already used)"; curl -s -o /dev/null -w "hf-mirror -> %{http_code} (%{time_total}s)\n" --max-time 12 https://hf-mirror.com 2>&1 || echo "hf-mirror unreachable"
