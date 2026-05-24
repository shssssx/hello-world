# seq=16: pull codex-MCP wiring guide + SSH/server config snippets
cd ~/aris_repo
echo "===== CODEX_CLAUDE_REVIEW_GUIDE_CN (1..130) ====="
sed -n '1,130p' docs/CODEX_CLAUDE_REVIEW_GUIDE_CN.md 2>/dev/null
echo "===== 'claude mcp add' / 'codex mcp' occurrences ====="
grep -rIn -iE "claude mcp add|codex mcp|mcp-config|\.mcp\.json" docs README_CN.md AGENT_GUIDE.md 2>/dev/null | head -25
echo "===== SSH / GPU-server config mentions ====="
grep -rIn -iE "ssh -p|ssh root@|remote.{0,12}server|GPU.{0,12}server|CLAUDE\.md" docs/PROJECT_FILES_GUIDE_CN.md README_CN.md 2>/dev/null | head -25
