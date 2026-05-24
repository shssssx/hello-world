# seq=15: extract ARIS setup specifics (codex MCP wiring, SSH/experiment-bridge config)
cd ~/aris_repo
echo "## top-level"; ls
echo "## root md files"; ls *.md 2>/dev/null
echo "## docs dir"; ls docs 2>/dev/null | head -40
echo "## files mentioning codex+mcp"; grep -rIl -iE "codex.*mcp|mcp.*codex|claude mcp add" . 2>/dev/null | head -15
echo "## files mentioning ssh/remote server config"; grep -rIl -iE "ssh -p|remote.*gpu|server.*ssh|experiment-bridge" --include=*.md . 2>/dev/null | head -15
echo "===== experiment-bridge SKILL (head) ====="; sed -n '1,60p' skills/experiment-bridge/SKILL.md 2>/dev/null
