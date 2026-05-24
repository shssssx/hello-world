# seq=10: force modern Node (>=20) from conda-forge, reinstall Claude Code + Codex
echo "## upgrade nodejs"
conda install -y --override-channels -c conda-forge "nodejs>=20" 2>&1 | tail -6
echo "## versions"; node -v 2>&1; npm -v 2>&1
echo "## reinstall claude code + codex"
npm i -g @anthropic-ai/claude-code @openai/codex 2>&1 | tail -12
echo "## verify"
(command -v claude && claude --version) 2>&1 || echo "claude missing"
(command -v codex && codex --version) 2>&1 || echo "codex missing"
