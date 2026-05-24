# seq=9: install Node.js (conda-forge) + Claude Code + Codex CLI; non-interactive
echo "## install nodejs"
conda install -y -c conda-forge nodejs 2>&1 | tail -6
echo "## node/npm versions"; node -v 2>&1; npm -v 2>&1
echo "## npm global prefix"; npm config get prefix 2>&1
echo "## install Claude Code + Codex globally"
npm i -g @anthropic-ai/claude-code @openai/codex 2>&1 | tail -10
echo "## verify"
(command -v claude && claude --version) 2>&1 || echo "claude missing"
(command -v codex && codex --version) 2>&1 || echo "codex missing"
