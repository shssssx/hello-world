# seq=11: install Node 22 via nvm (official prebuilt, avoids conda sqlite issue), reinstall CLIs
export NVM_DIR="$HOME/.nvm"
echo "## install nvm"
curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh 2>/dev/null | bash 2>&1 | tail -2 \
  || wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash 2>&1 | tail -2
. "$NVM_DIR/nvm.sh"
echo "## install node 22"
nvm install 22 2>&1 | tail -3
nvm use 22 >/dev/null 2>&1
echo "## versions (which node)"; which node; node -v 2>&1; npm -v 2>&1
echo "## install Claude Code + Codex"
npm i -g @anthropic-ai/claude-code @openai/codex 2>&1 | tail -12
echo "## verify"
(command -v claude && claude --version) 2>&1 || echo "claude missing"
(command -v codex && codex --version) 2>&1 || echo "codex missing"
