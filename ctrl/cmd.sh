# seq=8: probe toolchain + clone ARIS repo (read-only + clone, no installs yet)
echo "## node/npm"; node -v 2>&1; npm -v 2>&1
echo "## claude"; (command -v claude && claude --version) 2>&1 || echo "claude: not installed"
echo "## codex"; (command -v codex && codex --version) 2>&1 || echo "codex: not installed"
echo "## npm global prefix"; npm config get prefix 2>&1
echo "## clone ARIS"; cd ~ && rm -rf aris_repo && git clone --depth 1 https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep.git aris_repo 2>&1 | tail -3
echo "## skills count"; ls ~/aris_repo/skills 2>/dev/null | wc -l
echo "## install script present?"; ls -la ~/aris_repo/tools/install_aris.sh 2>&1
