# seq=17: discover codex MCP subcommand name (no login needed for --help)
export NVM_DIR="$HOME/.nvm"; . "$NVM_DIR/nvm.sh"; nvm use 22 >/dev/null 2>&1
echo "## codex top-level help (mcp-related)"; codex --help 2>&1 | grep -iE "mcp|server" 
echo "## codex mcp --help"; codex mcp --help 2>&1 | head -25
