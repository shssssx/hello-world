# seq=13: real ARIS skill install into ~/research (quiet, no prompts)
export NVM_DIR="$HOME/.nvm"; . "$NVM_DIR/nvm.sh"; nvm use 22 >/dev/null 2>&1
cd ~/research && bash ~/aris_repo/tools/install_aris.sh --aris-repo ~/aris_repo --quiet 2>&1 | tail -30
echo "## linked skills count"; ls ~/research/.claude/skills 2>/dev/null | wc -l
echo "## sample"; ls ~/research/.claude/skills 2>/dev/null | head -20
echo "## CLAUDE.md head"; head -15 ~/research/CLAUDE.md 2>/dev/null
