# seq=12: inspect install_aris.sh options and dry-run into a dedicated ~/research dir
export NVM_DIR="$HOME/.nvm"; . "$NVM_DIR/nvm.sh"; nvm use 22 >/dev/null 2>&1
echo "## help"; bash ~/aris_repo/tools/install_aris.sh --help 2>&1 | head -50
mkdir -p ~/research
echo "## DRY-RUN in ~/research"
cd ~/research && bash ~/aris_repo/tools/install_aris.sh --aris-repo ~/aris_repo --dry-run 2>&1 | tail -40
