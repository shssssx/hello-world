#!/usr/bin/env bash
# One-time-start runner: polls the git branch for a command, executes it,
# pushes back combined output + any generated artifacts. Lets the remote
# (Claude) drive the experiment over git as the only available channel.
#
# Usage (run once on the GPU server, inside the repo):
#   nohup bash runner.sh > runner.log 2>&1 &
#   tail -f runner.log
#
# Protocol (all under ctrl/ on branch BRANCH):
#   ctrl/seq      monotonically increasing integer; a new value means "run me"
#   ctrl/cmd.sh   the command to execute (bash)
#   ctrl/out.txt  <- runner writes combined stdout/stderr here
#   ctrl/done     <- runner writes the seq it just finished
set -u
BRANCH="${BRANCH:-claude/zen-allen-7Y8Bx}"
POLL="${POLL:-5}"
export GIT_TERMINAL_PROMPT=0   # never hang on a credential prompt; fail fast
cd "$(dirname "$0")" || exit 1
git config pull.rebase true >/dev/null 2>&1

LAST=-1
echo "[runner] started; branch=$BRANCH poll=${POLL}s cwd=$(pwd)"
while true; do
  git fetch -q origin "$BRANCH" 2>/dev/null
  SEQ=$(git show "origin/$BRANCH:ctrl/seq" 2>/dev/null | tr -d '[:space:]')
  if [[ -n "$SEQ" && "$SEQ" =~ ^[0-9]+$ && "$SEQ" -gt "$LAST" ]]; then
    echo "[runner] new command seq=$SEQ -> executing"
    git reset --hard -q "origin/$BRANCH"
    {
      echo "### seq=$SEQ  $(date -u +%FT%TZ)"
      echo "### host=$(hostname)  python=$(python --version 2>&1)"
      echo "----- BEGIN -----"
    } > ctrl/out.txt
    bash ctrl/cmd.sh >> ctrl/out.txt 2>&1
    RC=$?
    echo "----- END (rc=$RC) -----" >> ctrl/out.txt
    echo "$SEQ" > ctrl/done
    git add -A
    git commit -q -m "runner: result seq=$SEQ (rc=$RC)" || true
    pushed=0
    for i in 1 2 3 4 5; do
      git pull -q --rebase origin "$BRANCH" 2>/dev/null || true
      if git push -q origin "$BRANCH" 2>/dev/null; then pushed=1; break; fi
      echo "[runner] push retry $i"; sleep 3
    done
    [[ "$pushed" == 1 ]] && echo "[runner] pushed result seq=$SEQ rc=$RC" \
                         || echo "[runner] WARNING push failed seq=$SEQ"
    LAST="$SEQ"
  fi
  sleep "$POLL"
done
