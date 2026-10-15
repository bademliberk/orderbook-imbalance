#!/usr/bin/env bash
# Control the recorder on a VPS from this laptop (needs SSH access to the VPS).
#
#   deploy/vps.sh deploy  user@host   # copy code, install deps + systemd service, start
#   deploy/vps.sh status  user@host   # service state, data size, disk, recent log
#   deploy/vps.sh logs    user@host   # follow the recorder log
#   deploy/vps.sh stop|start|restart user@host
#   deploy/vps.sh sync    user@host   # pull recordings to ./data/vps/ (incremental)
#   deploy/vps.sh backtest user@host  # sync, then run the 1%-depth backtest on them
#   deploy/vps.sh remove  user@host   # stop and remove the service (data kept on VPS)
#
# Re-running `deploy` after code changes updates the code and restarts the
# service; recordings on the VPS are never touched by deploy.
# Options via env: COINS="BTC ETH" AGG="4 3" REMOTE_DIR=orderbook-imbalance SSH_OPTS="-i key.pem"

set -euo pipefail

CMD="${1:-}"; HOST="${2:-}"
[ -n "$CMD" ] && [ -n "$HOST" ] || { sed -n '2,15p' "$0"; exit 1; }

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="${REMOTE_DIR:-orderbook-imbalance}"
COINS="${COINS:-BTC ETH SOL HYPE DOGE PUMP FARTCOIN}"
AGG="${AGG:-4 3}"
SSH_OPTS="${SSH_OPTS:-}"
SERVICE="obi-recorder"
ssh_() { ssh $SSH_OPTS "$HOST" "$@"; }
remote_sudo() { ssh_ "if [ \$(id -u) -eq 0 ]; then $*; else sudo $*; fi"; }

case "$CMD" in
  deploy)
    echo "==> copying code to $HOST:~/$REMOTE_DIR"
    ssh_ "mkdir -p ~/$REMOTE_DIR; command -v rsync >/dev/null || { S=; [ \$(id -u) -ne 0 ] && S=sudo; \$S apt-get update -qq && \$S apt-get install -y -qq rsync >/dev/null; }"
    rsync -az --delete -e "ssh $SSH_OPTS" \
      --exclude /venv/ --exclude /data/ --exclude /reports/ --exclude /models/ \
      --exclude /.git/ --exclude __pycache__/ --exclude .pytest_cache/ \
      "$ROOT/" "$HOST:~/$REMOTE_DIR/"
    ssh_ "cd ~/$REMOTE_DIR && COINS='$COINS' AGG='$AGG' SKIP_SYSTEMD='${SKIP_SYSTEMD:-0}' bash deploy/remote_setup.sh"
    ;;
  status)
    ssh_ "systemctl is-active $SERVICE >/dev/null && echo 'service: running' || echo 'service: NOT running';
          systemctl show $SERVICE -p ActiveEnterTimestamp -p NRestarts | sed 's/^/  /';
          echo \"data: \$(du -sh ~/$REMOTE_DIR/data/live 2>/dev/null | cut -f1)\";
          df -h ~ | awk 'NR==2{print \"disk: \"\$4\" free of \"\$2}';
          echo '--- reconnects in the last 24 h:' \$(find ~/$REMOTE_DIR/data/live -name recorder.log -mtime -1 -exec grep -c 'stream error' {} \; 2>/dev/null);
          echo '--- last log lines'; tail -5 ~/$REMOTE_DIR/data/live/recorder.log"
    ;;
  logs)
    ssh_ "tail -f ~/$REMOTE_DIR/data/live/recorder.log" ;;
  stop|start|restart)
    remote_sudo systemctl "$CMD" "$SERVICE"; echo "$CMD: done" ;;
  sync)
    mkdir -p "$ROOT/data/vps"
    rsync -az -e "ssh $SSH_OPTS" "$HOST:~/$REMOTE_DIR/data/live/" "$ROOT/data/vps/"
    echo "synced to $ROOT/data/vps ($(du -sh "$ROOT/data/vps" | cut -f1))" ;;
  backtest)
    "$0" sync "$HOST"
    cd "$ROOT" && venv/bin/python scripts/depth_backtest.py --data data/vps --band-bps 100 --out reports/depth_backtest_vps.md ;;
  remove)
    remote_sudo "systemctl disable --now $SERVICE; rm -f /etc/systemd/system/$SERVICE.service; systemctl daemon-reload"
    echo "service removed (data kept in ~/$REMOTE_DIR/data/live on the VPS)" ;;
  *)
    sed -n '2,15p' "$0"; exit 1 ;;
esac
