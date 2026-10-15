#!/usr/bin/env bash
# Runs ON THE VPS (Ubuntu 22.04+/Debian 12+), invoked by `deploy/vps.sh deploy`.
# Installs Python deps into ./venv and the recorder as a systemd service that
# starts on boot, restarts on crash and reconnects after network drops.
#
# Env (passed by vps.sh): COINS, AGG, SKIP_SYSTEMD=1 (for container tests)

set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COINS="${COINS:-BTC ETH SOL HYPE DOGE PUMP FARTCOIN}"
AGG="${AGG:-4 3}"
SERVICE="obi-recorder"
RUN_USER="$(id -un)"
SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

echo "==> system packages"
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq python3 python3-venv python3-pip rsync >/dev/null

PYV=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' || {
  echo "Python $PYV found; 3.11+ is required (use Ubuntu 24.04 or Debian 13)"; exit 1; }

echo "==> python venv (Python $PYV)"
cd "$APP_DIR"
[ -d venv ] || python3 -m venv venv
venv/bin/pip install -q --upgrade pip
venv/bin/pip install -q -r requirements.txt
mkdir -p data/live

echo "==> self-test"
venv/bin/python -m pytest -q 2>&1 | tail -1

if [ "${SKIP_SYSTEMD:-0}" = "1" ]; then
  echo "SKIP_SYSTEMD=1: not installing the service"; exit 0
fi

echo "==> systemd service: $SERVICE"
AGG_ARGS=""
[ -n "$AGG" ] && AGG_ARGS="--agg-sig-figs $AGG"
$SUDO tee /etc/systemd/system/$SERVICE.service >/dev/null <<EOF
[Unit]
Description=Hyperliquid order book recorder (orderbook-imbalance)
Wants=network-online.target
After=network-online.target

[Service]
User=$RUN_USER
WorkingDirectory=$APP_DIR
Environment=PYTHONUNBUFFERED=1
ExecStart=$APP_DIR/venv/bin/python $APP_DIR/scripts/record_live.py --out $APP_DIR/data/live --coins $COINS $AGG_ARGS
Restart=always
RestartSec=10
StandardOutput=append:$APP_DIR/data/live/recorder.log
StandardError=append:$APP_DIR/data/live/recorder.log

[Install]
WantedBy=multi-user.target
EOF
# keep the clock accurate (timestamps come from the exchange, but logs use local time)
$SUDO timedatectl set-ntp true 2>/dev/null || true
$SUDO systemctl daemon-reload
$SUDO systemctl enable --now $SERVICE
sleep 5
systemctl --no-pager --lines=0 status $SERVICE | head -5
echo "==> done. Recording to $APP_DIR/data/live"
