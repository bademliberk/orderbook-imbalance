#!/usr/bin/env bash
# Run the Hyperliquid recorder as a macOS launchd service.
#
# The service starts at login, restarts automatically if it crashes, and
# reconnects by itself after Wi-Fi drops or sleep/wake. Each run writes its own
# files, so stopping and starting again simply continues collecting.
#
#   scripts/recorder_service.sh install    # create the service and start it
#   scripts/recorder_service.sh stop       # stop (starts again at next login)
#   scripts/recorder_service.sh start      # start again after stop
#   scripts/recorder_service.sh restart
#   scripts/recorder_service.sh status     # running? how much data? last log lines
#   scripts/recorder_service.sh logs       # follow the log
#   scripts/recorder_service.sh uninstall  # stop and remove the service (data is kept)
#
# Coins / aggregation can be changed with env vars before `install`, e.g.
#   COINS="BTC ETH SOL" AGG="4 3" scripts/recorder_service.sh install

set -euo pipefail

LABEL="${LABEL:-com.obi.hyperliquid-recorder}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/venv/bin/python"
OUT="${OUT:-$ROOT/data/live}"
COINS="${COINS:-BTC ETH SOL HYPE DOGE PUMP FARTCOIN}"
AGG="${AGG:-4 3}"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"
LOG="$OUT/recorder.log"

write_plist() {
  mkdir -p "$OUT" "$(dirname "$PLIST")"
  local args=""
  for c in $COINS; do args+="    <string>$c</string>"$'\n'; done
  local agg=""
  for n in $AGG; do agg+="    <string>$n</string>"$'\n'; done
  cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$PY</string>
    <string>-u</string>
    <string>$ROOT/scripts/record_live.py</string>
    <string>--out</string>
    <string>$OUT</string>
    <string>--coins</string>
$args    <string>--agg-sig-figs</string>
$agg  </array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>StandardOutPath</key><string>$LOG</string>
  <key>StandardErrorPath</key><string>$LOG</string>
</dict>
</plist>
EOF
  plutil -lint "$PLIST" >/dev/null
}

stop_manual_recorders() {
  [ "${STOP_MANUAL:-1}" = "1" ] || return 0
  # A recorder started by hand would record the same data twice.
  local pids
  pids=$(pgrep -f "scripts/record_live.py" || true)
  for pid in $pids; do
    if ! launchctl print "$DOMAIN/$LABEL" 2>/dev/null | grep -q "pid = $pid"; then
      echo "stopping manually started recorder (pid $pid)"
      kill "$pid" 2>/dev/null || true
    fi
  done
}

is_loaded() { launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; }

case "${1:-status}" in
  install)
    stop_manual_recorders
    is_loaded && launchctl bootout "$DOMAIN/$LABEL" || true
    write_plist
    launchctl bootstrap "$DOMAIN" "$PLIST"
    echo "installed and started: $LABEL"
    sleep 3; "$0" status ;;
  start)
    [ -f "$PLIST" ] || { echo "not installed; run: $0 install"; exit 1; }
    is_loaded && { echo "already running"; exit 0; }
    stop_manual_recorders
    launchctl bootstrap "$DOMAIN" "$PLIST"; echo "started" ;;
  stop)
    is_loaded && launchctl bootout "$DOMAIN/$LABEL" && echo "stopped (will start again at next login; 'uninstall' to remove)" || echo "not running" ;;
  restart)
    "$0" stop || true; sleep 1; "$0" start ;;
  status)
    if is_loaded; then
      pid=$(launchctl print "$DOMAIN/$LABEL" | awk '/^\tpid = /{print $3}')
      echo "service: loaded, pid ${pid:-none (restarting?)}"
    else
      echo "service: not running"
      manual=$(pgrep -f "scripts/record_live.py" || true)
      [ -n "$manual" ] && echo "manually started recorder running: pid $manual"
    fi
    if [ -d "$OUT" ]; then
      echo "data: $(du -sh "$OUT" | cut -f1) in $OUT"
      newest=$(find "$OUT" -name '*.jsonl.gz' -type f -print0 2>/dev/null | xargs -0 ls -t 2>/dev/null | head -1)
      [ -n "$newest" ] && echo "newest file: ${newest#$OUT/} (modified $(stat -f '%Sm' "$newest"))"
    fi
    [ -f "$LOG" ] && { echo "--- last log lines"; tail -5 "$LOG"; } ;;
  logs)
    tail -f "$LOG" ;;
  uninstall)
    is_loaded && launchctl bootout "$DOMAIN/$LABEL" || true
    rm -f "$PLIST"; echo "uninstalled (data kept in $OUT)" ;;
  *)
    sed -n '2,20p' "$0"; exit 1 ;;
esac
