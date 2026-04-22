#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
SERVICE_PATH="$SERVICE_DIR/voice-dictation.service"
CONFIG_PATH="${1:-$SCRIPT_DIR/config.local.toml}"

if [ ! -f "$CONFIG_PATH" ]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  echo "Pass an explicit config path or create $SCRIPT_DIR/config.local.toml first." >&2
  exit 1
fi

mkdir -p "$SERVICE_DIR"

cat > "$SERVICE_PATH" <<EOF
[Unit]
Description=Voice Dictation
PartOf=graphical-session.target
After=graphical-session.target

[Service]
Type=simple
WorkingDirectory=$SCRIPT_DIR
ExecStart=$SCRIPT_DIR/run-daemon.sh --config $CONFIG_PATH
Restart=on-failure
RestartSec=2

[Install]
WantedBy=graphical-session.target
EOF

systemctl --user daemon-reload
systemctl --user disable voice-dictation.service >/dev/null 2>&1 || true
systemctl --user enable voice-dictation.service >/dev/null
systemctl --user restart voice-dictation.service

echo "Installed $SERVICE_PATH"
systemctl --user status voice-dictation.service --no-pager
