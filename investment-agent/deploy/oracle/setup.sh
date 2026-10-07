#!/usr/bin/env bash
# One-time setup on an Oracle Cloud Ubuntu 22.04/24.04 VM (Always Free Ampere A1 works).
# Run as a sudo-capable user:  bash deploy/oracle/setup.sh
set -euo pipefail

APP_DIR=/opt/invest-agent
SRC_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip git

sudo useradd --system --create-home --home-dir /var/lib/invest-agent --shell /usr/sbin/nologin invest || true
sudo mkdir -p "$APP_DIR"
sudo rsync -a --exclude state --exclude reports --exclude .venv "$SRC_DIR/" "$APP_DIR/"
sudo python3 -m venv "$APP_DIR/.venv"
sudo "$APP_DIR/.venv/bin/pip" install --upgrade pip
sudo "$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"

[ -f "$APP_DIR/config.yaml" ] || sudo cp "$APP_DIR/config.example.yaml" "$APP_DIR/config.yaml"
[ -f "$APP_DIR/.env" ] || sudo cp "$APP_DIR/.env.example" "$APP_DIR/.env"
sudo mkdir -p "$APP_DIR/state"
sudo chown -R invest:invest "$APP_DIR"
sudo chmod 600 "$APP_DIR/.env"

sudo cp "$SRC_DIR"/deploy/oracle/invest-agent-*.service "$SRC_DIR"/deploy/oracle/invest-agent-run.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now invest-agent-serve.service
sudo systemctl enable --now invest-agent-run.timer

echo
echo "Done. Next:"
echo "  1. sudo -u invest nano $APP_DIR/.env          # credentials"
echo "  2. sudo -u invest nano $APP_DIR/config.yaml   # start with mode: dry_run"
echo "  3. sudo systemctl restart invest-agent-serve"
echo "  4. sudo systemctl start invest-agent-run      # one run now; watch: journalctl -u invest-agent-run -f"
