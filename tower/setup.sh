#!/bin/sh
# Runs on the tower (tower.ps1 calls it): install packages, then (re)start the service.
set -e
cd "$HOME/paper-editor"

# The tower's Python has no pip or venv and there's no sudo, so use uv in ~/.local/bin.
if [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
UV="$HOME/.local/bin/uv"
[ -d .venv ] || "$UV" venv -q --python 3.12 .venv
"$UV" pip install -q --python .venv/bin/python -r requirements-server.txt

mkdir -p "$HOME/.config/systemd/user"
cp tower/paper-editor.service "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
systemctl --user enable -q paper-editor
systemctl --user restart paper-editor
