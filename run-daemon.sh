#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "$HOME/.cargo/env"
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  . "$SCRIPT_DIR/.env"
  set +a
fi
cd "$SCRIPT_DIR"
# The CLI treats --config as a global option, so any wrapper arguments need to
# come before the `daemon` subcommand.
exec cargo run --release -- "$@" daemon
