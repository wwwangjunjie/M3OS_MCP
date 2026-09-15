#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/tamgen_mcp.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "TamGen MCP server is not running"
  exit 0
fi
PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  for _attempt in {1..60}; do
    if ! kill -0 "$PID" 2>/dev/null; then
      break
    fi
    sleep 0.5
  done
fi
mv "$PID_FILE" "$PID_FILE.stopped"
echo "TamGen MCP server stopped"

