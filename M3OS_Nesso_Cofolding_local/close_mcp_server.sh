#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$PROJECT_DIR/nesso_local_mcp.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "Local Nesso MCP server is not recorded as running."
  exit 0
fi
PID="$(<"$PID_FILE")"
if [[ "$PID" =~ ^[0-9]+$ ]] && kill -0 "$PID" 2>/dev/null; then
  CMDLINE="$(tr '\0' ' ' < "/proc/$PID/cmdline" 2>/dev/null || true)"
  if [[ "$CMDLINE" != *"mcp_server_nesso_cofolding_local.py"* ]]; then
    echo "Refusing to signal unrelated process pid=$PID: $CMDLINE" >&2
    exit 1
  fi
  kill "$PID"
  for _attempt in {1..60}; do
    if ! kill -0 "$PID" 2>/dev/null; then
      break
    fi
    sleep 0.5
  done
  if kill -0 "$PID" 2>/dev/null; then
    echo "MCP server did not exit within 30 seconds: pid=$PID" >&2
    exit 1
  fi
  echo "Stopped local Nesso MCP server pid=$PID"
else
  echo "Recorded MCP process is not running: pid=$PID"
fi
rm -f -- "$PID_FILE"
echo "The resident Nesso worker was retained; use ./stop_local_worker.sh to stop it."
