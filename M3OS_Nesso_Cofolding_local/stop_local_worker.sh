#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNS_ROOT="${NESSO_LOCAL_RUNS_ROOT:-$PROJECT_DIR/nesso_runs}"
GPU_IDS="${NESSO_LOCAL_GPU_IDS:-${NESSO_LOCAL_GPU_ID:-}}"
if [[ -z "$GPU_IDS" && -d "$RUNS_ROOT/.nesso_local" ]]; then
  for QUEUE_DIR in "$RUNS_ROOT"/.nesso_local/gpu_*; do
    [[ -d "$QUEUE_DIR" ]] || continue
    GPU_ID="${QUEUE_DIR##*/gpu_}"
    GPU_IDS="${GPU_IDS:+$GPU_IDS,}$GPU_ID"
  done
fi
if [[ -z "$GPU_IDS" ]]; then
  echo "No resident Nesso GPU workers were found." >&2
  exit 1
fi

IFS=',' read -r -a GPU_ID_LIST <<< "$GPU_IDS"
for GPU_ID in "${GPU_ID_LIST[@]}"; do
  if ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "Invalid physical GPU ID: $GPU_ID" >&2
    exit 1
  fi
  "$PROJECT_DIR/.venv/bin/python" \
    "$PROJECT_DIR/nesso_cofolding_skill/local_client.py" stop \
    --runs-root "$RUNS_ROOT" \
    --gpu-id "$GPU_ID"
done
