#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_DIR="${TAMGEN_DOWNLOAD_DIR:-/tmp}"
CHECKPOINT_ZIP="$DOWNLOAD_DIR/tamgen_checkpoints.zip"
GPT_ZIP="$DOWNLOAD_DIR/tamgen_gpt_model.zip"
CHECKPOINT_URL='https://zenodo.org/api/records/13751391/files/checkpoints.zip/content'
GPT_URL='https://zenodo.org/api/records/13751391/files/gpt_model.zip/content'

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*'
export no_proxy='*'
mkdir -p "$DOWNLOAD_DIR" "$SCRIPT_DIR/checkpoints/crossdock_pdb_A10" "$SCRIPT_DIR/gpt_model"

download_file() {
  local url="$1"
  local output="$2"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c \
      --max-connection-per-server=16 \
      --split=16 \
      --min-split-size=1M \
      --file-allocation=none \
      --max-tries=0 \
      --retry-wait=2 \
      --allow-overwrite=true \
      --auto-file-renaming=false \
      --dir="$(dirname "$output")" \
      --out="$(basename "$output")" \
      "$url"
  else
    curl -L -C - --retry 20 --retry-all-errors --connect-timeout 30 \
      -o "$output" "$url"
  fi
}

download_file "$CHECKPOINT_URL" "$CHECKPOINT_ZIP"
download_file "$GPT_URL" "$GPT_ZIP"

echo '5815d681256eabaf62fb3df0ef3dfb0e  '"$CHECKPOINT_ZIP" | md5sum -c -
echo '17e7182c88be61d671c6b88423534586  '"$GPT_ZIP" | md5sum -c -

unzip -p "$CHECKPOINT_ZIP" checkpoints/crossdock_pdb_A10/checkpoint_best.pt \
  > "$SCRIPT_DIR/checkpoints/crossdock_pdb_A10/checkpoint_best.pt"
unzip -p "$GPT_ZIP" gpt_model/checkpoint_best.pt \
  > "$SCRIPT_DIR/gpt_model/checkpoint_best.pt"
unzip -p "$GPT_ZIP" gpt_model/dict.txt \
  > "$SCRIPT_DIR/gpt_model/dict.txt"

ls -lh \
  "$SCRIPT_DIR/checkpoints/crossdock_pdb_A10/checkpoint_best.pt" \
  "$SCRIPT_DIR/gpt_model/checkpoint_best.pt" \
  "$SCRIPT_DIR/gpt_model/dict.txt"
