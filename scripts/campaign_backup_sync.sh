#!/usr/bin/env bash
# Sync campaign artifacts: pod -> local -> S3
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/logs/overnight/campaign_pod.env"

POD_ID="${POD_ID:?}"
if [[ "$POD_ID" == "${BLACKLIST_POD}" ]]; then
  echo "REFUSE blacklisted pod $POD_ID" >&2
  exit 2
fi

SSH_KEY="${SSH_KEY:-$HOME/.ssh/casmi_runpod}"
HOST="${SSH_DIRECT_HOST}"
PORT="${SSH_DIRECT_PORT}"
USER="${SSH_USER:-root}"
RSYNC_SSH="ssh -i $SSH_KEY -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -p $PORT"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOCAL="${LOCAL_BACKUP_DIR}/${STAMP}"
LATEST="${LOCAL_BACKUP_DIR}/latest"
mkdir -p "$LOCAL" "$LATEST" "$ROOT/preds/matrix_v2" "$ROOT/logs/overnight"

echo "[backup] $STAMP pull from pod $POD_ID ($HOST:$PORT)"
rsync -az --timeout=180 -e "$RSYNC_SSH" \
  "$USER@$HOST:/workspace/s6e11/preds/matrix_v2/" "$ROOT/preds/matrix_v2/" 2>/dev/null || true
rsync -az --timeout=180 -e "$RSYNC_SSH" \
  "$USER@$HOST:/workspace/s6e11/logs/overnight/" "$ROOT/logs/overnight/" 2>/dev/null || true

rsync -a "$ROOT/preds/matrix_v2/" "$LOCAL/preds_matrix_v2/" 2>/dev/null || true
rsync -a "$ROOT/logs/overnight/" "$LOCAL/logs_overnight/" 2>/dev/null || true
rsync -a --delete "$LOCAL/" "$LATEST/" 2>/dev/null || true

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true
export AWS_EC2_METADATA_DISABLED=true
if command -v aws >/dev/null 2>&1; then
  aws s3 sync "$LOCAL" "${S3_BACKUP_URI}/${STAMP}/" --only-show-errors || true
  aws s3 sync "$LATEST" "${S3_BACKUP_URI}/latest/" --only-show-errors || true
fi

# Keep last 24 local stamp dirs (macOS bash 3 compatible)
ls -1dt "$LOCAL_BACKUP_DIR"/20* 2>/dev/null | tail -n +25 | while read -r d; do rm -rf "$d"; done || true
echo "[backup] done $STAMP -> local+s3"
