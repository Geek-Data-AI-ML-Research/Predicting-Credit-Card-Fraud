#!/usr/bin/env bash
set -euo pipefail
POD=p4sli9zvpc097s
HOST=213.173.102.132
PORT=26429
SSH=(ssh -i "$HOME/.ssh/casmi_runpod" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p "$PORT" root@"$HOST")
RP=$(command -v runpodctl || echo /tmp/runpodctl)
LOG=/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/logs/stop_pod_when_done.log
ROOT=/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud
: > "$LOG"
echo "[stopwatch] START $(date -u) pod=$POD" | tee -a "$LOG"
while true; do
  if "${SSH[@]}" 'test -f /tmp/s6e11_work/logs/CAMPAIGN_10TESTS_COMPLETE' 2>/dev/null; then
    echo "[stopwatch] complete marker $(date -u)" | tee -a "$LOG"
    mkdir -p "$ROOT/logs/pod_final" "$ROOT/preds/unattended" "$ROOT/submissions"
    rsync -az -e "ssh -i $HOME/.ssh/casmi_runpod -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -p $PORT" \
      root@$HOST:/tmp/s6e11_work/logs/ "$ROOT/logs/pod_final/" || true
    rsync -az -e "ssh -i $HOME/.ssh/casmi_runpod -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -p $PORT" \
      root@$HOST:/tmp/s6e11_work/preds/unattended/ "$ROOT/preds/unattended/" || true
    rsync -az -e "ssh -i $HOME/.ssh/casmi_runpod -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -p $PORT" \
      root@$HOST:/tmp/s6e11_work/submissions/ "$ROOT/submissions/" || true
    echo "[stopwatch] stopping pod $POD" | tee -a "$LOG"
    "$RP" pod stop "$POD" -o json 2>&1 | tee -a "$LOG"
    echo "[stopwatch] DONE $(date -u)" | tee -a "$LOG"
    exit 0
  fi
  sleep 120
done
