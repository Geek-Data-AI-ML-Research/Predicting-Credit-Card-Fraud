#!/usr/bin/env bash
# Continuous local backup of gpt2-medium pod outputs (read-only rsync).
# Does NOT touch PID 3021 / training — pulls files only.
# Small artifacts every INTERVAL_SEC; checkpoints every CKPT_EVERY cycles (with flock).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_LOCAL="$ROOT/preds/gpt2_med_4ep"
CKPT_LOCAL="$OUT_LOCAL/ckpt"
LOG="$ROOT/logs/overnight/watch_sync_gpt2_med.log"
LOCK="$ROOT/logs/overnight/watch_sync_gpt2_med.ckpt.lock"
SSH=(ssh -i "$HOME/.ssh/casmi_runpod" -p 22002 -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o ServerAliveInterval=30)
RSH="ssh -i $HOME/.ssh/casmi_runpod -p 22002 -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o ServerAliveInterval=30"
HOST="root@194.68.245.23"
REMOTE_OUT="/workspace/s6e11/lm/out/gpt2_med_4ep/"
REMOTE_CKPT="/workspace/s6e11/lm/ckpt/gpt2_med_4ep/"
INTERVAL_SEC="${INTERVAL_SEC:-90}"
CKPT_EVERY="${CKPT_EVERY:-3}"   # every 3rd loop ≈ 4.5 min

mkdir -p "$OUT_LOCAL" "$CKPT_LOCAL" "$(dirname "$LOG")"
echo "===== watch_sync start $(date -u +%Y-%m-%dT%H:%M:%SZ) interval=${INTERVAL_SEC}s ckpt_every=${CKPT_EVERY} =====" | tee -a "$LOG"

sync_small() {
  local ts rc=0
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if ! "${SSH[@]}" "$HOST" 'ps -p 3021 -o pid=,etime=,pcpu= --no-headers 2>/dev/null; ls /workspace/s6e11/lm/out/gpt2_med_4ep/ 2>/dev/null' >>"$LOG" 2>&1; then
    echo "[$ts] WARN: SSH/pod unreachable" | tee -a "$LOG"
    return 1
  fi
  rsync -a --partial --inplace --timeout=120 -e "$RSH" \
    "${HOST}:${REMOTE_OUT}" \
    --include='*/' \
    --include='oof_*.npy' \
    --include='test_*.npy' \
    --include='run.log' \
    --include='status.json' \
    --include='sha256*.txt' \
    --include='PHASE1*' \
    --exclude='*' \
    "$OUT_LOCAL/" >>"$LOG" 2>&1 || rc=$?
  local n_oof n_test
  n_oof=$(ls "$OUT_LOCAL"/oof_*.npy 2>/dev/null | wc -l | tr -d ' ')
  n_test=$(ls "$OUT_LOCAL"/test_*.npy 2>/dev/null | wc -l | tr -d ' ')
  echo "[$ts] small sync rc=$rc oof=$n_oof test=$n_test" | tee -a "$LOG"
  return $rc
}

sync_ckpt() {
  local ts rc=0
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  # Skip if another ckpt rsync already holds the lock / is running
  if ! (set -C; echo $$ >"$LOCK") 2>/dev/null; then
    local holder
    holder="$(cat "$LOCK" 2>/dev/null || echo '?')"
    if kill -0 "$holder" 2>/dev/null; then
      echo "[$ts] ckpt sync skipped (lock held by pid $holder)" | tee -a "$LOG"
      return 0
    fi
    echo $$ >"$LOCK"
  fi
  trap 'rm -f "$LOCK"' RETURN
  echo "[$ts] ckpt sync begin" | tee -a "$LOG"
  # Use temp-dir staging (no --inplace) so a mid-transfer never truncates a good local .pt
  rsync -a --partial --timeout=900 -e "$RSH" \
    "${HOST}:${REMOTE_CKPT}" \
    --include='*/' \
    --include='fold_*.pt' \
    --exclude='*' \
    "$CKPT_LOCAL/" >>"$LOG" 2>&1 || rc=$?
  for f in "$CKPT_LOCAL"/fold_*.pt; do
    [ -f "$f" ] || continue
    bn="$(basename "$f")"
    if [ ! -e "$OUT_LOCAL/$bn" ]; then
      ln "$f" "$OUT_LOCAL/$bn" 2>/dev/null || cp -n "$f" "$OUT_LOCAL/$bn" 2>/dev/null || true
    fi
  done
  local sizes
  sizes="$(ls -lah "$CKPT_LOCAL"/fold_*.pt 2>/dev/null | awk '{print $5,$9}' | xargs -n2 printf '%s=%s ')"
  echo "[$ts] ckpt sync rc=$rc $sizes" | tee -a "$LOG"
  rm -f "$LOCK"
  trap - RETURN
  return $rc
}

i=0
while true; do
  sync_small || true
  i=$((i + 1))
  if (( i % CKPT_EVERY == 0 )); then
    # Don't start if an urgent/external rsync for fold_*.pt is already running
    if pgrep -f 'rsync.*gpt2_med_4ep.*(fold_|ckpt)' >/dev/null 2>&1; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ckpt deferred — external rsync active" | tee -a "$LOG"
    else
      sync_ckpt || true
    fi
  fi
  sleep "$INTERVAL_SEC"
done
