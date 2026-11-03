#!/usr/bin/env bash
# After T01 fold0 AUC prints: cache matrices, trim seeds, restart suite on cache.
set -uo pipefail
ROOT=/tmp/s6e11_work
LOG=/tmp/s6e11_logs
PY=/tmp/venv/bin/python
export S6E11_ROOT=$ROOT S6E11_PERSIST=/workspace/s6e11 DIGLEAD_CACHE=/workspace/cache_diglead_folds
cd "$ROOT"
mkdir -p /workspace/cache_diglead_folds "$LOG"

echo "[hotpatch] START $(date -u)" | tee -a "$LOG/hotpatch.log"

# 1) capture fold0 line
F0=$(grep -E "T01 fold0 auc=" "$LOG/unattended_10tests.nohup.out" | tail -1 || true)
echo "[hotpatch] $F0" | tee -a "$LOG/hotpatch.log"
echo "$F0" > "$LOG/t01_fold0_auc.txt"
cp -a "$LOG/t01_fold0_auc.txt" /workspace/s6e11/logs/ 2>/dev/null || true

# 2) stop current run (fold0 already logged)
pkill -f run_unattended_10tests.py || true
pkill -f run_10_tests_unattended.sh || true
sleep 2

# 3) prebuild 5-fold cache (vectorized FE + save npy)
echo "[hotpatch] prebuild cache..." | tee -a "$LOG/hotpatch.log"
$PY -u scripts/prebuild_diglead_cache.py 2>&1 | tee -a "$LOG/prebuild_diglead_cache.log"
ls -la /workspace/cache_diglead_folds/ | tee -a "$LOG/hotpatch.log"

# 4) verify 2-seed bag config
$PY -c "import sys; sys.path.insert(0,'scripts'); from unattended_common import BAG_SEEDS; print('BAG_SEEDS', BAG_SEEDS); assert BAG_SEEDS==[42,2026]"

# 5) restart full suite (cache hits → fast)
rm -f logs/CAMPAIGN_10TESTS_COMPLETE logs/REQUEST_POD_STOP
nohup bash scripts/run_10_tests_unattended.sh >/tmp/s6e11_logs/unattended_10tests.nohup.out 2>&1 &
echo "[hotpatch] restarted pid=$! $(date -u)" | tee -a "$LOG/hotpatch.log"
sleep 5
tail -20 "$LOG/unattended_10tests.nohup.out" | tee -a "$LOG/hotpatch.log"
