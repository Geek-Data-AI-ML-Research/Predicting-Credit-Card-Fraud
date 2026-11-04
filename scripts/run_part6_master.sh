#!/usr/bin/env bash
# Master upgrade: cache gboost mats → GPU families → Part6 repro+expand stack.
# NO kaggle submit.
set -euo pipefail
export S6E11_ROOT="${S6E11_ROOT:-/workspace/s6e11}"
cd "$S6E11_ROOT"
mkdir -p logs/overnight cache/gboost_mats preds/part6_expand submissions
# Ensure gboost.find() can see train.csv + gpt2 tokenizer
export INPUT_DIR="$S6E11_ROOT"
export PYTHONUNBUFFERED=1
PY="${PYTHON:-/workspace/s6e11/.venv/bin/python}"
[[ -x "$PY" ]] || PY=python3

# Symlink tokenizer into searchable tree if needed
if [[ -f external_bank/goodpjw_artifacts/gpt2-tokenizer.json && ! -f gpt2-tokenizer.json ]]; then
  ln -sfn external_bank/goodpjw_artifacts/gpt2-tokenizer.json gpt2-tokenizer.json || cp external_bank/goodpjw_artifacts/gpt2-tokenizer.json gpt2-tokenizer.json
fi

echo "[master] START $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee logs/overnight/part6_master.log

echo "[master] GPU families (cache + cat/xgb/mlp)" | tee -a logs/overnight/part6_master.log
set +e
$PY -u scripts/gboost_gpu_families.py 2>&1 | tee logs/overnight/gboost_gpu_families.log
G_RC=${PIPESTATUS[0]}
set -e
echo "[master] gpu_families rc=$G_RC" | tee -a logs/overnight/part6_master.log

echo "[master] Part6 repro + expand stack" | tee -a logs/overnight/part6_master.log
set +e
$PY -u scripts/part6_repro_and_expand.py 2>&1 | tee logs/overnight/part6_expand.log
S_RC=${PIPESTATUS[0]}
set -e
echo "[master] stack rc=$S_RC" | tee -a logs/overnight/part6_master.log
echo "[master] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) — NO SUBMIT" | tee -a logs/overnight/part6_master.log
exit 0
