#!/usr/bin/env bash
# Pass340 tomorrow helper — EDIT SLOT after reading LB; does not auto-submit.
set -euo pipefail
cd "$(dirname "$0")/../.."
KAGGLE="${KAGGLE:-.venv_forensics/bin/kaggle}"
COMP=playground-series-s6e11
echo "=== Live LB (top) ==="
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy \
  "$KAGGLE" competitions leaderboard -c "$COMP" --show | head -15
echo
echo "=== Your recent subs ==="
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy \
  "$KAGGLE" competitions submissions -c "$COMP" | head -12
echo
echo "Copy-paste when ready (Slot1 default — SAFE ×0.35):"
echo "$KAGGLE competitions submit -c $COMP -f submissions/sub_pass340_S_fam21_x035.csv -m 'Pass340 S1: St9+0.35*FAM21(+21.37e-5 OOF)'"
echo "Plan: logs/overnight/PASS340_PLAN.md"
