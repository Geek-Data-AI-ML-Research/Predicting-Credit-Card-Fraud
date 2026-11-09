#!/bin/bash
# One-shot status probe for s6e11 overnight GPU job
set -e
SSH=(ssh -i /Users/mz/.ssh/casmi_runpod -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -p 11107 root@69.8.146.173)
ROOT=/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud
OUT="$ROOT/logs/overnight/monitor_last_tick.txt"
"${SSH[@]}" 'bash -s' <<'REMOTE' | tee "$OUT"
date -u +"UTC=%Y-%m-%dT%H:%M:%SZ"
echo "PID5157=$(ps -p 5157 -o etime= 2>/dev/null | tr -d " " || echo DEAD)"
echo "CAMPAIGN=$(pgrep -f "python3 -u scripts/post_gpt2_3ep_campaign.py" >/dev/null && echo RUNNING || (pgrep -f arm_post_campaign.sh >/dev/null && echo WAITING || echo NONE))"
tail -8 /workspace/src/lm/out/lm/nohup.log 2>/dev/null || true
echo "---STATUS---"
cat /workspace/src/lm/out/lm/status.json 2>/dev/null || true
echo "---CAMPAIGN_STATE---"
cat /workspace/s6e11/logs/overnight/post_gpt2_3ep_state.json 2>/dev/null || echo none
echo "---CAMPAIGN_LOG_TAIL---"
tail -15 /workspace/s6e11/logs/overnight/post_gpt2_3ep_campaign.nohup.log 2>/dev/null || echo none
echo "---DONE---"
if [ -f /workspace/s6e11/logs/overnight/CAMPAIGN_DONE.json ]; then
  echo CAMPAIGN_DONE_PRESENT=1
  python3 -c "import json;d=json.load(open('/workspace/s6e11/logs/overnight/CAMPAIGN_DONE.json'));print('status=',d.get('status'),'ready=',d.get('ready_to_terminate'))"
else
  echo CAMPAIGN_DONE_PRESENT=0
fi
REMOTE
