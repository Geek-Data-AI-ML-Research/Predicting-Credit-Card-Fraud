# GM V14.1 Watchdog / Canary / Step5 Prep

**Updated:** 2026-11-13 ~17:58 UTC · **NO_SUBMIT=True**

Ops paths (post–`/tmp` full):

- Sync: `tmp/v14_sync_monitor.sh` · log `tmp/logs/v14_sync_monitor.log`
- Audit bank: `tmp/v14_2_audit/`
- Watchdog on pod: `sleep 21600` (360 m from ~15:31 UTC)

```json
{
  "NO_SUBMIT": true,
  "L7F_s7": {
    "src": "external/.../lm7_oof_5fold.npy (via tmp/v14_2_audit/s7)",
    "folds": [0.9475445293730193, 0.9497671496639838, 0.9551661607076907, 0.9490652062349009, 0.9456760135623762],
    "mean": 0.9494438119083941
  },
  "LM_megayak_5f": {
    "folds": [0.948505568399054, 0.9506818231843808, 0.9560849667798148, 0.9506042586063622, 0.946380901046167],
    "mean": 0.9504515036031558
  },
  "L7_s2026_partial_bayes": [
    0.9474933760825137,
    0.9495782516566903,
    0.9548350967066196,
    0.949330,
    null
  ],
  "local_files": {
    "oof_test_0": true,
    "oof_test_1": true,
    "oof_test_2": true,
    "oof_test_3": false,
    "oof_test_4": false
  },
  "canary": {
    "running_process_gate": "auc26 > 0.9485",
    "F1_auc": 0.9495782516566903,
    "F1_pass": true,
    "solo_only_canary_on_pod": true
  },
  "eta_obs": {
    "train_min_per_fold": 48.9,
    "score_min_per_fold": 26.8,
    "wall_min_per_fold": 75.7,
    "watchdog_min": 360.0,
    "note": "F3 OOF done 0.949330; test-score in flight; F4 remaining"
  },
  "v14_2": "logs/overnight/GM_V14_2_FOLD1_VERIFICATION.md"
}
```
