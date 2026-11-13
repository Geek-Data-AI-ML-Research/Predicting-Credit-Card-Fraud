# GM V14.2 — Fold1 Verification + F0+F1 Stack

**NO_SUBMIT=True** · Audit frozen at F0+F1.  
**Progress stamp (~17:58 UTC):** F1–F2 local; F2 AUC=`0.954835`; F3 OOF=`0.949330` (test-score in flight); F4 pending. Living status: `GM_V14_COMPLETE_5FOLD_L7_S2026_REPORT.md`.

## Ops
- Workdir: `/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/tmp/v14_2_audit` (repo volume — not system `/tmp`)
- Watchdog: `sleep 21600; PID 257 untouched`
- Sync: `repo /Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/tmp/v14_sync_monitor.sh — 60s size-stable pull`

## Fold1 Solo / Two-Seed (139,992 rows, folds==1)

- L7_s2026_f1 Bayes=`0.949578` · RankLogitMean=`0.949662`
- L7F_s7_f1=`0.949767` (ref 0.94977) · LM_f1=`0.950682` (ref 0.95068)
- **2-seed `0.5·L(L7F)+0.5·L(s2026)` = `0.950214` · Δ vs L7F = `+4.47` bps**
- F0 same 2-seed Δ = `+5.04` bps ≡ `+50.4e-5` AUC (prior “+50.4 bps” was e-5 units; F1 `+4.47` bps ≡ `+44.7e-5` — same order as F0)
- corr(L(s2026), L(L7F)) OOF F1 = `0.985692`
- corr(L(test_1), L(L7F_test)) = `0.987716`
- 1−spearmanr(L(test_0), L(test_1)) = `7.351324e-03`

## F0+F1 ATH Stack (279,984 OOF rows · 2 main / 10 sf)

- ATH=`stageB4fam+0.03*d1_v4` · OOF AUC=`0.9523468` (V13.2 target 0.9524301)
- d1=`preds/gm_v4_polish/d1_v4_oof.npy`

| Move | pos/2 | meanΔ e-5 | [F0,F1] | sf10 | t_stat_10sf | W-p |
|---|---|---|---|---|---|---|
| Move1_L7swap | 2/2 @w=0.015 | 0.008 | [0.008, 0.007] | 6/10 | 0.87 | 0.285 |
| Move2_LMswap | 2/2 @w=0.02 | 0.008 | [0.002, 0.014] | 4/10 | -0.10 | 0.59 |
| Move3_lofo_orth | 1/2 @w=0.012 | -0.000 | [0.003, -0.004] | 3/10 | -0.48 | 0.742 |

```json
{
  "NO_SUBMIT": true,
  "workdir": "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/tmp/v14_2_audit",
  "fold1": {
    "n_rows": 139992,
    "auc_bayes": 0.9495782516566903,
    "auc_ranklogit": 0.9496618938822113,
    "auc_L7F": 0.9497671496639838,
    "auc_LM": 0.9506818231843808,
    "ref_L7F_expected": 0.94977,
    "ref_LM_expected": 0.95068,
    "auc_2seed": 0.9502137247001919,
    "delta_bps_vs_L7F": 4.465750362081522,
    "F0_2seed_delta_bps": 5.038176360324664,
    "F0_ref_bps": 50.4,
    "corr_L_oof_s2026_vs_L7F": 0.9856919018734207,
    "corr_L_test1_vs_L7F_test": 0.9877159620490461,
    "spearman_dist_test0_vs_test1": 0.0073513239782039275
  },
  "ath_src": "stageB4fam+0.03*d1_v4",
  "ath_auc_oof": 0.9523467523006806,
  "ath_target_auc": 0.9524301242520837,
  "ath_candidates": {
    "stageB4fam+0.03*d1_v4": 0.9523467523006806,
    "L(l7f)+0.03*d1_v4": 0.949557764732478,
    "lm_raw_super_v4+0.03*d1_v4": 0.950947496546543
  },
  "d1_src": "preds/gm_v4_polish/d1_v4_oof.npy",
  "moves_F0F1": {
    "Move1_L7swap": {
      "main": {
        "w": 0.015,
        "pos": 2,
        "mean_e5": 0.007614294367241925,
        "folds": [
          0.008,
          0.007
        ]
      },
      "sf10": {
        "pos10": 6,
        "t": 0.8731683713477678,
        "mean_e5": 0.006215541957788062,
        "wilcox": 0.28515625
      }
    },
    "Move2_LMswap": {
      "main": {
        "w": 0.02,
        "pos": 2,
        "mean_e5": 0.0077682811405743735,
        "folds": [
          0.002,
          0.014
        ]
      },
      "sf10": {
        "pos10": 4,
        "t": -0.09729460106285183,
        "mean_e5": -0.0007926043099626412,
        "wilcox": 0.58984375
      }
    },
    "Move3_lofo_orth": {
      "main": {
        "w": 0.012,
        "pos": 1,
        "mean_e5": -0.00031006467948735406,
        "folds": [
          0.003,
          -0.004
        ]
      },
      "sf10": {
        "pos10": 3,
        "t": -0.4814818488201323,
        "mean_e5": -0.0015571811795567214,
        "wilcox": 0.7421875
      }
    }
  },
  "watchdog": "sleep 21600; PID 257 untouched",
  "sync": "repo /Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/tmp/v14_sync_monitor.sh \u2014 60s size-stable pull"
}
```
