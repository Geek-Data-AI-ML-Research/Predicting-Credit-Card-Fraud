# CPU Campaign Report — Pass 0.95340+

**Elapsed:** 137.1s
**FAM21:** +21.37e-5
**CPU cumulative OOF:** +21.98e-5 (5/5) folds=[0.74, 4.31, 35.34, 32.68, 35.77]

## Phases
- Leaders: {'n': 4, 'names': ['aman', 'tuboa', 'mzishran', 'alex']}
- Alex restack: {
  "hits": [
    {
      "file": "oof_b890168ab4ab0fdc.npy",
      "solo_auc": 0.948447,
      "delta": 0.61,
      "w": -0.015,
      "pos": 4
    }
  ],
  "accepted": [
    {
      "file": "oof_b890168ab4ab0fdc.npy",
      "w": -0.015,
      "add": 0.61
    }
  ],
  "delta_vs_base": 21.98,
  "delta_vs_fam": 0.61
}
- 20k prior: {'status': 'skip'}
- Transductive: {'solo_auc': 0.952043, 'delta': 0.0, 'w': 0.0, 'pos': 0}
- Soft real test: {'status': 'skipped_low_ram', 'avail_gb': 2.17, 'need_gb': 4.0, 'note': 'proxy test already on disk; real regen needs >=4GB free (prefer 6GB+)'}

## Submit candidates (new)
- `sub_pass340_L1_st9_aman12_tuboa08.csv` 1-ρ=1.058e-4 — multi-leader ranklogit
- `sub_pass340_L2_st9_aman18_mz12.csv` 1-ρ=2.446e-4 — multi-leader stronger
- `sub_pass340_R1_alexrestack_x035.csv` 1-ρ=2.358e-4 — St9+0.35*alexrestack(+22.0e-5)
- `sub_pass340_R2_alexrestack_x055.csv` 1-ρ=5.814e-4 — St9+0.55*alexrestack(+22.0e-5)
- `sub_pass340_R3_alexrestack_x070.csv` 1-ρ=9.406e-4 — St9+0.7*alexrestack(+22.0e-5)
- `sub_pass340_CPU_x055.csv` 1-ρ=5.814e-4 — CPU campaign cumulative x0.55 (+22.0e-5)
- `sub_pass340_CPU_x070.csv` 1-ρ=9.406e-4 — CPU campaign cumulative x0.70 (+22.0e-5)

## GPU phase — READY TO START when Mac is idle / on GPU box
- A1 Bulsah-style encoding rebuild (TE+GPT2 tokens+digit keys, multi-seed)
- Scaled GPT-2 target-first (more orders/epochs)
- Real soft_w015/025 test regen if still missing (gboost KMAT)
- Optional: CatBoost/XGB large seed bags on new encodings

Monitor: `/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/logs/overnight/cpu_campaign_monitor.jsonl`
JSON: `/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/logs/overnight/cpu_campaign_pass340_report.json`