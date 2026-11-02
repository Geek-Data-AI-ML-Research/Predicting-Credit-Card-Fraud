# S6E11 Validation Roadmap

Grounded in **Fraud Detection Handbook Chapter 5.2 (Validation Strategies)**, adapted to Kaggle Playground Series S6E11 (*Predicting Credit Card Fraud*).

## Core rule

> Your validation split must mirror how the test set was separated from the training set.

| Setting | Reality |
|---|---|
| Real-world fraud (Handbook 5.2) | Transactions arrive over time with label delay \(\delta_{\text{delay}}\) → **Prequential / rolling time-block** validation |
| Kaggle S6E11 | 699,960 train / 299,984 test rows, synthetically generated from [Udit Jain’s 20k NumPy dataset](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026). **Do not assume** the train/test split — **prove** it, then lock CV |

Official leaderboard metric: **ROC-AUC**. Diagnostics: PR-AUC (AP), Brier score, Train−Valid gap.

---

## Handbook → S6E11 mapping

| Handbook concept (Ch. 5.2) | Real-world fraud meaning | How we apply it in S6E11 |
|---|---|---|
| Train vs Test overfitting curve (Sec. 2.1) | Deep trees (`max_depth=50`) can hit ~1.000 Train AUC while Test collapses (~0.739) | **Stage 5.1**: CV harness logs Train AUC − Valid AUC across depths 2→12. On synthetic NumPy rules, shallow/medium trees often generalize better |
| Hold-out & Prequential split (\(\delta_{\text{train}}, \delta_{\text{delay}}, \delta_{\text{test}}\)) | Predict future blocks after label delay | **Stages 1.1–1.2**: test chronological signal (`id`, `day_of_week`, `time_of_day_hour`). If adversarial AUC ≈ 0.50, Stratified K-Fold is optimal |
| Confidence intervals across splits (Sec. 2.3–2.4) | Single hold-outs are high-variance; repeat splits for mean ± SD | **Stage 2.2**: new feature/HP must lift overall OOF ROC-AUC **and** ≥4/5 folds |
| Multi-metric evaluation (ROC-AUC, AP, CP@100) | ROC-AUC and AP react differently to FPs | **Stage 5.1**: log ROC-AUC (LB), Average Precision, Brier |

---

## 5-stage checklist

### Stage 1 — Split discovery (adversarial + chronology)

- [x] **1.1 Chronology audit** — `id` train[0,699959] / test[699960,999943] contiguous (Kaggle write artifact); fraud AUC(id)=0.499; `day_of_week`/`time_of_day_hour` L1(train,test)≈0.008–0.010; fraud rates flat across id quintiles (~1.6–1.7%). **No real time-order structure.**
- [x] **1.2 Adversarial validation** — OOF Adv AUC = **0.50080** (folds 0.50080±0.00092). Target ≈0.50 met.
- [x] **1.3 Decision lock** — **LOCKED:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Artifact: `logs/stage1_adversarial_result.json`.

### Stage 2 — Master CV harness + promotion rule

- [x] **2.1 Handbook CV logger** — Baseline shallow LGBM (`max_depth=4`, `num_leaves=15`) under locked SKF(5,rs=42). OOF ROC-AUC **0.946977** (fold mean 0.946982 ± 0.003582); mean Train−Valid gap **0.011465**; OOF PR-AUC 0.509643; OOF Brier 0.011022. Artifact: `logs/stage2_handbook_cv_result.json`.
- [x] **2.2 Promotion gate** — Rule locked: **OOF ROC-AUC↑ AND ≥4/5 folds↑**. Challenger deeper LGBM (`depth=8`, `leaves=63`) **REJECTED** (OOF Δ=−0.001109, fold wins 0/5; gap ballooned to 0.0268).

### Stage 3 — Baseline under locked CV

- [x] **3.1** Shallow LGBM (`num_leaves=4`) / XGB (`max_depth=2`) under locked SKF(5,rs=42); arms = raw / digit / TE. Best: **te_xgb OOF 0.949315**, **te_lgbm 0.949306**. Digit-only arms lag (digit_lgbm 0.944711, digit_xgb 0.941676, larger gaps). Script: `scripts/validation_stage3_baselines.py`.
- [x] **3.2** Control table + fold vectors + gaps in `logs/stage3_baselines_result.json`; OOF preds `preds/stage3/*_oof.npy`. Control champion for Stage 4 levers: **te_xgb** (fold vector ≈ [0.9484, 0.9500, 0.9538, 0.9504, 0.9444], gap 0.0054).

### Stage 4 — Feature / modeling levers (one variable at a time)

- [ ] **4.1** Digit keys / target encoding / raw / GLM / NN arms — each proposed as a single lever
- [ ] **4.2** Reject levers that fail Stage 2.2 even if public LB ticks up

### Stage 5 — Overfitting curve + multi-metric diagnostics

- [ ] **5.1 Depth sweep** — `max_depth` / `num_leaves` 2→12; plot Train−Valid gap vs Valid AUC
- [ ] **5.2 Calibration / PR** — Brier + PR-AUC as secondary; do not optimize LB solely on PR
- [ ] **5.3 Stack policy** — Ensemble only members with honest OOF under locked folds; no submit until human approval

---

## Execution order

1. Run Stage 1 adversarial + chronology (`scripts/validation_stage1_adversarial.py`)
2. Run Stage 2 harness on a shallow baseline (`scripts/validation_stage2_handbook_cv.py`)
3. Lock CV contract in writing (this file + Studio roadmap when MCP supports it)
4. Only then resume FE / stack / submit experiments under the Stage 2.2 gate

## Related local artifacts

- Competition data: `data/train.csv`, `data/test.csv`
- Original: `original/credit_card_fraud_2026.csv`
- Prior stacks (not validation-locked): `submissions/` — best public so far **0.95163** (human approval required before further submits)

## Status

| Item | State |
|---|---|
| Local roadmap doc | **Updated** (this file) |
| Studio Roadmap tab | **Synced** on `playground-cc` (`prj_01M6EB3QGMT4RCCZ3D23CJ34RP`) — 6 strategies (core rule + Stages 1–5) |
| MCP | `list_roadmap` / `add_roadmap_strategy` / `update_roadmap_strategy` (after mcp reload) |
