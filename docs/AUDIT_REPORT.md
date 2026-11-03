# S6E11 Campaign Audit

**Competition:** playground-series-s6e11 — Predicting Credit Card Fraud (ROC-AUC)  
**Snapshot:** 2026-11-02 ~17:34 UTC · **393 teams**  
**Sources:** Kaggle public LB + local validation logs (`logs/audit_lb_*.json`, `logs/stage*`)

Live canvas: `S6E11-Campaign-Audit.canvas.tsx`

---

## Current position

| | |
|---|---|
| **Team** | Moran Zavdi |
| **Rank** | **#23 / 393** |
| **Best public** | **0.95163** |
| **Leader** | 0.95192 (Adama Traore) |
| **Gap to #1** | −0.00029 |

Tied band around 0.95163 with several teams (~ranks 23–26). Private LB unknown until end.

### Our submissions (scored)

| File | Public | CV | Notes |
|------|--------|-----|-------|
| `hillclimb_mix_0.95111.csv` | **0.95163** | 0.95111 | Best — rank hillclimb + logit mix over public+self OOFs |
| `mega_logit_0.95115.csv` | 0.95158 | 0.95115 | Logit LR over pooled public OOF banks |
| `artifact_stack.csv` | 0.95143 | — | First logit stack of public RealMLP/EBM/LGB/GLM OOFs |

---

## Current leaders (public top 20)

| Rank | Team | Public ROC-AUC |
|------|------|----------------|
| 1 | Adama Traore | 0.95192 |
| 2 | Hoai Vo | 0.95188 |
| 3 | Juste_ME | 0.95187 |
| 4 | Thiago Mantuani | 0.95185 |
| 5 | Aldani Prasetyo | 0.95184 |
| 6 | Ayush Maurya | 0.95183 |
| 7 | Jatin Gera | 0.95182 |
| 8 | Nilanjan_Saha123 | 0.9518 |
| 9 | Masha Kholodenko | 0.9518 |
| 10 | Mansi Aggarwal | 0.9518 |
| 11 | Areeb Ali | 0.9518 |
| 12 | KNAI \| Michael Jan | 0.9518 |
| 13 | rafli | 0.9518 |
| 14 | OigaZhDn | 0.95179 |
| 15 | kazuhiro3381 | 0.95179 |
| 16 | starkhushi | 0.95177 |
| 17 | MaglitNostra | 0.95173 |
| 18 | jazivxt | 0.95172 |
| 19 | Laura Liepa | 0.9517 |
| 20 | Overfitter | 0.9517 |

---

## Validation program (Handbook Stages 1–5)

| Stage | Status | Key result |
|-------|--------|------------|
| Core rule / prove the split | Done | Adv AUC 0.50080 → locked `StratifiedKFold(5, shuffle, rs=42)` |
| Stage 1 | Done | No time-order leakage / chronology OK |
| Stage 2 | Done | Baseline OOF 0.94698; promotion gate OOF↑ AND ≥4/5 folds↑ |
| Stage 3 | Done | Control `te_xgb` OOF 0.949315 |
| Stage 4 | Done | Promoted `te_xgb_smooth50` OOF 0.949447 |
| Stage 5 | Done | Honest self-stack OOF 0.949697 → est public ~0.9502 |

---

## What we tested

1. **Split integrity** — adversary train/test AUC ≈ 0.50; SKF locked.
2. **Feature views** — TE (target encoding), raw, digit-key under the same CV.
3. **Model families** — LGBM / XGB depth & leaves sweeps; TE-GLM.
4. **Promotion gate (Stage 2.2)** — reject unless OOF↑ and ≥4/5 folds↑ vs control.
5. **Self stacks (Stage 3–5)** — rank-mean / logit over our OOF banks only.
6. **Public-OOF ensembles** — hillclimb + logit over shared RealMLP/EBM/LGB/GLM OOFs (what hit LB).

---

## What worked

| Item | Detail |
|------|--------|
| Public OOF bank + rank/logit hillclimb | CV 0.95111 → public **0.95163** (best). Public−CV ≈ +0.00052 |
| Core rule / Stage 1 split proof | Adv AUC 0.50080; locked SKF |
| Stage 2 promotion gate | Rejected deeper LGBM (Train−Valid gap ~0.027) |
| TE view as control family | `te_xgb` / `te_lgbm` ~0.9493 OOF; TE beat raw and digit |
| Stage 4: TE smooth 20→50 | Only promoted lever — `te_xgb_smooth50` 0.949447 (5/5 folds) |
| Stage 5: shallow leaves | Best TE-LGBM at leaves=3 (0.949371); deeper trees overfit |

---

## What did not

| Item | Detail |
|------|--------|
| Self-only Stage 3–5 stacks for LB | Honest OOF 0.94970 → est public ~0.9502 (below 0.95163) |
| Digit-key-only arms | digit_lgbm 0.9447 / digit_xgb 0.9417; large Train−Valid gaps |
| Deeper trees (depth 8 / leaves 63) | Rejected by Stage 2.2 — OOF down, gap up |
| num_leaves 4→7, max_depth 2→3, TE-GLM | All failed promotion vs `te_xgb` control |
| Continuous hillclimb on full 700k×58 | Too slow; switched to thresholded rank climb |

---

## Headroom & next-submit estimates (no auto-submit)

- Gap to #1: **0.00029** among 393 teams.
- Self-trained Stage 5 stack cannot close it (~0.9502 est).
- Competitive path remains **public-OOF ensembles** (already at 0.95163).
- Optional restack with `smooth50` control: est public **~0.9516–0.9517**.

**Policy:** no Kaggle submit without explicit approval.
