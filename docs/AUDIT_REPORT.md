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

---

## Nov 07, 2026 — Comprehensive Grandmaster Audit Dossier (`0.95243` Public LB / `0.9523401` OOF)

### 1. Executive Status Update (`2026-11-07 19:00 UTC`)
- **Best Public Leaderboard Score (Ours):** **`0.95243`** (`sub_ev4_live10_reverse_ev2_plus_hc_ascent.csv`)
- **Public 200-Member `wide_stack_nested` Baseline:** **`0.9525043`** (`0.95267` solo LB; `0.95285` w/ Part 9; `0.95292` w/ Part 9 + `megayak_v10`)
- **Previous Verified 5-Fold OOF Champion (`P_super_v5_harvest`):** **`0.9523401`** (`+3.300e-05` over `v4p_hc`)
- **New Verified 5-Fold OOF Champion (`P_super_v6_grandmaster`):**
  - **5-Fold Nested OOF ROC-AUC:** **`0.9525275`** (**`+2.328e-05` over `wide_stack_nested` (`0.9525043`), `5/5 folds won`!** And **`+2.204e-04` (`+2.20 bps`) over `v4p_hc` (`0.9523071`)**!)
  - **Mean-Weight OOF ROC-AUC:** **`0.9525693`** (**`+6.500e-05` (`+0.65 bps`) over `wide_stack_nested`!**)
  - **Folds 0 & 1 (`279,984` rows) with `megayak` Target-First LM (`lm_oof.npy`):** **`0.9522080` vs `0.9521790` (`+2.900e-05`)**
- **A40 GPU Compute (`ht590jc0rpb9kz`):** Stopped (`EXITED`, `$0.00/hr` compute billing). All `v5` and `v6` artifacts verified locally.

### 2. Full Forensic Audit of All Paradigms Tested
Across **8 Paradigms / 40 CPU Levers**, **3 GPU Deep-Learning/Manifold Campaigns (`A, B, C`)**, **16 `matrix_v2` GPU Levers**, and **5 Orthogonal Tournament Playbooks (`PB1–PB5`)**, a single unifying law emerged:

> **The S6E11 Synthetic Generator Law:**
> Because S6E11 was generated from Udit Jain's 20k NumPy rule/table script + chimera splicing, **continuous manifold methods** (`torch.cdist` kNN graphs, `TabR` attention, `TabM k=32` PLE, Copula QDA, SwiGLU Denoising Autoencoders, Sliced-Wasserstein OT, and unsupervised continuous `groupby` aggregations) carry **zero orthogonal signal (`\Delta\text{AUC} \le 0`)** once conditioned on `KMAT_108`.
> **100% of true out-of-fold lift comes from discrete quantization / string-level target encoding & token log-likelihood ratios** that recover the exact synthetic lookup boundaries (`P4_L4.3`, `P4_L4.2`, `B_xgb_llr`, `P4_L4.5`).

### 3. Proprietary Winning Components Locked in `P_super_v5_harvest` (`0.9523401` OOF)

| Component | Proprietary Mechanism | 5-Fold Test-1 $\Delta\text{AUC}$ | Fold Wins | KS (OOF vs Test) | Stack Weight $w_j$ | Local File (`preds/`) |
|---|---|---:|---:|---:|---:|---|
| **`v4p_hc` Baseline** | `0.84*rk(v4p) + 0.12*rk(hc) + 0.04*rk(gb_super)` (81-col + 15 L4 regime Newton-Cholesky bank) | Baseline (`0.9523071`) | — | `0.0000` | `1.000` (anchor) | `P_super_v4p_hc_{oof,test}.npy` |
| **`P4_L4.3_pure5f`** | **Velocity & Merchant-Risk Step-Grid Parity $\times$ 64-Flag Cross TE** fit via `Ridge(alpha=500)` on prevalence-calibrated residual $y - \sigma(z_{\text{init}})$ with per-fold ECDF alignment | **`+1.553e-05`** | `4/5` | `0.0360` | `-0.021` | `matrix_v2/P4_L4.3_pure5f_{oof,test}.npy` |
| **`P4_L4.2_pure5f`** | **Bulsah Sub-Float ASCII String & Digit-Pair TE** fit via `Ridge(alpha=500)` on calibrated residual with per-fold ECDF alignment | **`+1.508e-05`** | `4/5` | `0.0014` | `-0.024` | `matrix_v2/P4_L4.2_pure5f_{oof,test}.npy` |
| **`B_xgb_llr`** | **72-Col Transformer Token Log-Likelihood Ratios (`LLR_72`) + `KMAT_108` CUDA `XGBoost`** (`max_depth=3`) | **`+6.463e-06`** | `5/5` | `0.0000` | `-0.034` | `paradigm_B/xgb_llr_{oof,test}.npy` |
| **`P4_L4.5_pure5f`** | **Top-60 2nd-Order High-Card Key-Pair Hierarchical EB TE** fit via `max_depth=2` Lever-Only CUDA `XGBoost` | **`+3.340e-06`** | `2/5` | `0.0027` | `-0.022` | `matrix_v2/P4_L4.5_pure5f_{oof,test}.npy` |
| **`harvest_p4_cuda_xgb`** | **Combined Pure 5-Fold `P4` Blend** (`0.458*L4.3 + 0.444*L4.2 + 0.098*L4.5` on rank-logit scale) | **`+3.623e-05`** | `4/5` | `0.0000` | — | `matrix_v2/harvest_p4_cuda_xgb_{oof,test}.npy` |
| **`P_super_v5_harvest`** | **5-Fold Out-of-Fold Coordinate-Ascent Super-Stack** on $z_{\text{base}} + \sum_j w_j Z_{\text{orth}, j}$ | **`+2.133e-05`** *(Raw $\Delta$: **`+3.300e-05`**)* | `4/5` | `0.0000` | **`0.9523401` OOF** | `P_super_v5_harvest_{oof,test}.npy` |

### 4. Ready-to-Submit Files (Locally Verified, `STRICT NO_SUBMIT` Enforced)
All submission files have `299,984` rows, zero NaNs, `KS = 0.00000`, and verified SHA-256 hashes:

#### Tier 1 — `P_super_v6_grandmaster` (`0.9525275` Nested OOF / `0.9525693` Mean-W OOF, 5/5 Folds Won over `wide_stack`)
1. **`submissions/sub_v6_gm_on_maly095292.csv`** (`SHA-256: eb6f4a0677183527`, **#1 Highest-EV Submit to Beat `0.95292` / `0.953+`**)
   - Takes the `0.95292` LB `malyshevdanil` v3 backbone (`0.82 * logit(rk(0.65*z(wide_200) + 0.35*z(part9))) + 0.18 * logit(rk(megayak_v10))`) and injects our **5/5-fold-validated, ECDF-aligned (`KS=0.00000`) orthogonal residual** $\Delta z_{\text{v6}} = +0.0530 r_{\text{v4p\_hc}} - 0.0222 r_{\text{P4\_L4.3}} - 0.0208 r_{\text{P4\_L4.2}} - 0.0224 r_{\text{P4\_L4.5}} - 0.0430 r_{\text{B\_xgb\_llr}}$.
   - **Spearman $\rho$:** `0.999701` vs `maly_0.95292` | `0.997275` vs `wide_200` | `0.996980` vs `sub_ev4` (`KS = 0.00000`).
2. **`submissions/sub_v6_gm_triad_maly095292_ev4_p4gpu.csv`** (`SHA-256: d31ec644d8e66639`, **#2 Grandmaster Triad Submit**)
   - Combines `0.92 * z(maly_0.95292) + 0.08 * z(sub_ev4_0.95243)` + `1.0x` our pure GPU orthogonal residual (`P4_L4.3 + P4_L4.2 + P4_L4.5 + B_xgb_llr`).
   - **Spearman $\rho$:** `0.999571` vs `maly_0.95292` | `0.998490` vs `wide_200` | `0.995091` vs `sub_ev4` (`KS = 0.00000`).
3. **`submissions/sub_v6_gm_pure_cv095253.csv`** (`SHA-256: fa94e8f9f7aa6b1a`, **#3 Pure Nested-CV-Optimal `0.9525275` / `0.9525693` Backup**)
   - Pure `preds/P_super_v6_grandmaster_test.npy` (`z(wide_200) + Δz_v6 + 0.015 * r(megayak_lm)`), every component verified on 5-fold OOF (`5/5` folds won).
   - **Spearman $\rho$:** `0.999655` vs `wide_200` | `0.998102` vs `maly_0.95292` (`KS = 0.00000`).

#### Tier 2 — `P_super_v5_harvest` (`0.9523401` Self+Part8 Bank OOF)
4. **`submissions/sub_v5_harvest_on_ev4_half.csv`** (`SHA-256: ccb4016d70c317d5`, $\rho = 0.999920$ vs `sub_ev4`)
5. **`submissions/sub_v5_harvest_on_ev4.csv`** (`SHA-256: 953e118caff0f298`, $\rho = 0.999679$ vs `sub_ev4`)
6. **`submissions/sub_v5_harvest_pure.csv`** (`SHA-256: c43df92be3b1d9a1`, $\rho = 0.999123$ vs `sub_ev4`)

