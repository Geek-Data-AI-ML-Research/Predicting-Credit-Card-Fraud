# GM v10.2 Hardened Audit — Midpoint Paradox, G26/Fold1, S7 Two-Seed Staging

`NO_SUBMIT=True` · elapsed prep · smugcq read-only

## Executive verdict

1. **Midpoint paradox:** Slot3 was NOT a true 50/50 midpoint of official_s354 and z_S7_ours. gm_v10 built s7_clean = S7 + 0.08*L7F + 0.06*Gmed + 0.05*δ_v10 + 0.04*Strike9.5, then mid = 0.5*s354 + 0.5*s7_clean + 0.04*δ_v10. That degraded the S7 crest (Slot3 is 3.126e-04 from S7 and 9.188e-05 from s354). Slot1 stayed glued to s354 (0.12e-4) — public crest noise → 0.95352. A true 50/50 mid would be ~half the S7↔s354 distance from each (~1.86e-4).
2. **G26 5-fold test:** still **NOT** uploaded (`has_test_5fold=False`; local=['README.md', 'gpt2_s2026_oof_5fold.npy', 'gpt2_s2026_test_f0.npy']).
3. **Fold-1 canary BLOCKED:** AssertionError: oof shape (699960,10,2) vs expected (139992,10,2) after fold_1 ep8 — process DEAD; fold_1.pt exists; no oof_1/test_1 written Pod process dead; `fold_1.pt` on volume; **no auto-restart** (untouched rule).
4. Staged 3 NOSUBMIT candidates on `z_S7_ours` with `1−ρ≤0.15e-4` where possible.

## Step 1 — Pairwise 1−ρ matrix

| | s354 | S7 | S1 | S2 | S3 | s352 | areeb |
|---|---|---|---|---|---|---|---|
| s354 | 0.000e+00 | 3.723e-04 | 1.200e-05 | 2.800e-05 | 9.188e-05 | 3.088e-05 | 2.282e-04 |
| S7 | 3.723e-04 | 0.000e+00 | 4.021e-04 | 4.275e-04 | 3.126e-04 | 3.248e-04 | 1.075e-04 |
| S1 | 1.200e-05 | 4.021e-04 | 0.000e+00 | 3.354e-06 | 8.907e-05 | 4.235e-05 | 2.439e-04 |
| S2 | 2.800e-05 | 4.275e-04 | 3.354e-06 | 0.000e+00 | 9.727e-05 | 5.807e-05 | 2.619e-04 |
| S3 | 9.188e-05 | 3.126e-04 | 8.907e-05 | 9.727e-05 | 0.000e+00 | 9.125e-05 | 2.700e-04 |
| s352 | 3.088e-05 | 3.248e-04 | 4.235e-05 | 5.807e-05 | 9.125e-05 | 0.000e+00 | 1.924e-04 |
| areeb | 2.282e-04 | 1.075e-04 | 2.439e-04 | 2.619e-04 | 2.700e-04 | 1.924e-04 | 0.000e+00 |

- true_mid vs Slot3: `1.092877e-04` (0 ⇒ Slot3 was true mid)
- Slot3 was true 50/50? **False**
- Slot3 vs S7 / s354: `3.126464e-04` / `9.187537e-05`

### Linear decomp `unitize(L(s354))−unitize(L(S7))`

R²=`0.9044` · resid_rel=`0.3092`

| basis | coef |
|---|---:|
| A-F | -0.0055 |
| LM-F | -0.0014 |
| L7F-L7f0 | 0.0071 |
| GPD-F | 0.0187 |
| G26-MS | 0.0076 |
| -d1_v4 | 0.0231 |
| MLP-F | 0.0023 |
| GB-F | 0.0098 |

## Step 2 — G26 update + Fold-1

- G26 local files: `['README.md', 'gpt2_s2026_oof_5fold.npy', 'gpt2_s2026_test_f0.npy']`
- has_test_5fold: **False**
- Fold1 train_alive: **False**
- oof_1/test_1: **False** / **False**
- **CRASH:** AssertionError: oof shape (699960,10,2) vs expected (139992,10,2) after fold_1 ep8 — process DEAD; fold_1.pt exists; no oof_1/test_1 written
- Fix needed (not applied): save only held-out `oof[te]` rows shape (139992,10,2), or change expect_shape to (699960,10,2).

## Step 3 — Surgical moves on evaluators

- simple `z_ref+0.05*d1` AUC=`0.9526953`
- `z_S7_oof_matched` AUC=`0.9524522` folds=[0.95162, 0.952499, 0.957359, 0.952514, 0.94826]

### M1_twoseed_G26_MS

- **on_simple_best:** pos=5/5 meanΔ=0.026 ΔF0=0.021 folds=[0.021, 0.039, 0.009, 0.034, 0.028]
- **on_S7matched_best:** pos=5/5 meanΔ=0.041 ΔF0=0.049 folds=[0.049, 0.054, 0.002, 0.066, 0.033]

### M2_threeseed_G26_Gmed

- **on_simple_best:** pos=2/5 meanΔ=-0.003 ΔF0=0.003 folds=[0.003, -0.012, 0.006, -0.007, -0.008]
- **on_S7matched_best:** pos=2/5 meanΔ=-0.009 ΔF0=0.001 folds=[0.001, -0.025, 0.016, -0.024, -0.012]

### M3_exact_L7F_swap

- **areeb_recon_1rho_vs_pub095347:** `0.0`
- **1rho_S7_L7F_vs_S7:** `2.1067366061777726e-05`
- **proj_on_private_climb_S7_minus_areeb:** `-0.0299374272197206`
- **proj_on_s354_minus_S7:** `0.19841890345668203`
- **zero_neg_rung1_note:** `If proj_on_s354_minus_S7 ≈ 0 and proj_on_private_climb ≥ 0, swap stays off the public-ladder overstep that produced v9.2 Rung1=0.95351.`
- **oof_swap_vs_S7matched:** `{'fold_d_e5': [-0.287, -0.001, 0.006, 0.011, -0.009], 'pos5': 2, 'mean_d_e5': -0.056, 'auc_base': 0.952452237438358, 'auc_new': 0.9524516656688359}`

## Step 4 — Staged NOSUBMIT candidates

### cand1
- **path:** `/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/submissions/sub_v10p2_cand1_S7ours_exact_L7F_swap_NOSUBMIT.csv`
- **1rho_S7:** `2.1067366061777726e-05`
- **mean:** `0.016699998834038065`
- **sha16:** `918e7a08198b6dd7`

### cand2
- **path:** `/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/submissions/sub_v10p2_cand2_S7ours_plus_twoseed_G26_MS_0p11e4_NOSUBMIT.csv`
- **1rho_S7:** `1.1199999764177626e-05`
- **target:** `1.12e-05`
- **got:** `1.1199999764177626e-05`
- **weight:** `2.4349278246518224`
- **mean:** `0.016699998834038065`
- **sha16:** `ab82b9a8508f56e8`

### cand3
- **path:** `/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/submissions/sub_v10p2_cand3_S7ours_exact_L7F_plus_twoseed_0p14e4_NOSUBMIT.csv`
- **1rho_S7:** `2.097981575721608e-05`
- **1rho_vs_cand1:** `5.599426642355354e-07`
- **mean:** `0.016699998834038065`
- **sha16:** `237a4b8ff88a1fb6`
- **scale:** `0.2`

`NO_SUBMIT=True` — no Kaggle upload. Pod not restarted despite Fold1 crash.


## Live submit addendum (2026-11-12 13:20Z)

- **Pod:** `smugcqcy2oqzeg` terminated via MCP `delete-pod`; `list-pods` = empty ($0.00/hr). Optional ckpt rsync skipped (SSH refused mid-terminate).
- **cand2 scale check:** `1-rho(S7)=1.120e-5`, mean=`0.016700`, n=`299984`, NaN=`0`. Equivalent to `force_law(S7, L(S7)+α(G26-MS))` with `α_exact=0.03612685` (`α_user=0.036134`; `1-rho` vs cand2 = `0`).
- **LB_cand2 = 0.95355** (ref `58274077`) — new ATH → triggered α=0.060 follow-up.
- **LB_cand2b (α=0.060, 1-rho=3.089e-5) = 0.95355** (ref `58274099`) — flat vs cand2.
- Remaining daily submissions after both: **4**.
