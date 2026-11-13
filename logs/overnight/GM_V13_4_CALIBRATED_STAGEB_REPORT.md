# GM V13.4 — Platt-Calibrated Init + Full Tabular×Disagreement Stage-B

**STRICT NO_SUBMIT=True** · POD_TOUCHES=0 · elapsed 1133.4s

**Script:** `scripts/gm_v13_4_calibrated_init_tabular_stageB.py`

## Bugs fixed

1. **Uncalibrated zero-mean `init_score`:** `mean(σ(z_ATH))` was ~0.50; true prevalence=0.016700 (logit≈-4.076). LOFO Platt η now asserts `|mean(σ(η_tr))-y_tr.mean()|<1e-6`.
2. **Restored X_v15:** 89 cols (68 KMAT + P4 + zero-sum contrasts + disagreement + `q_ATH_bin`); **no raw z_ATH in X** (max|ρ| vs z_ATH=0.912).

## Step 1 — daoviet CSVs + G26

- G26 `test_5fold`: **ABSENT**
- `submission_095354_exact.csv`: 1−ρ(LIVE)=2.087441e-03 · R²(Bt)=0.9846479497674777 · spanned>0.999=False
- `submission_summit_095370.csv`: 1−ρ(LIVE)=2.608250e-03 · R²(Bt)=0.9855761024618416 · spanned>0.999=False
- `submission_verified_095370.csv`: 1−ρ(LIVE)=8.587434e-04 · R²(Bt)=0.9728849999158506 · spanned>0.999=False

## Step 2 — Platt η + X_v15

- τ=1.0: mean(σ(η_OOF))=0.016699 · fold prev_err max=1.04e-17
- τ=0.5: mean(σ(η_OOF))=0.016699 · fold prev_err max=6.94e-18
- X_v15 shape OOF=(699960, 89) · denoise disp G26=1.0000 L7s=1.0000

## Step 3 — δ1_v15 4-bar screen

| mode | w | main | min | sf25 | t | W-p | boot90 | disp | R²(δ\|LIVE) | PASS |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| `d1_v15_tau1.00` | 0.02 | 5/5 | 1.33 | 18/25 | 3.18 | 0.00254 | [2.509, 7.131] | 1.0 | 1e-05 | True |
| `d1_v15_tau1.00` | 0.03 | 5/5 | 1.11 | 18/25 | 3.04 | 0.00441 | [3.124, 10.166] | 1.0 | 1e-05 | True |
| `d1_v15_tau1.00` | 0.04 | 5/5 | 0.15 | 16/25 | 2.66 | 0.011 | [3.04, 12.58] | 1.0 | 1e-05 | False |
| `d1_v15_tau1.00` | 0.06 | 3/5 | -3.55 | 15/25 | 2.02 | 0.0479 | [1.411, 16.517] | 1.0 | 1e-05 | False |
| `d1_v15_tau1.00` | 0.08 | 3/5 | -9.26 | 15/25 | 1.32 | 0.126 | [-1.918, 18.709] | 1.0 | 1e-05 | False |
| `d1_v15_tau0.50` | 0.02 | 1/5 | -3.34 | 7/25 | -1.83 | 0.972 | [-2.331, -0.041] | 1.0 | 6e-05 | False |
| `d1_v15_tau0.50` | 0.03 | 1/5 | -5.07 | 7/25 | -1.88 | 0.972 | [-3.61, -0.349] | 1.0 | 6e-05 | False |
| `d1_v15_tau0.50` | 0.04 | 1/5 | -7.15 | 7/25 | -2.10 | 0.976 | [-5.079, -0.775] | 1.0 | 6e-05 | False |
| `d1_v15_tau0.50` | 0.06 | 1/5 | -11.87 | 7/25 | -2.78 | 0.996 | [-8.625, -2.51] | 1.0 | 6e-05 | False |
| `d1_v15_tau0.50` | 0.08 | 1/5 | -17.35 | 7/25 | -3.17 | 0.999 | [-12.454, -4.599] | 1.0 | 6e-05 | False |
| `d1_v15_blend` | 0.02 | 3/5 | -1.62 | 16/25 | 1.13 | 0.105 | [-0.342, 2.762] | 1.0 | 4e-05 | False |
| `d1_v15_blend` | 0.03 | 3/5 | -2.82 | 16/25 | 0.96 | 0.126 | [-0.767, 3.722] | 1.0 | 4e-05 | False |
| `d1_v15_blend` | 0.04 | 3/5 | -4.55 | 15/25 | 0.75 | 0.19 | [-1.542, 4.54] | 1.0 | 4e-05 | False |
| `d1_v15_blend` | 0.06 | 3/5 | -8.20 | 14/25 | 0.33 | 0.365 | [-3.492, 5.652] | 1.0 | 4e-05 | False |
| `d1_v15_blend` | 0.08 | 3/5 | -12.74 | 10/25 | -0.12 | 0.674 | [-6.17, 5.849] | 1.0 | 4e-05 | False |

## Staging

- Both `d1_v15_tau1.00` **w=0.02** and **w=0.03** clear Bars 1–4; staged the higher-mean passer:
  - **STAGED** `submissions/sub_v13p4_verified_d1_v15_tau1.00_w0.03_NOSUBMIT.csv` (still **NOSUBMIT**)
- `τ=0.50` and blend fail (tempered slope alone is too soft without the full Newton residual path).

## Bottom line

1. Platt-calibrated `init_score` restores correct prevalence (~0.0167) and zero-mean initial gradients.
2. Full tabular×disagreement `X_v15` restores the feature surface Stage-B needs without z_ATH shortcuts.
3. **δ1_v15 cleared Bars 1–4** — staged NOSUBMIT candidate(s).
4. G26 5-fold test still absent; daoviet 095370 CSVs are (near-)spanned podium blends, not new GPD.
