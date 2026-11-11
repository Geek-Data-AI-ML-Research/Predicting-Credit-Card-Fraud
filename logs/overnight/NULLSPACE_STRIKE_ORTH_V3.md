# Null-Space Strike-Orth Contrast Stacker v3

**NO_SUBMIT=True**. Elapsed **119.7s**. z_ref_o **0.9526612**.

## Per-column Method-B smooth hygiene

| col | disp | KS | ok |
|---|---:|---:|:---:|
| c01_gpt2m_lm12 | 1.000000 | 0.000188 | Y |
| c02_gpt2m_prec | 1.000000 | 0.000131 | Y |
| c03_lm12_rev | 1.000000 | 0.000212 | Y |
| c04_masa_lm12 | 1.000000 | 0.000158 | Y |
| c05_gpt2_3ep_clean_rev | 1.000000 | 0.000387 | Y |
| c06_xgbf_tabm | 1.000000 | 0.000193 | Y |
| c07_gpt2m_tabm | 1.000000 | 0.000124 | Y |
| c08_P4_L42 | 1.000000 | 0.000138 | Y |
| c09_P4_L43 | 1.000000 | 0.000300 | Y |
| c10_P4_L45 | 1.000000 | 0.000047 | Y |
| c11_B_xgb_llr | 1.000000 | 0.005946 | N |
| c12_v10_cb_d4 | 1.000000 | 0.004650 | N |
| c13_v10_ridge | 1.000000 | 0.002671 | N |

Hygiene pass: **10/13**.

## Nested comparison (after feature-space $P_\perp$)

| model | ΔOOF e-5 | pos | folds | disp | KS | cos(e1,e2,leader,soft) |
|---|---:|---:|---|---:|---:|---|
| SetA_c01..c12/NullSpace_Ridge | -1.24 | 2/5 | [-8.15, 3.84, -0.06, 3.66, -4.23] | 1.0000 | 0.000309 | (+0.034,-0.0000,-0.0000,+0.0000) |
| SetA_c01..c12/NullSpace_AUC_Polish | -0.97 | 2/5 | [-9.09, -2.61, 2.71, 5.32, -0.01] | 1.0000 | 0.000738 | (+0.062,-0.0000,-0.0000,-0.0000) |
| SetA_c01..c12_e1prior/NullSpace_Ridge | -1.24 | 2/5 | [-8.15, 3.84, -0.06, 3.66, -4.23] | 1.0000 | 0.000309 | (+0.034,-0.0000,-0.0000,+0.0000) |
| SetA_c01..c12_e1prior/NullSpace_AUC_Polish | -0.97 | 2/5 | [-9.09, -2.61, 2.71, 5.32, -0.01] | 1.0000 | 0.000738 | (+0.062,-0.0000,-0.0000,-0.0000) |
| SetB_c01..c13/NullSpace_Ridge | +2.53 | 4/5 | [-5.96, 1.69, 5.83, 6.15, 5.2] | 1.0000 | 0.001703 | (+0.033,+0.0000,-0.0000,-0.0000) |
| SetB_c01..c13/NullSpace_AUC_Polish | +4.14 | 4/5 | [-6.35, 0.77, 9.96, 7.16, 9.0] | 1.0000 | 0.000876 | (+0.056,-0.0000,-0.0000,-0.0000) |
| SetB_c01..c13_e1prior/NullSpace_Ridge | +2.53 | 4/5 | [-5.96, 1.69, 5.83, 6.15, 5.2] | 1.0000 | 0.001703 | (+0.033,+0.0000,-0.0000,-0.0000) |
| SetB_c01..c13_e1prior/NullSpace_AUC_Polish | +4.14 | 4/5 | [-6.35, 0.77, 9.96, 7.16, 9.0] | 1.0000 | 0.000876 | (+0.056,-0.0000,-0.0000,-0.0000) |

**Winner:** `SetB_c01..c13/NullSpace_AUC_Polish` Δ=+4.14e-5 pos=4/5

### Winner weights

```
{
  "c01_gpt2m_lm12": 0.031065,
  "c02_gpt2m_prec": -0.031375,
  "c03_lm12_rev": 5e-06,
  "c04_masa_lm12": -0.025612,
  "c05_gpt2_3ep_clean_rev": -0.022186,
  "c06_xgbf_tabm": 0.004448,
  "c07_gpt2m_tabm": 0.000421,
  "c08_P4_L42": 0.011053,
  "c09_P4_L43": -0.001641,
  "c10_P4_L45": -0.00934,
  "c11_B_xgb_llr": 0.011291,
  "c12_v10_cb_d4": 0.000964,
  "c13_v10_ridge": 0.033811
}
```

## Staged onto St9.5 (NOT submitted)

- `/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/submissions/sub_strike10_nullspace_v3_0p15e4.csv`: 1−ρ=0.15e-4 w=0.131268 mean=0.016700
- `/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/submissions/sub_strike10_nullspace_v3_0p30e4.csv`: 1−ρ=0.3e-4 w=0.185762 mean=0.016700

`NO_SUBMIT=True`.
