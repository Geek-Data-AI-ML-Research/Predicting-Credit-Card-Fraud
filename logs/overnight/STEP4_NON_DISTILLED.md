# Step 4 — Non-Distilled Audit + Native CTR CatBoost (NO_SUBMIT)

Elapsed **2790.4s**. SoftPseudo/zanch **CLOSED**. z_ref_o **0.9526612**.

## Part 1 — Strike 6→9 axis

Exact formulas:

- **Strike4A**: `z_st3 + 0.70*(z_st3-z_st2) + 0.15*(z_st2-z_a3)  → LB 0.95323`
- **Strike6**: `z_st4 + 0.30*(z_st4-z_st3) - 0.35*v_bad_new5  → LB 0.95325`
- **Strike7**: `z_st6 + 0.85*(z_st6-z_st4) + 0.15*(z_st4-z_st3)  → LB 0.95327`
- **Strike8**: `z_st7 + 0.90*(z_st7-z_st6)  → LB 0.95329`
- **Strike9**: `z_st8 + 0.78*(z_st8-z_st7) - 0.14*u_bad5  → LB 0.95330`
- **u_bad5**: `u_bad5 = (z_st5-z_st4) - 0.45*(z_st4-z_st3)`

St9 reconstruct corr: **1.0**

Ladder steps:

```
{
  "St6->St7": {
    "1m_rho_e4": 0.1329,
    "dLB": 2e-05
  },
  "St7->St8": {
    "1m_rho_e4": 0.1075,
    "dLB": 2e-05
  },
  "St8->St9": {
    "1m_rho_e4": 0.1302,
    "dLB": 1e-05
  }
}
```

### Strike 9.5 / 10_axis continuation candidates (test-space, NO_SUBMIT)

| a_cont | b_neg | 1-ρ(St9) e-4 | step_rms | formula |
|---:|---:|---:|---:|---|
| 0.55 | 0.08 | 0.0698 | 0.00733 | `z_st9 + 0.55*(z_st9-z_st8) - 0.08*u_bad5` |
| 0.7 | 0.1 | 0.112 | 0.00927 | `z_st9 + 0.7*(z_st9-z_st8) - 0.1*u_bad5` |
| 0.85 | 0.12 | 0.1641 | 0.0112 | `z_st9 + 0.85*(z_st9-z_st8) - 0.12*u_bad5` |
| 1.0 | 0.14 | 0.2261 | 0.01313 | `z_st9 + 1.0*(z_st9-z_st8) - 0.14*u_bad5` |

## Part 1 — LM / H2_5F

gpt2_med folds present: **True**
LM bank: `['lm12_wide', 'lm12_raw', 'masaya_gpt2', 'gpt2_rev', 'gpt2_3ep', 'lm_gpt2_3ep', 'gpt2_medium', 'B_xgb_llr', 'megayak']`
- H2_5F orth[z_ref,lm12]: Δ=0.07e-5 3/5 folds=[0.72, -0.45, 0.32, -0.41, 0.16] disp=0.8761 KS=0.02886
- LM↔Tree disagreement gate: Δ=1.58e-5 2/5 disp=1.2229 KS=0.040307

## Part 2 — `cb_native_ctr4_5f` forensic gate

| member | ΔOOF e-5 | w | pos | folds | disp | KS | 1-ρ e-4 | gate |
|---|---:|---:|---:|---|---:|---:|---:|:---:|
| cb_native_ctr4_5f | 0.63 | 0.01 | 4/5 | [0.4, 0.6, 1.64, -0.11, 1.88] | 0.6807 | 0.105478 | 0.063 | fail |

### Step grid

| 1-ρ e-4 | w | ΔOOF e-5 | folds | pos |
|---:|---:|---:|---|---:|
| 0.2 | 0.017924 | 0.48 | [0.57, 0.78, 1.67, -0.27, 2.49] | 4/5 |
| 0.35 | 0.023752 | 0.05 | [0.66, 0.7, 1.19, -0.38, 2.47] | 4/5 |
| 0.5 | 0.02843 | -0.45 | [0.61, 0.54, 0.32, -0.41, 2.09] | 4/5 |
| 1.0 | 0.040347 | -2.38 | [0.49, -0.06, -2.73, -0.61, -0.11] | 1/5 |

Artifacts: `/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/preds/cb_native_ctr4_5f`

`NO_SUBMIT=True`.

## Part 2 salvage — residual CTR (no OD/baseline collapse)

| cb_native_ctr4_5f_resid | 2.41 | 0.03 | 3/5 | [-2.12, 0.87, 7.03, -1.41, 7.33] | 1.0128 | 0.023914 | 0.233 | fail |

### Salvage step grid

| 1-ρ e-4 | w | ΔOOF e-5 | folds | pos |
|---:|---:|---:|---|---:|
| 0.2 | 0.027773 | 2.42 | [-1.88, 0.94, 6.72, -1.04, 7.03] | 3/5 |
| 0.35 | 0.036838 | 2.22 | [-3.26, 0.46, 7.9, -2.43, 8.17] | 3/5 |
| 0.5 | 0.044113 | 1.74 | [-4.66, -0.16, 8.57, -3.86, 8.65] | 2/5 |
| 1.0 | 0.062684 | -0.5 | [-8.92, -2.5, 9.19, -8.68, 8.53] | 2/5 |
