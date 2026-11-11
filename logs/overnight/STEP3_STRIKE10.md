# Step 3 — De-bias v10 + Strike-10 Stage (NO_SUBMIT)

Elapsed **1289.2s**. z_ref_o **0.9526612**. `NO_SUBMIT=True`.

## Part 1 — z_anchor diagnosis

- zanch features: **140**/481
- Top-20 LGB gain: **14/20** are `zanch_*`

| rank | feat | gain | split | zanch? |
|---:|---|---:|---:|:---:|
| 1 | `zanch_eb20_account_balance_usd` | 1384.1 | 67 | True |
| 2 | `zanch_eb5_account_balance_usd` | 1144.1 | 52 | True |
| 3 | `zanch_eb5_distance_from_home_km` | 590.7 | 54 | True |
| 4 | `zanch_eb5_hours_since_last_txn` | 535.6 | 48 | True |
| 5 | `zanch_eb20_hours_since_last_txn` | 520.4 | 43 | True |
| 6 | `zanch_eb20_distance_from_home_km` | 437.1 | 37 | True |
| 7 | `zanch_eb5_amount_usd` | 433.6 | 35 | True |
| 8 | `loo_mean_velocity_score_account_balance_usd` | 318.8 | 32 | False |
| 9 | `loo_mean_merchant_risk_score_account_balance_usd` | 313.2 | 29 | False |
| 10 | `zanch_eb20_account_balance_usdxcard_type` | 266.0 | 19 | True |
| 11 | `zanch_eb5_account_balance_usdxcard_type` | 240.6 | 15 | True |
| 12 | `loo_mean_amount_usd_velocity_scorexflag_bitmask_64` | 226.5 | 21 | False |
| 13 | `zanch_disp20_device_type` | 210.4 | 18 | True |
| 14 | `loo_mean_merchant_risk_score_account_balance_usdxcard_type` | 206.8 | 21 | False |
| 15 | `zanch_eb20_card_type` | 201.8 | 18 | True |
| 16 | `zanch_disp20_card_type` | 195.2 | 17 | True |
| 17 | `loo_mean_velocity_score_amount_usdxcard_type` | 193.9 | 20 | False |
| 18 | `zanch_eb5_account_balance_usdxflag_bitmask_64` | 191.5 | 14 | True |
| 19 | `loo_mean_amount_usd_velocity_score` | 184.6 | 18 | False |
| 20 | `zanch_eb20_amount_usd` | 180.2 | 16 | True |

### Worst zanch std_te/std_tr shifts

| feat | std_ratio | mean_diff |
|---|---:|---:|
| `zanch_disp5_auth_method` | 0.9908 | -0.000357 |
| `zanch_disp20_auth_method` | 0.9908 | -0.000357 |
| `zanch_disp5_card_typexauth_method` | 0.9909 | -0.000343 |
| `zanch_disp20_card_typexauth_method` | 0.9909 | -0.000343 |
| `zanch_disp5_auth_methodxdevice_type` | 0.9909 | -0.000303 |
| `zanch_disp20_auth_methodxdevice_type` | 0.9909 | -0.000302 |
| `zanch_disp5_velocity_scorexauth_method` | 0.992 | -0.000385 |
| `zanch_eb5_auth_methodxdevice_type` | 0.9922 | -0.003551 |
| `zanch_eb20_auth_methodxdevice_type` | 0.9922 | -0.003548 |
| `zanch_eb5_auth_method` | 0.9923 | -0.003613 |

## Part 1 — pure-FE orth audit

| member | ΔOOF e-5 | w | pos | disp | KS | 1-ρ e-4 | gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| v10_ridge_pure_fe_orth_LM | 0.0 | 0.0 | 0/5 | 0.9261 | 0.025387 | 0.0 | fail |
| v10_lgb_pure_fe_orth_LM | 5.41 | -0.045 | 5/5 | 0.8101 | 0.051242 | 3.837 | fail |

Portfolio A includes lgb_pure: **False**. A weights: `[0.436, 0.564]`

## Residual correlation (OOF)

```
{
  "v10_ridge_orth_LM": {
    "v10_ridge_orth_LM": 1.0,
    "v10_cb_orth_LM": 0.5344,
    "v10_lgb_pure_fe_orth_LM": 0.2524,
    "soft_diff_orth_LM": 0.0504
  },
  "v10_cb_orth_LM": {
    "v10_ridge_orth_LM": 0.5344,
    "v10_cb_orth_LM": 1.0,
    "v10_lgb_pure_fe_orth_LM": 0.1725,
    "soft_diff_orth_LM": 0.0348
  },
  "v10_lgb_pure_fe_orth_LM": {
    "v10_ridge_orth_LM": 0.2524,
    "v10_cb_orth_LM": 0.1725,
    "v10_lgb_pure_fe_orth_LM": 1.0,
    "soft_diff_orth_LM": 0.0321
  },
  "soft_diff_orth_LM": {
    "v10_ridge_orth_LM": 0.0504,
    "v10_cb_orth_LM": 0.0348,
    "v10_lgb_pure_fe_orth_LM": 0.0321,
    "soft_diff_orth_LM": 1.0
  }
}
```

## Part 2/3 — Strike-10 step-grid

| portfolio | 1-ρ e-4 | w | ΔOOF e-5 | folds e-5 | pos | disp | KS | stage |
|---|---:|---:|---:|---|---:|---:|---:|:---:|
| A_V10_PassOnly | 0.2 | 0.023452 | 4.46 | [0.34, 1.63, 4.83, 7.88, 8.25] | 5/5 | 0.9136 | 0.006199 | fail |
| A_V10_PassOnly | 0.4 | 0.033155 | 4.41 | [-0.92, 0.38, 4.06, 9.48, 10.07] | 4/5 | 0.9136 | 0.006199 | fail |
| A_V10_PassOnly | 0.6 | 0.040547 | 3.49 | [-2.48, -1.76, 1.94, 9.95, 10.79] | 3/5 | 0.9136 | 0.006199 | fail |
| A_V10_PassOnly | 1.0 | 0.052152 | 0.39 | [-5.98, -6.74, -4.15, 9.44, 10.9] | 2/5 | 0.9136 | 0.006199 | fail |
| B_V10_Combo3_Clamped | 0.2 | 0.01457 | 7.65 | [4.4, 5.82, 8.5, 10.33, 10.1] | 5/5 | 0.6969 | 0.089976 | fail |
| B_V10_Combo3_Clamped | 0.4 | 0.020613 | 10.21 | [5.87, 7.49, 11.34, 14.21, 13.7] | 5/5 | 0.6969 | 0.089976 | fail |
| B_V10_Combo3_Clamped | 0.6 | 0.025249 | 11.98 | [6.88, 8.6, 13.16, 16.91, 16.19] | 5/5 | 0.6969 | 0.089976 | fail |
| C_V10_PassOnly_plus_soft | 0.3 | 0.016699 | 6.81 | [2.53, 3.31, 9.94, 9.39, 10.09] | 5/5 | 0.958 | 0.006251 | PASS |
| C_V10_PassOnly_plus_soft | 0.5 | 0.021565 | 8.38 | [2.97, 3.88, 12.28, 11.77, 12.54] | 5/5 | 0.958 | 0.006251 | PASS |
| C_V10_PassOnly_plus_soft | 0.8 | 0.027287 | 10.0 | [3.18, 4.38, 14.65, 14.19, 15.42] | 5/5 | 0.958 | 0.006251 | fail |
| D_soft_diff_orth_LM_micro | 0.3 | 0.013069 | 4.35 | [2.0, 1.92, 7.24, 5.41, 5.9] | 5/5 | 1.028 | 0.006901 | fail |

## Staged CSVs (NOT submitted)

- `/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/submissions/sub_strike10A_v10_clean_micro.csv` portfolio=C_V10_PassOnly_plus_soft Δ=8.38e-5 @1-ρ=0.5e-4 mean=0.016699998741294804
- `/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/submissions/sub_strike10B_v10_softdiff_micro.csv` portfolio=C_V10_PassOnly_plus_soft Δ=6.81e-5 @1-ρ=0.3e-4 mean=0.01669999876073698

**Await confirmation before 1-slot live LB probe.**
