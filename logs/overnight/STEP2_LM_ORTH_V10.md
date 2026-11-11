# Step 2 — LM Orth + v10 Transductive Bulsah (NO_SUBMIT)

Elapsed **1981.0s**. z_ref_o AUC **0.9526612**.

## Part 1 — soft_diff / gb_soft double-orth vs LM bank

Dropped permanently: `idf_p3`, `clean035`.

### soft_diff_orth_LM

- Free-grid nested: Δ=15.64e-5 w=0.08 4/5 | disp=1.028 KS=0.006901
- vs z_ref only (pre-LM): Δ=15.76e-5 4/5
- max|corr| vs controls: 1.11e-02

| target 1-ρ e-4 | achieved | w | ΔOOF e-5 | pos | disp | KS |
|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.2 | 0.010669 | 3.63 | 5/5 | 1.028 | 0.006901 |
| 0.5 | 0.5 | 0.016876 | 5.5 | 5/5 | 1.028 | 0.006901 |
| 1.0 | 1.0 | 0.023877 | 7.39 | 5/5 | 1.028 | 0.006901 |

### gb_soft_bag3_orth_LM

- Free-grid nested: Δ=6.36e-5 w=0.06 4/5 | disp=0.8643 KS=0.048078
- vs z_ref only (pre-LM): Δ=4.88e-5 4/5
- max|corr| vs controls: 1.91e-02

| target 1-ρ e-4 | achieved | w | ΔOOF e-5 | pos | disp | KS |
|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.2 | 0.011691 | 2.26 | 5/5 | 0.8643 | 0.048078 |
| 0.5 | 0.5 | 0.018486 | 3.34 | 5/5 | 0.8643 | 0.048078 |
| 1.0 | 1.0 | 0.026145 | 4.37 | 5/5 | 0.8643 | 0.048078 |

## Part 2 — v10_transductive_bulsah audit

| member | ΔOOF e-5 | w | pos | disp | KS | 1-ρ e-4 | gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| v10_ridge_resid | 6.03 | 0.03 | 5/5 | 0.9294 | 0.008715 | 0.663 | PASS |
| v10_lgb_d3_bag5 | 45.11 | 0.08 | 5/5 | 0.944 | 0.017096 | 12.889 | fail |
| v10_cb_d4_bag3 | 2.11 | 0.02 | 4/5 | 0.8813 | 0.005974 | 0.067 | PASS |
| v10_combo3 | 18.33 | 0.06 | 5/5 | 1.0435 | 0.0025 | 3.397 | PASS |

Artifacts: `/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/preds/v10_transductive_bulsah`
