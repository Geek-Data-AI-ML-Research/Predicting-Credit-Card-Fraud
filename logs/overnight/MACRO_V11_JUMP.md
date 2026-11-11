# Macro-Jump v11 Super Refeed (NO_SUBMIT)

Elapsed **802.3s**. z_ref_o **0.9526612**. Anchor St9.5 **0.95332**.

X_macro cols: **113**. Soft-bad hygiene cols: 100.

## Family + Macro gate

| member | ΔOOF e-5 | w | pos | folds | disp | KS |
|---|---:|---:|---:|---|---:|---:|
| xgb | 0.66 | -0.015 | 2/5 | [1.96, -0.46, -3.44, 2.08, -1.13] | 0.9539 | 0.030371 |
| lgb | 0.3 | -0.01 | 2/5 | [0.67, -0.52, -2.47, 1.19, -0.53] | 0.9392 | 0.041399 |
| cat | 1.05 | -0.02 | 1/5 | [1.39, -0.98, -2.45, -0.34, -0.64] | 0.9138 | 0.150928 |
| l4 | 0.75 | -0.02 | 2/5 | [-1.48, -0.92, 0.51, -0.35, 1.47] | 0.9551 | 0.018962 |
| macro_combo | 1.2 | -0.025 | 2/5 | [-0.3, -1.57, -1.01, 0.4, 1.03] | 1.0 | 0.018863 |

Portfolio weights: `{'xgb': 0.0, 'lgb': 0.1202, 'cat': 0.1904, 'l4': 0.6894}`

Strict gate (5/5, disp∈[0.98,1.02], KS<0.002): **False**

## Steps onto St9.5

| 1-ρ e-4 | w | ΔOOF e-5 | pos | folds |
|---:|---:|---:|---:|---|
| 0.5 | 0.026492 | 1.18 | 1/5 | [-2.47, 0.27, -1.33, -2.23, -5.48] |
| 1.0 | 0.0375 | 0.63 | 0/5 | [-4.23, -0.01, -2.26, -3.66, -9.12] |
| 2.0 | 0.053081 | 0.0 | 0/5 | [-7.25, -0.99, -4.55, -6.14, -15.29] |
| 2.5 | 0.059365 | 0.0 | 0/5 | [-8.73, -1.45, -5.64, -7.33, -18.14] |
| 3.5 | 0.070266 | 0.0 | 0/5 | [-11.59, -2.64, -7.65, -9.56, -23.66] |

## Staged

- `{'file': '/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/preds/macro_v11/research_macro_v11_slotA_1p0e4.csv', 'research_only': True, 'w': 0.03750013234093785, 'target_1e4': 1.0}`
- `{'file': '/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/preds/macro_v11/research_macro_v11_slotB_2p5e4.csv', 'research_only': True, 'w': 0.05936457635834813, 'target_1e4': 2.5}`

`NO_SUBMIT=True`.
