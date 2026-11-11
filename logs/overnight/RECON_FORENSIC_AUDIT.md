# Recon & Forensic Audit (ZERO submits)

Elapsed **1074.8s**. St9-proxy OOF AUC **0.9526612**.

## Part A — SoftFamily autopsy (why P2 → 0.95306)

St9 has no pure nested OOF; z_ref_o (≈0.9526612) is the St9-stack OOF proxy. Test anchor = PCHIP(St9 submission → z_ref_o).

| component | ΔOOF e-5 | w | pos | disp stdT/stdO | KS | 1-ρ(St9) e-4 | gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| soft_diff | 15.76 | 0.08 | 4/5 | 1.0034 | 0.001877 | 10396.393 | PASS |
| idf_p3 | 1.7 | 0.025 | 3/5 | 1.0003 | 0.002469 | 7558.898 | fail |
| gb_soft_bag3 | 4.88 | 0.05 | 4/5 | 0.9554 | 0.010092 | 123.917 | PASS |
| clean035 | 1.66 | 0.03 | 3/5 | 1.0076 | 0.009874 | 179.152 | fail |
| SoftFamily_combined | 2.56 | 0.08 | 5/5 | 0.9682 | 0.01167 | 14.22 | PASS |

**Reading:** `dispersion_ratio_raw << 1` or `>> 1` ⇒ train/test residual scale mismatch; `KS` large ⇒ covariate shift in residual. Combined SoftFamily had strong nested OOF but public P2×0.85 crashed **−0.00024** — do not trust OOF alone when disp/KS are bad.

## Part B — Subodh 38 features

**Count:** 38 (excludes id/fold/is_fraud)

```
amount_usd
hours_since_last_txn
txn_count_last_24h
distance_from_home_km
card_age_months
customer_age
account_balance_usd
cvv_retry_count
velocity_score
time_of_day_hour
day_of_week
merchant_risk_score
prior_disputes
merchant_category
card_type
auth_method
channel
device_type
is_foreign_transaction
is_new_merchant
used_vpn
ip_country_mismatch
billing_shipping_mismatch
is_ai_generated_scam_attempt
risk_flag_count
vpn_and_mismatch
foreign_and_new_merchant
amount_cents
amount_is_round
amount_is_round_10
is_night
is_weekend
amount_usd_dec
hours_since_last_txn_dec
velocity_score_dec
distance_from_home_km_dec
hours_te
amount_bin200_te
```

## Part B — Flagged OOF/Test pairs (strict gate)

Gate: Δ≥0.5e-5, ≥4/5, 0.85≤disp≤1.15, KS<0.015

_None passed the full gate._

## Near-misses (OOF ok, shift fails gate)

| component | ΔOOF e-5 | w | pos | disp stdT/stdO | KS | 1-ρ(St9) e-4 | gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| sk20220623:oof_xgb_exp019 | 1.73 | -0.03 | 4/5 | 0.915 | 0.026271 | 904.44 | fail |

JSON: `logs/overnight/recon_and_forensic_audit.json`