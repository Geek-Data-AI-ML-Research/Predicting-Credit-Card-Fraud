# Forensic Covariance Fix & St9.5 Orthogonalization

**NO_SUBMIT=True**. Elapsed **50.4s**. Did **not** launch v12.

## Part 1 — Why OOF winners conflicted with Strike 6→9.5

Strike ladder basis (test rank-logit):
- $\vec{e}_1 = z_{st4}-z_{st3}$ (positive LB slope)
- $\vec{e}_2 = v_{bad\_new5}$ (negative / tree dilution)
- $\vec{e}_3 = u_{bad5}$ (same construction; removed by St9/9.5)
- $\vec{d}_{st9p5} = z_{st9p5}-z_{v8}$

| residual | cos e1 | cos e2 | cos e3 | cos d_st9p5 |
|---|---:|---:|---:|---:|
| soft_diff_orth_LM | -0.004 | +0.066 | +0.066 | -0.052 |
| v10_ridge_resid | +0.026 | +0.002 | +0.002 | +0.056 |
| v10_cb_d4_bag3 | +0.010 | +0.003 | +0.003 | +0.016 |
| harvest_p4_cuda_xgb | -0.203 | +0.003 | +0.003 | -0.138 |
| B_xgb_llr | -0.213 | +0.462 | +0.462 | -0.516 |
| macro_delta_xgb | -0.014 | +0.054 | +0.054 | -0.010 |
| macro_delta_lgb | -0.033 | +0.059 | +0.059 | -0.025 |
| macro_delta_cat | -0.006 | +0.011 | +0.011 | -0.011 |
| macro_delta_l4 | +0.011 | +0.011 | +0.011 | -0.037 |
| macro_r_macro_orth | +0.021 | +0.024 | +0.024 | -0.023 |

### Live negative probes vs ladder

| probe | cos e1 | cos e2 | cos e3 | cos d_st9p5 |
|---|---:|---:|---:|---:|
| Slot1_softdiff | -0.004 | +0.065 | +0.065 | -0.052 |
| ANTI | +0.083 | -0.155 | -0.155 | +0.136 |

### Energy remaining after orth to span(St9.5, e1, e2, e3, d_bad_leader)

```
{
  "soft_diff_orth_LM": {
    "var_frac_remaining": 0.9938,
    "std_remaining": 1.008769,
    "std_orig": 1.011919,
    "cos_e1_after": -0.0,
    "cos_e2_after": 0.0,
    "cos_e3_after": 0.0,
    "cos_dst9p5_after": -0.0223
  },
  "v10_ridge_resid": {
    "var_frac_remaining": 0.9969,
    "std_remaining": 0.008194,
    "std_orig": 0.008207,
    "cos_e1_after": 0.0,
    "cos_e2_after": 0.0,
    "cos_e3_after": 0.0,
    "cos_dst9p5_after": 0.032
  }
}
```

## Part 2 — Pairwise Difference Covariance Shrinkage

Across **21** LM/neural pairs with individually PCHIP-aligned marginals, **7** pairs show `disp_diff < 0.85` — proving 5-fold test averaging inflates `corr_test` vs `corr_oof` and shrinks difference std.

Worst shrinkers:

| pair | corr_oof | corr_test | disp_diff | KS_diff |
|---|---:|---:|---:|---:|
| gpt2_3ep - gpt2_rev | 0.982 | 0.991 | 0.705 | 0.06345 |
| gpt2_3ep - gpt2_med_4ep_5f | 0.977 | 0.987 | 0.759 | 0.06231 |
| masaya_gpt2 - gpt2_3ep | 0.976 | 0.985 | 0.795 | 0.05308 |
| gpt2_3ep - TabM_5f | 0.869 | 0.917 | 0.797 | 0.08066 |
| gpt2_rev - TabM_5f | 0.880 | 0.922 | 0.805 | 0.07202 |
| gpt2_med_4ep_5f - TabM_5f | 0.865 | 0.909 | 0.822 | 0.05975 |
| masaya_gpt2 - TabM_5f | 0.866 | 0.904 | 0.845 | 0.05782 |
| TabM_5f - z_ref_o | 0.859 | 0.896 | 0.859 | 0.04705 |

### gpt2_med − lm12: Method A vs B

```
{
  "A_naive_5f_avg": {
    "disp": 1.0271,
    "ks": 0.014461,
    "beta": 0.9635506751948605
  },
  "B_post_diff_pchip": {
    "disp": 1.0,
    "ks": 0.000926
  },
  "Bpf_perfold_matched_post_pchip": {
    "disp": 1.0,
    "ks": 0.001619
  },
  "A_vs_zref": {
    "disp": 1.0077,
    "ks": 0.003347
  },
  "B_vs_zref": {
    "disp": 1.0,
    "ks": 0.001052
  }
}
```

Method B / Bpf restores `disp≈1.000` by PCHIP-aligning the **contrast** after construction (and optionally matching per-fold `test_f`).

## Part 3 — Covariance-corrected Ridge / LogReg on `z_ref_o`

D_clean: **23** cols, hygiene_ok **0/23** (target disp∈[0.999,1.001], KS<0.0005).

| model | ΔOOF e-5 | pos | folds | disp | KS | cos(e1,e2,e3) |
|---|---:|---:|---|---:|---:|---|
| ridge | +4.96 | 5/5 | [3.0, 3.35, 5.8, 6.21, 7.6] | 1.0000 | 0.009634 | (+0.023,+0.027,+0.027) |
| logreg_newton | -27.59 | 0/5 | [-55.89, -35.45, -15.34, -13.16, -4.27] | 1.0000 | 0.000889 | (+0.014,+0.004,+0.004) |

Feature hygiene (all):

| col | disp | KS | ok |
|---|---:|---:|:---:|
| d_gpt2m_lm12 | 1.000000 | 0.00161937 | N |
| d_gpt2m_perm_prec | 1.000000 | 0.00156758 | N |
| d_xgbf_tabm | 1.000000 | 0.00136187 | N |
| d_P4_L4.3_pure5f | 1.000000 | 0.03602910 | N |
| d_P4_L4.2_pure5f | 1.000000 | 0.00147936 | N |
| d_P4_L4.5_pure5f | 1.000000 | 0.00527922 | N |
| d_v10_ridge | 1.000000 | 0.00319424 | N |
| d_soft_orth | 1.000000 | 0.00064548 | N |
| r_rare | 0.000000 | 0.00000000 | N |
| r_disp | 1.000000 | 0.06997039 | N |
| bound_unc | 1.000000 | 0.00052336 | N |
| d_gpt2m_lm12_x_rare | 0.000000 | 0.00000000 | N |
| d_gpt2m_lm12_x_disp | 1.000000 | 0.00702186 | N |
| d_gpt2m_lm12_x_bnd | 1.000000 | 0.09671731 | N |
| d_v10_ridge_x_rare | 0.000000 | 0.00000000 | N |
| d_v10_ridge_x_disp | 1.000000 | 0.03949875 | N |
| d_v10_ridge_x_bnd | 1.000000 | 0.39437171 | N |
| d_soft_orth_x_rare | 0.000000 | 0.00000000 | N |
| d_soft_orth_x_disp | 1.000000 | 0.00459596 | N |
| d_soft_orth_x_bnd | 1.000000 | 0.05988811 | N |
| d_xgbf_tabm_x_rare | 0.000000 | 0.00000000 | N |
| d_xgbf_tabm_x_disp | 1.000000 | 0.00757665 | N |
| d_xgbf_tabm_x_bnd | 1.000000 | 0.07227342 | N |

`NO_SUBMIT=True`.
