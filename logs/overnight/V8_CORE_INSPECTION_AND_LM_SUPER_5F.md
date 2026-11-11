# V8 Core Inspection & LM-Super 5F / F0–F1 GBDT Audit

**NO_SUBMIT=True**. Scripts: `scripts/gm_inspect_v8_and_fix_f0f1_gbdt.py` + Slot3/4 bayes recompute. Elapsed ~114s (+ recompute).

## 0 — Quarantine `nullspace_v3`

Moved out of `submissions/`:

- `sub_strike10_nullspace_v3_0p15e4.csv`
- `sub_strike10_nullspace_v3_0p30e4.csv`

→ `submissions/_quarantine_nullspace_v3/` (confirmed absent from live `submissions/`).

Reason: `NULLSPACE_STRIKE_ORTH_V3.md` SetA (no `c13_v10_ridge`) = **−1.24e-5 / 2/5**; SetB **+4.14e-5** is 100% `c13_v10_ridge` = SoftPseudo/`zanch_*` (closed in `STEP3_STRIKE10.md`).

---

## 1 — Exact `v8` formula (`0.9525970` OOF / LB `0.95307`)

- **Script:** `scripts/build_v8_part12_rect_slot2.py`
- **Artifacts:** `preds/P_super_v8_part12_rect_{oof,test}.npy`
- **LB file:** `submissions/sub_v8_part12_fsl_prop_rect.csv`

**OOF:**

```
z_w12 = zrk(external/nov08_part12/wide_lm_stack_nested.npy)
R     = KS-rectified orth residuals of proprietary components vs z_w12
z_v8  = z_w12 + R @ w_fold     # nested 5F coordinate ascent
```

**Slot-2 test:**

```
z_p12_fsl   = zrk(0.68 * zrk(w12_t) + 0.32 * zrk(part9_t))
dz_orth_fsl = qmap( dz_test - β * z_p12_fsl , dz_oof )
sub         = rk( z_p12_fsl + 1.0 * dz_orth_fsl )
```

**KS-rect orth components vs Part12 (`R` columns):**

1. `P_super_v4p_hc`
2. `paradigm_B/xgb_llr`
3. `matrix_v2/P4_L4.3_pure5f`
4. `matrix_v2/P4_L4.2_pure5f`
5. `matrix_v2/P4_L4.5_pure5f`
6. `matrix_v2/P4_L4.3_ext`
7. `gpt2_3ep/gpt2_bidir`
8. `gpt2_3ep/xgb_gpt2_3ep_d2_d3_bag6` ← **F0/F1 corruption enters v8 here**

**Per-fold OOF AUC:** `[0.9518700, 0.9525373, 0.9573745, 0.9527644, 0.9484825]` → **0.9525970**

---

## 2 — Exact `z_ref_o` formula (`0.9526612` OOF)

- **Script:** `scripts/recon_and_forensic_audit.py::build_z_ref`

```
z_ref_o = z_v8_o
        + 0.025 * (r_xgb_fast − r_tabm)
        + 0.85  * (U5 @ [0.0135, 0.0115, −0.0085, −0.0085, −0.0105])
        + 0.85  * (U2 @ [0.0150, 0.0125])
```

- **U5** (orth vs `[z_v8, z_w12, r_xgb−r_tabm]`): `lab4_tabular_nade`, `sameershaikh27 nested_winner`, `orig20k_knn`, `harvest_p4_cuda_xgb`, `lab1_zjoint_graph_lgb4`
- **U2** (orth vs X7): `ladder/glm_quad`, `part11/lab1_glm_plus_graph`

**Per-fold OOF AUC:** `[0.9519569, 0.9525950, 0.9573680, 0.9528724, 0.9485101]` → **0.9526612**

---

## 3 — Per-fold OOF AUC bank

| model | OOF AUC | F0 | F1 | F2 | F3 | F4 |
|---|---:|---:|---:|---:|---:|---:|
| v8 | 0.9525970 | 0.9518700 | 0.9525373 | 0.9573745 | 0.9527644 | 0.9484825 |
| z_ref_o | 0.9526612 | 0.9519569 | 0.9525950 | 0.9573680 | 0.9528724 | 0.9485101 |
| xgb_gpt2_3ep | 0.9523089 | 0.9515038 | 0.9526136 | 0.9569483 | 0.9527395 | 0.9479815 |
| lgb_gpt2_3ep_fixed | 0.9523254 | 0.9515162 | 0.9525525 | 0.9569832 | 0.9527819 | 0.9480161 |
| gb_gpt2_bag5 | 0.9516766 | 0.9507388 | 0.9517337 | 0.9566607 | 0.9522136 | 0.9473523 |
| xgb_clean_decorrupt | 0.9510711 | 0.9494986 | 0.9515224 | 0.9563119 | 0.9513127 | 0.9467115 |
| xgb_fast_harvest | 0.9511435 | 0.9495162 | 0.9516343 | 0.9561821 | 0.9514802 | 0.9469115 |

### Complete 5-fold LM bank

| LM | OOF AUC | F0 | F1 | F2 | F3 | F4 | path |
|---|---:|---:|---:|---:|---:|---:|---|
| lm12_wide | 0.9525651 | 0.9517736 | 0.9525308 | 0.9573724 | 0.9526969 | 0.9484516 | `external/nov08_part12/wide_lm_stack_nested.npy` |
| lm12_raw | 0.9504283 | 0.9485056 | 0.9506818 | 0.9560850 | 0.9506043 | 0.9463809 | `external/nov08_part12/lm_oof.npy` |
| megayak | 0.9504283 | 0.9485056 | 0.9506818 | 0.9560850 | 0.9506043 | 0.9463809 | `external/banks/megayak_…/lm_oof.npy` (**≡ lm12_raw**) |
| masaya_gpt2 | 0.9500272 | 0.9488823 | 0.9503172 | 0.9553120 | 0.9498742 | 0.9458540 | `external/banks/masayakawamata_…/oof.npy` |
| gpt2_3ep | 0.9491505 | 0.9485056 | 0.9506818 | 0.9532775 | 0.9490265 | 0.9442636 | `preds/gpt2_3ep/gpt2_3ep_oof.npy` |
| lm_gpt2_3ep | 0.9491334 | 0.9485056 | 0.9506818 | 0.9532775 | 0.9490265 | 0.9442636 | `preds/lm_gpt2_3ep/lm_oof_5f.npy` |
| gpt2_rev | 0.9481826 | 0.9462519 | 0.9490109 | 0.9533194 | 0.9485382 | 0.9438955 | `preds/gpt2_3ep/gpt2_rev_2ep_fixed_oof.npy` |
| gpt2_med_4ep_5f (mean-LLR) | 0.9482383 | 0.9458300 | 0.9487682 | 0.9537358 | 0.9486671 | 0.9442065 | `preds/gpt2_med_4ep/oof_{0..4}.npy` |
| gpt2_med_4ep_5f (bayes_fold) | — | 0.9458179 | 0.9487187 | 0.9537252 | 0.9485707 | 0.9440638 | Slot3/4 `logaddexp` reduction |

---

## 4 — `gpt2_3ep` F0/F1 corruption (exact)

Corruption is vs **`lm12_raw` / `megayak`**, not vs `lm12_wide`:

| fold | pearson(`lm12_wide`) | pearson(`lm12_raw`) | F0/F1 AUC identity |
|---:|---:|---:|---|
| 0 | 0.981421 | **0.999911** | `gpt2_3ep` F0 AUC = `lm12_raw` F0 = **0.9485056** |
| 1 | 0.979858 | **0.999987** | `gpt2_3ep` F1 AUC = `lm12_raw` F1 = **0.9506818** |
| 2 | 0.962202 | 0.975305 | distinct |
| 3 | 0.960057 | 0.974945 | distinct |
| 4 | 0.957156 | 0.972446 | distinct |

Also: `lm_gpt2_3ep` F0/F1 AUCs match `gpt2_3ep` exactly (same poisoned folds). `megayak ≡ lm12_raw` (identical fold AUCs).

---

## 5 — 5-Fold `lm_super` law (reproduced Slot3/4 protocol)

### Protocol that produced the historical lifts

```
z_m[f]  = zrk( bayes_fold(oof_f, prior_f) )     # logaddexp orders + logit(prior)
r_n[f]  = unitize( z_m[f] − β_f * z_lm[f] )     # β = cov/var within fold
z_super = z_lm + w * r_n
```

(`bayes_fold` = `logaddexp(logp1) − logaddexp(logp0) + log(prior/(1−prior))` from `verify_slot4_gm_f0123.py` / `de_corrupt_and_boost_5f.py`.)

### A. vs `lm12_raw` (historical base — matches +9.27 / +25 / +9.26 story)

| fold | best w | best Δ×1e-5 | **w=+0.070 Δ×1e-5** |
|---:|---:|---:|---:|
| F0 | +0.0645 | +9.23 | **+9.14** |
| F1 | +0.1100 | +26.99 | **+23.55** |
| F2 | +0.0700 | +9.10 | **+9.10** |
| **F3** | +0.0850 | +16.64 | **+16.16** |
| **F4** | +0.0820 | +12.62 | **+12.53** |

**All 5 folds positive at `w≈+0.07` vs `lm12_raw`.** Combined story: F0–F2 match the remembered `+9.27 / +25.07 / +9.26`; F3/F4 add **+16.2 / +12.5**.

### B. vs `lm12_wide` (the actual v8 backbone — critical)

| fold | best w | best Δ×1e-5 | **w=+0.070 Δ×1e-5** |
|---:|---:|---:|---:|
| F0 | −0.050 | +12.04 | **−30.89** |
| F1 | +0.022 | +1.10 | **−2.39** |
| F2 | −0.005 | +0.03 | **−7.57** |
| F3 | −0.006 | +0.21 | **−10.92** |
| F4 | −0.009 | +0.34 | **−11.63** |

**`w=+0.07` on `lm12_wide` is toxic.** Nested LOFO prefers `w≈0` / slightly negative; mean-LLR residual (non-bayes) same qualitative result.

### C. Pure-LM consensus `{lm12_wide, masaya, gpt2_rev, lm_gpt2_3ep}` + `0.07 * r_gpt2m`

- Consensus OOF **0.9512018** → +0.07 residual **0.9510047**
- Δ×1e-5 by fold: **F0=−39.8, F1=−18.8, F2=−8.7, F3=−14.2, F4=−16.7** (all negative)

**Verdict:** The historical `lm_super` lift is a **`lm12_raw`-space** phenomenon (same family as the F0/F1 corruption). It does **not** transfer as a positive `+0.07` blend onto `lm12_wide` / pure-LM consensus. gpt2-medium signal for a clean Stage-2 should enter as a **feature / orth residual column**, not as a fixed `+0.07` add-on to the Part12 wide stack.

---

## 6 — `xgb_gpt2_3ep` Stage-2 recipe & clean replacement status

### Training lineage

| item | value |
|---|---|
| Script | `scripts/post_gpt2_3ep_campaign.py::stage_b` |
| Artifact | `preds/gpt2_3ep/xgb_gpt2_3ep_d2_d3_bag6_{oof,test}.npy` |
| Runtime log | `logs/overnight/post_gpt2_3ep_campaign.nohup.log` |
| `gboost` import | `GEN=False` **KMAT=(999944, 68)** **BASE=(999944, 41)** |
| **X_fusion shape** | **o=(699960, 132) t=(299984, 132)** |
| Shape decomposition | `68` KMAT TE (`min(108,68)`) + `7` extras + `16` h16 + `41` BASE = **132** |
| Extras | `z_gpt2_3ep`, `z_gpt−z_v4p`, `z_gpt*r_rare`, `z_gpt*r_disp`, `zrk(P4_L4.3/L4.2/L4.5_pure5f)` |
| `base_margin` | Platt-calibrated `zrk(P_super_v5_harvest)` via `calibrate_logit_to_prevalence` |
| Bag | 3×`max_depth=2` + 3×`max_depth=3` (seeds 42..44, 100..102) |
| Hyperparams | `eta=0.025`, `subsample=0.8`, `colsample_bynode=0.35`, `reg_lambda=30`, `min_child_weight=200`, `num_boost_round=400`, `early_stopping=40`, `tree_method=hist`, `device=cuda`, `objective=binary:logistic` |
| Solo OOF | **0.9523089** folds `[0.9515038, 0.9526136, 0.9569483, 0.9527395, 0.9479815]` |
| Corruption path | `z_gpt2_3ep` feature ≡ `lm12_raw` on F0/F1 → poisoned Stage-2 inputs |

### Existing clean / candidate artifacts

| candidate | on disk? | note | solo AUC |
|---|---|---|---:|
| `preds/gpt2_3ep/xgb_gpt2_3ep_d2_d3_bag6_oof.npy` | YES | original corrupted-input booster | 0.9523089 |
| `preds/fast_gpu_harvest/oof_xgb_clean.npy` | YES | `de_corrupt_and_boost_5f.py` Step1: **LGB** `max_depth=4, num_leaves=12`, 3 seeds; features = clean LM block (`lm12/masa/gpt2_rev`) + TabM±std + knn24 + KMAT top-30 TE; **NO** gpt2_3ep, **NO** gpt2-medium, **NO** P4 pure5f cols, **NO** Stage-B `base_margin` | 0.9510711 |
| `xgb_lm_super_clean_5f` | **NO** | proposed homogeneous Stage-B replacement | — |

### Proposed clean Stage-2 replacement (`xgb_lm_super_clean_5f`) — recipe to run later

Reuse Stage-B exactly, with these substitutions:

```
X = [
  KMAT_TE(68),                          # same gboost.KMAT EB-TE
  z_gpt2m_5f_bayes,                     # REPLACE corrupted z_gpt2_3ep
  z_gpt2m − z_v4p,                      # interactions vs clean LM residual
  z_gpt2m * r_rare, z_gpt2m * r_disp,
  zrk(P4_L4.3_pure5f), zrk(P4_L4.2_pure5f), zrk(P4_L4.5_pure5f),
  z_tabm_5f, s_tabm_5f,                 # ADD TabM_5f (from harvest)
  optional: drop PLS h16 (was fit on corrupted gpt2_3ep),
  gboost.BASE (41),
]
base_margin = Platt(zrk(clean Stage-1))   # v8 or z_ref_o, NOT v5-with-poisoned-gpt2 path
XGB bag6    = identical hyperparams (depth 2/3, eta 0.025, …, hist/cuda)
```

Expected shape ≈ `(699960, 68+7+2+41)` = **118** without h16, or **134** with TabM±std replacing h16.

`de_corrupt` LGB clean exists but is **not** a drop-in Stage-B replacement (different model family, no P4/gpt2m/base_margin).

---

## 7 — Bottom line

1. Nullspace v3 CSVs quarantined.
2. `v8` / `z_ref_o` formulas and per-fold AUCs locked.
3. `gpt2_3ep` F0/F1 = `lm12_raw` (pearson ≥ 0.9999); that poison is inside `v8` via the `xgb_gpt2_3ep` residual column.
4. Full-5F bayes `lm_super` vs `lm12_raw`: **F0..F4 = +9.14, +23.55, +9.10, +16.16, +12.53** (×1e-5) at `w=+0.07`. Same law vs `lm12_wide` / pure-LM bank at `w=+0.07` is **negative**.
5. No `xgb_lm_super_clean_5f` on disk; closest is LGB `oof_xgb_clean.npy`. Stage-B exact shape was **132 = 68 TE + 7 + 16 h16 + 41 BASE**.

`NO_SUBMIT=True`.
