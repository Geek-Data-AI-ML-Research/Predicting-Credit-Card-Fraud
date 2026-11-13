# GM V13.4.1 HARDENED — d1_v15_tau1.00 Forensic Audit

**STRICT NO_SUBMIT=True** · POD_TOUCHES=0 · elapsed 707.3s

**Script:** `scripts/gm_v13_4_1_hardened_d1_v15_forensic_audit.py`

## Step 1 — Pre-PCHIP raw dispersion & pre-orth collinearity

| family | raw_disp | in [0.92,1.08] | ρ(d1) oof/test | |Δρ(d1)| | ρ(z) oof/test | |Δρ(z)| |
|---|---:|---|---|---:|---|---:|
| `lgb` | 1.0123 | True | 0.067/-0.022 | 0.089 | -0.999/-1.000 | 0.000 |
| `xgb` | 0.9219 | True | 0.036/0.075 | 0.040 | -0.205/-0.200 | 0.005 |
| `cat` | 1.0184 | True | 0.087/-0.008 | 0.095 | -0.985/-0.985 | 0.000 |
| `mean` | 1.0147 | True | 0.067/-0.016 | 0.083 | -0.998/-0.998 | 0.000 |

**Verdict:** raw_disp OK=True · OOF/Test corr match (±0.03) OK=False

## Step 2 — Split-gain by group + top features

| group | % total gain |
|---|---:|
| `KMAT_68` | 43.03% |
| `P4` | 5.22% |
| `CONTRASTS` | 43.77% |
| `q_ATH_bin` | 7.98% |

### Top 20 features by gain

| # | feature | gain | lgb | xgb | cat |
|---:|---|---:|---:|---:|---:|
| 1 | `MS-LM` | 6260.6 | 6183.4 | 77.2 | 0.0 |
| 2 | `q_ATH_bin` | 3445.0 | 3129.4 | 47.7 | 267.8 |
| 3 | `denoise(G26-MS)` | 2859.5 | 2802.8 | 56.6 | 0.0 |
| 4 | `sadam_v4_cat-LM` | 1675.3 | 1634.3 | 41.0 | 0.0 |
| 5 | `KMAT_6` | 1199.7 | 1164.9 | 34.9 | 0.0 |
| 6 | `KMAT_11` | 1142.7 | 1103.9 | 38.5 | 0.3 |
| 7 | `GPD-MLP` | 1050.9 | 1015.4 | 35.1 | 0.4 |
| 8 | `P4_L4.5` | 1021.0 | 983.2 | 37.7 | 0.1 |
| 9 | `P4_L4.3` | 988.9 | 950.7 | 38.2 | 0.0 |
| 10 | `std_LM` | 972.7 | 936.3 | 36.1 | 0.4 |
| 11 | `Gmed-MS` | 970.1 | 931.9 | 38.2 | 0.0 |
| 12 | `gap_Gen_vs_Disc` | 916.9 | 877.8 | 38.8 | 0.2 |
| 13 | `KMAT_39` | 823.5 | 785.5 | 38.0 | 0.0 |
| 14 | `KMAT_47` | 821.6 | 783.3 | 37.8 | 0.5 |
| 15 | `GPD-GB` | 781.8 | 746.7 | 34.9 | 0.2 |
| 16 | `KMAT_49` | 777.3 | 743.3 | 33.7 | 0.2 |
| 17 | `KMAT_5` | 615.2 | 579.0 | 33.6 | 2.5 |
| 18 | `KMAT_9` | 587.3 | 557.1 | 30.3 | 0.0 |
| 19 | `megayak_shallow_tree-GB` | 542.3 | 506.2 | 35.6 | 0.5 |
| 20 | `KMAT_19` | 536.4 | 504.6 | 29.5 | 2.3 |

### Top-15 feature corr symmetry (|Δ|>0.04 flagged)

| feature | corr_oof | corr_test | |Δ| | shift |
|---|---:|---:|---:|---|
| `MS-LM` | 0.302 | 0.298 | 0.004 | False |
| `q_ATH_bin` | -0.152 | -0.168 | 0.016 | False |
| `denoise(G26-MS)` | -0.241 | -0.255 | 0.014 | False |
| `sadam_v4_cat-LM` | 0.127 | 0.047 | 0.080 | True |
| `KMAT_6` | -0.006 | -0.003 | 0.003 | False |
| `KMAT_11` | -0.051 | 0.006 | 0.056 | True |
| `GPD-MLP` | 0.126 | 0.188 | 0.062 | True |
| `P4_L4.5` | -0.098 | -0.101 | 0.002 | False |
| `P4_L4.3` | -0.083 | -0.009 | 0.074 | True |
| `std_LM` | 0.176 | 0.184 | 0.009 | False |
| `Gmed-MS` | -0.127 | -0.114 | 0.012 | False |
| `gap_Gen_vs_Disc` | 0.063 | 0.133 | 0.070 | True |
| `KMAT_39` | -0.016 | -0.004 | 0.012 | False |
| `KMAT_47` | -0.051 | 0.006 | 0.056 | True |
| `GPD-GB` | 0.127 | 0.185 | 0.059 | True |

### Ablation (mean family, w∈{0.02,0.03})

| abl | w | main | min | sf25 | t | pass123 | raw_disp |
|---|---:|---:|---:|---:|---:|---|---:|
| `KMAT_only` | 0.02 | 1/5 | -6.73 | 10/25 | -0.91 | False | 1.0176 |
| `KMAT_only` | 0.03 | 1/5 | -10.95 | 9/25 | -1.38 | False | 1.0176 |
| `CONTRASTS_only` | 0.02 | 5/5 | 0.61 | 20/25 | 3.08 | True | 1.0117 |
| `CONTRASTS_only` | 0.03 | 5/5 | 0.73 | 20/25 | 2.78 | False | 1.0117 |
| `FULL` | 0.02 | 3/5 | -1.58 | 14/25 | 0.43 | False | 1.0147 |
| `FULL` | 0.03 | 3/5 | -2.55 | 15/25 | 0.41 | False | 1.0147 |

## Step 3 — Per-booster family audit

| family | w | main | meanΔ | min | folds | sf25 | t | W-p | boot90 | raw_disp | pass123 |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| `LGB_only` | 0.015 | 5/5 | 3.86 | 1.69 | [7.482, 4.119, 3.712, 2.281, 1.691] | 19/25 | 3.85 | 0.000514 | [2.53, 5.384] | 1.0123 | True |
| `LGB_only` | 0.02 | 5/5 | 4.93 | 1.89 | [9.794, 5.24, 4.793, 2.928, 1.891] | 19/25 | 3.66 | 0.000726 | [3.141, 6.976] | 1.0123 | True |
| `LGB_only` | 0.025 | 5/5 | 5.85 | 1.87 | [12.049, 6.08, 5.773, 3.461, 1.871] | 19/25 | 3.55 | 0.00125 | [3.555, 8.41] | 1.0123 | True |
| `LGB_only` | 0.03 | 5/5 | 6.69 | 1.88 | [14.202, 6.784, 6.661, 3.945, 1.875] | 19/25 | 3.47 | 0.00113 | [3.839, 9.728] | 1.0123 | True |
| `XGB_only` | 0.015 | 5/5 | 4.16 | 1.66 | [8.038, 4.755, 4.386, 1.662, 1.973] | 18/25 | 3.58 | 0.000813 | [2.456, 5.994] | 0.9219 | True |
| `XGB_only` | 0.02 | 5/5 | 5.25 | 2.01 | [10.477, 5.999, 5.608, 2.007, 2.136] | 18/25 | 3.40 | 0.00171 | [2.857, 7.712] | 0.9219 | True |
| `XGB_only` | 0.025 | 5/5 | 6.26 | 2.21 | [12.743, 7.064, 6.94, 2.342, 2.214] | 18/25 | 3.31 | 0.00279 | [3.261, 9.311] | 0.9219 | True |
| `XGB_only` | 0.03 | 5/5 | 7.11 | 2.07 | [14.968, 8.017, 8.025, 2.462, 2.065] | 16/25 | 3.13 | 0.00482 | [3.495, 10.801] | 0.9219 | False |
| `CAT_only` | 0.015 | 1/5 | -0.71 | -1.83 | [-0.685, 0.535, -0.768, -0.812, -1.83] | 6/25 | -2.39 | 0.988 | [-1.194, -0.199] | 1.0184 | False |
| `CAT_only` | 0.02 | 1/5 | -0.99 | -2.51 | [-0.938, 0.658, -1.029, -1.133, -2.509] | 7/25 | -2.22 | 0.984 | [-1.642, -0.313] | 1.0184 | False |
| `CAT_only` | 0.025 | 1/5 | -1.27 | -3.05 | [-1.189, 0.795, -1.335, -1.59, -3.049] | 8/25 | -2.26 | 0.986 | [-2.072, -0.4] | 1.0184 | False |
| `CAT_only` | 0.03 | 1/5 | -1.59 | -3.81 | [-1.459, 0.821, -1.622, -1.904, -3.809] | 8/25 | -2.23 | 0.982 | [-2.553, -0.58] | 1.0184 | False |
| `LGB+CAT` | 0.015 | 2/5 | -0.11 | -1.46 | [0.415, 1.165, -0.276, -0.382, -1.462] | 14/25 | -0.81 | 0.674 | [-0.72, 0.556] | 1.0151 | False |
| `LGB+CAT` | 0.02 | 2/5 | -0.20 | -1.98 | [0.455, 1.504, -0.304, -0.678, -1.981] | 12/25 | -0.66 | 0.729 | [-1.024, 0.649] | 1.0151 | False |
| `LGB+CAT` | 0.025 | 2/5 | -0.30 | -2.58 | [0.651, 1.748, -0.432, -0.892, -2.583] | 13/25 | -0.85 | 0.771 | [-1.379, 0.782] | 1.0151 | False |
| `LGB+CAT` | 0.03 | 2/5 | -0.47 | -3.21 | [0.649, 1.929, -0.506, -1.215, -3.215] | 12/25 | -0.88 | 0.788 | [-1.731, 0.79] | 1.0151 | False |
| `LGB+XGB` | 0.015 | 5/5 | 4.20 | 1.97 | [8.131, 4.613, 4.19, 2.091, 1.969] | 19/25 | 3.68 | 0.000648 | [2.57, 5.936] | 1.0116 | True |
| `LGB+XGB` | 0.02 | 5/5 | 5.32 | 2.06 | [10.618, 5.882, 5.448, 2.596, 2.063] | 19/25 | 3.58 | 0.000908 | [3.147, 7.689] | 1.0116 | True |
| `LGB+XGB` | 0.025 | 5/5 | 6.25 | 2.03 | [12.813, 6.765, 6.612, 3.025, 2.035] | 18/25 | 3.45 | 0.00154 | [3.575, 9.153] | 1.0116 | True |
| `LGB+XGB` | 0.03 | 5/5 | 7.15 | 2.04 | [15.114, 7.547, 7.532, 3.502, 2.036] | 18/25 | 3.36 | 0.00209 | [4.018, 10.571] | 1.0116 | True |
| `LGB+XGB+CAT` | 0.015 | 3/5 | 0.41 | -1.06 | [1.287, 1.737, 0.281, -0.208, -1.055] | 15/25 | 0.42 | 0.198 | [-0.327, 1.168] | 1.0147 | False |
| `LGB+XGB+CAT` | 0.02 | 3/5 | 0.48 | -1.58 | [1.754, 2.288, 0.353, -0.403, -1.576] | 14/25 | 0.43 | 0.229 | [-0.569, 1.536] | 1.0147 | False |
| `LGB+XGB+CAT` | 0.025 | 3/5 | 0.52 | -2.08 | [2.066, 2.668, 0.391, -0.464, -2.076] | 15/25 | 0.45 | 0.205 | [-0.767, 1.801] | 1.0147 | False |
| `LGB+XGB+CAT` | 0.03 | 3/5 | 0.56 | -2.55 | [2.469, 3.005, 0.522, -0.64, -2.554] | 15/25 | 0.41 | 0.198 | [-0.941, 2.062] | 1.0147 | False |

Independent weak-bar (≥4/5 & ≥16/25): `{'LGB_only': True, 'XGB_only': True, 'CAT_only': False}`
Best staging family: **`LGB_only`**

## Step 4 — Trust-region vs LIVE 0.95388

| w | 1−ρ (e-4) | max\|Δrank\| top5% | main | min | sf25 | t | pass123 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.015 | 0.439 | 657.0 | 5/5 | 1.69 | 19/25 | 3.85 | True |
| 0.020 | 0.781 | 896.0 | 5/5 | 1.89 | 19/25 | 3.66 | True |
| 0.025 | 1.220 | 1144.0 | 5/5 | 1.87 | 19/25 | 3.55 | True |
| 0.030 | 1.756 | 1396.0 | 5/5 | 1.88 | 19/25 | 3.47 | True |

### Staged (NOSUBMIT)

- `sub_v13p4_verified_d1_v15_tau1.00_w0.020_NOSUBMIT.csv` · w=0.020 · 1−ρ=0.781e-4 · pass123=True
- `sub_v13p4_verified_d1_v15_tau1.00_w0.025_NOSUBMIT.csv` · w=0.025 · 1−ρ=1.220e-4 · pass123=True
- `sub_v13p4_verified_d1_v15_tau1.00_w0.030_NOSUBMIT.csv` · w=0.030 · 1−ρ=1.756e-4 · pass123=True

## Bottom line

1. **Pre-PCHIP `raw_disp` PASS** (mean=1.015; all families in [0.92, 1.08]) — trees are not obviously over-dispersed on OOF vs Test.
2. **Pre-orth collinearity FAIL (critical):** LGB/CAT/mean have **ρ(raw_resid, z_ATH) ≈ −0.999** before `lofo_orth`. Boosters are mostly undoing/rescaling the Platt init, not carving orthogonal fraud pockets. `lofo_orth` then removes that axis — which is why post-orth bars can look strong even when the raw residual is an anti-`z_ATH` ghost. Also `|Δρ(d1)|` OOF vs Test is **0.08–0.10** (above ±0.03) for LGB/CAT/mean.
3. **Ablation:** `CONTRASTS_only` clears Bars 1–3 at w=0.02 (20/25, t=3.08); **`KMAT_only` dies** (1/5). Gain split ≈ 43% KMAT / 44% CONTRASTS / 8% `q_ATH_bin` / 5% P4 — but KMAT alone does not transfer. 7/15 top features show `|Δcorr|>0.04` OOF↔Test.
4. **Per-family:** `LGB_only` and `LGB+XGB` (and `XGB_only` at w≤0.025) **clear Bars 1–3**. **`CAT_only` is toxic** (1/5) and **poisons the 3-family mean** — `LGB+XGB+CAT` fails 3/5 here (v13.4’s reported 3-family pass is **not** robust once families are split). Best staging family = **`LGB_only`**.
5. Staged NOSUBMIT (LGB_only orth): w=0.020 / 0.025 / 0.030 with 1−ρ(LIVE) ≈ 0.78 / 1.22 / 1.76 e-4. **Still NO_SUBMIT — human go required; do not treat as live-ready until the ρ≈−1 init-undo pathology is accepted or redesigned.**
