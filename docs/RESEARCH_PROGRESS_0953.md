# 8-Paradigm x 5-Lever Autonomous Research Campaign (`STRICT NO_SUBMIT`)

- **Execution Mode:** Local CPU only (`NO_GPU_YET`), single-process low-memory footprint (`n_jobs=2`, `float32`, explicit `gc.collect()`), continuous RAM monitoring (`rss_mb` + `host_avail_mb` logged on every step and every 5-minute watchdog tick).
- **Hard Guardrail:** `STRICT NO_SUBMIT` (zero Kaggle submissions).
- **Step 0 Baseline Locked:**
  - `P_super_v4p` (81 cols + 15 L4 regime): Nested OOF = **`0.9522827`**, Fold-0 = **`0.9513610`**
  - `Lever 8.1` (`P_super_v4p_hc`: `+12% Nested Hill-Climb + 4% GB_Super` + Multi-Seed verified on `{42, 7, 2024}`): Nested OOF = **`0.9523071`** (`+2.44e-5`), Fold-0 = **`0.9513936`** (`+3.26e-5`), `|Δρ| = 0.00005` -> **PROMOTED (Step 0)**

## Live Lever Execution Log

| UTC Timestamp | Paradigm | Lever | Hypothesis | RSS / Avail RAM (MB) | Fold-0 Canary ΔAUC | 5-Fold Nested ΔAUC | 90% Bootstrap CI | `|Δρ|` | Decision | Running Champ OOF |
|---|---|---|---|---|---|---|---|---|---|---|
| `2026-11-06T20:10:00Z` | `P8` | `L8.1` | Step-0 Calibration + 12% Hill + 4% GB_Super | `640 / 7100` | `+3.26e-5` | `+2.44e-5` | `[+1.4e-5, +3.5e-5]` | `0.00005` | **PROMOTED** | **`0.9523071`** |
| `2026-11-06T20:25:47Z` | `P4` | `L4.1` | Row-ID Rolling Neighbor Residual Windows | `1935.1 / 5916.3` | `+3.07e-05` | `+0.00e+00` | `[+0.00e+00, +0.00e+00]` | `0.00041` | **STRIKE_1_5FOLD_REJECT** | **`0.9523071`** |
| `2026-11-06T20:25:53Z` | `P4` | `L4.2` | Row-ID Modulo Prompt-Batch Periodicity | `2093.6 / 5751.2` | `+0.00e+00` | — | — | — | **STRIKE_2_DEAD** | **`0.9523071`** |
| `2026-11-06T20:28:45Z` | `P1` | `L1.2` | GLM-Importance-Weighted Full-Row Prototype kNN | `1446.1 / 5916.0` | `+1.79e-06` | — | — | — | **STRIKE_2_DEAD** | **`0.9523071`** |
| `2026-11-06T20:30:45Z` | `P1` | `L1.5` | Orig20k Residual Error Transfer | `718.0 / 4691.2` | `+0.00e+00` | — | — | — | **STRIKE_5_DEAD** | **`0.9523071`** |
| `2026-11-06T20:30:56Z` | `P2` | `L2.1` | Out-of-Fold Key-Residual Propagation (smooth=15) | `1775.5 / 4528.7` | `+0.00e+00` | — | — | — | **STRIKE_1_DEAD** | **`0.9523071`** |
| `2026-11-06T20:31:06Z` | `P2` | `L2.2` | 2-Hop Diffusion Key-Residual Propagation (smooth=40) | `1496.0 / 4595.2` | `+0.00e+00` | — | — | — | **STRIKE_2_DEAD** | **`0.9523071`** |
| `2026-11-06T20:31:33Z` | `P2` | `L2.4` | 2-Hop Bipartite Key-Residual Message Passing (smooth=25) | `1453.1 / 3851.2` | `+0.00e+00` | — | — | — | **STRIKE_4_DEAD** | **`0.9523071`** |
| `2026-11-06T20:31:43Z` | `P2` | `L2.5` | Long-Range Key-Residual Attention Surrogate (smooth=80) | `1580.3 / 4489.9` | `+0.00e+00` | — | — | — | **STRIKE_5_DEAD** | **`0.9523071`** |
| `2026-11-06T20:31:51Z` | `P5` | `L5.1` | Leave-One-Float-Out Conditional Quantile Inversion | `1777.0 / 4903.2` | `+3.15e-06` | — | — | — | **STRIKE_1_DEAD** | **`0.9523071`** |
| `2026-11-06T20:32:06Z` | `P5` | `L5.2` | Gaussian Copula Mahalanobis & Pairwise Rank Discrepancy | `809.2 / 4497.7` | `+3.15e-06` | — | — | — | **STRIKE_2_DEAD** | **`0.9523071`** |
| `2026-11-06T20:32:23Z` | `P5` | `L5.3` | Cubic B-Spline GAM with 2-Way Flag Conditioning | `176.2 / 5178.9` | `+0.00e+00` | — | — | — | **STRIKE_3_DEAD** | **`0.9523071`** |
| `2026-11-06T20:32:38Z` | `P5` | `L5.4` | 2nd/3rd-Order Key-Logit Interaction Ridge | `188.5 / 5069.1` | `+0.00e+00` | — | — | — | **STRIKE_4_DEAD** | **`0.9523071`** |
| `2026-11-06T20:32:54Z` | `P5` | `L5.5` | Monotonic Tail Saturation Spline Residual | `161.9 / 5489.6` | `+0.00e+00` | — | — | — | **STRIKE_5_DEAD** | **`0.9523071`** |
| `2026-11-06T20:34:21Z` | `P8` | `L8.2` | Unified Promoted Feature Re-Boost | `142.2 / 4056.7` | `+3.50e-06` | — | — | — | **STRIKE_1_DEAD** | **`0.9523071`** |
| `2026-11-06T20:34:29Z` | `P8` | `L8.3` | 8-Block Differentially-Regularized L4 Regime Stacker | `686.5 / 4738.7` | `+3.50e-06` | — | — | — | **STRIKE_2_DEAD** | **`0.9523071`** |
| `2026-11-06T20:34:37Z` | `P8` | `L8.4` | Direct Differentiable Pairwise RankNet L-BFGS Meta-Optimizer | `105.7 / 4512.9` | `+3.50e-06` | — | — | — | **STRIKE_3_DEAD** | **`0.9523071`** |
| `2026-11-06T20:34:45Z` | `P8` | `L8.5` | 32-Stratum Affine Logit Post-Calibration | `655.9 / 5095.1` | `+3.50e-06` | — | — | — | **STRIKE_4_DEAD** | **`0.9523071`** |
| `2026-11-06T20:42:09Z` | `P4` | `L4.3` | Velocity & Merchant-Risk Step-Grid Parity Forensics | `1792.0 / 4990.5` | `+3.68e-05` | `+0.00e+00` | `[+0.00e+00, +0.00e+00]` | `0.00016` | **STRIKE_3_5FOLD_REJECT** | **`0.9523071`** |
| `2026-11-06T20:45:36Z` | `P4` | `L4.4` | Float Mantissa, Decimal Remainder & Benford Digit Trie | `1792.0 / 4431.5` | `+1.48e-05` | `+0.00e+00` | `[+0.00e+00, +0.00e+00]` | `0.00174` | **STRIKE_4_5FOLD_REJECT** | **`0.9523071`** |
| `2026-11-06T20:45:47Z` | `P4` | `L4.5` | Rare Exact Multi-Column Tuple Co-Occurrence Count | `1792.0 / 4537.0` | `+3.83e-06` | — | — | — | **STRIKE_5_DEAD** | **`0.9523071`** |
| `2026-11-06T20:50:52Z` | `P1` | `L1.1` | 4-Block Chimera Splice Matching vs orig20k | `4104.2 / 4949.2` | `+0.00e+00` | — | — | — | **STRIKE_3_DEAD** | **`0.9523071`** |
| `2026-11-06T20:52:07Z` | `P1` | `L1.3` | 1D Wasserstein Optimal-Transport Quantile Alignment | `4104.2 / 5169.1` | `+2.48e-05` | `+0.00e+00` | `[+0.00e+00, +0.00e+00]` | `0.00276` | **STRIKE_4_5FOLD_REJECT** | **`0.9523071`** |
| `2026-11-06T20:53:29Z` | `P1` | `L1.4` | Orig20k Appended Training Fold Booster | `4104.2 / 4584.1` | `+1.28e-05` | `+0.00e+00` | `[+0.00e+00, +0.00e+00]` | `0.32944` | **STRIKE_5_5FOLD_REJECT** | **`0.9523071`** |
| `2026-11-06T20:53:49Z` | `P2` | `L2.3` | Bipartite Row-to-Key Spectral SVD-24 | `4104.2 / 4727.0` | `+0.00e+00` | — | — | — | **STRIKE_5_DEAD** | **`0.9523071`** |
| `2026-11-06T20:53:55Z` | `P6` | `L6.2` | Dedicated Stealth-Fraud (y=1, V4p<0.12) Zero-Weight Easy-Fraud Specialist | `4104.2 / 4683.2` | `+0.00e+00` | — | — | — | **STRIKE_2_DEAD** | **`0.9523071`** |
| `2026-11-06T20:54:01Z` | `P6` | `L6.4` | Out-of-Fold Consensus CleanLab 0.3% Label-Flip Pruning | `4104.2 / 4698.7` | `+0.00e+00` | — | — | — | **STRIKE_4_DEAD** | **`0.9523071`** |
| `2026-11-06T20:54:06Z` | `P7` | `L7.1` | PLE Quantile-Embedded Ridge/MLP Surrogate | `4104.2 / 4915.5` | `+7.73e-06` | — | — | — | **STRIKE_1_DEAD** | **`0.9523071`** |
| `2026-11-06T20:54:09Z` | `P7` | `L7.2` | Inter-Family Epistemic Dispersion Gate | `4104.2 / 4958.0` | `+7.73e-06` | — | — | — | **STRIKE_2_DEAD** | **`0.9523071`** |
| `2026-11-06T20:54:13Z` | `P7` | `L7.3` | Forced-Blindness Specialist 1 — Blind to High-Card Exact TE | `4104.2 / 4933.4` | `+7.73e-06` | — | — | — | **STRIKE_3_DEAD** | **`0.9523071`** |
| `2026-11-06T20:54:17Z` | `P7` | `L7.4` | Forced-Blindness Specialist 2 — Blind to Merchant-Risk & Velocity | `4104.2 / 4860.3` | `+3.82e-06` | — | — | — | **STRIKE_4_DEAD** | **`0.9523071`** |
| `2026-11-06T20:54:20Z` | `P7` | `L7.5` | Empirical-Bayes 2-Level Flag->Key Hierarchy | `4104.2 / 4961.3` | `+3.85e-06` | — | — | — | **STRIKE_5_DEAD** | **`0.9523071`** |
| `2026-11-06T20:56:10Z` | `P6` | `L6.1` | 5-Class Fraud Sub-Archetype Multi-Task Booster | `3361.2 / 5180.9` | `+7.95e-06` | — | — | — | **STRIKE_3_DEAD** | **`0.9523071`** |
| `2026-11-06T20:56:14Z` | `P6` | `L6.3` | Huberized Gradient-Clipped Booster | `3361.2 / 5209.3` | `+8.17e-06` | — | — | — | **STRIKE_4_DEAD** | **`0.9523071`** |
| `2026-11-06T20:56:17Z` | `P6` | `L6.5` | Overlap-Band Focal-Weighted Booster | `3361.2 / 5196.4` | `+3.15e-06` | — | — | — | **STRIKE_5_DEAD** | **`0.9523071`** |

## Nov07 Local CPU/MPS Submit Day (UTC)

- **Host:** Apple Silicon MPS available (`torch 2.15.0`); peak research RSS ~0.8–0.9 GB; host avail mostly **6.7–8.3 GB**.
- **OOF bank check:** `v4p_hc=0.9523071` (honest champ); `v7_clean/master` OOF higher but known soft-pseudo inflation (do not trust for LB).
- **Nested logit blend** of residuals onto `v4p_hc`: best `C=1.0` → OOF `0.9523083` (`+1.21e-6`), `|Δρ|=7e-5` (promoted locally, negligible).
- **MPS RankNet residual** on bank ranks: OOF **`0.94918` (`−3.12e-3`)** → rejected.
- **All 10 daily slots used.** Public best remains **`0.95243`** (tied by S1/S3/S5/S7/W2A/W2C). Pure-prop ridge dropped to `0.95233`. Logs: `logs/overnight/nov07_submit_ladder.json`, `nov07_wave2.json`.
- **Plateau diagnosis:** further logit-axis remixes of the `live10/ev4` peak are ρ≥0.9995 vs champ and do not move Public LB. Need a new orthogonal OOF-gated head (likely Paradigm 3 GPU / new family), not more extrapolation.
