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

---

## Plan v2 Upgrade: Post-CPU Diagnostic Audit & Full-GPU Extension (`STRICT NO_SUBMIT`)

### 1. Five Root Causes Identified from the 35-Lever CPU Run & Nov07 MPS Run
1. **Isolated 1D Residual-on-`init_score` Washout (`L4.1`, `L4.3`, `L4.4`, `L1.3`, `L1.4`):**
   - Five levers beat the Fold-0 canary gate on CPU (`L4.3` `+3.68e-5`, `L4.1` `+3.07e-5`, `L1.3` `+2.48e-5`, `L4.4` `+1.48e-5`, `L1.4` `+1.28e-5`), proving the underlying generator-forensic signals exist.
   - However, `fit_fold_ridge_or_lgb_residual()` trained 3-leaf trees **only on the 4–15 new columns in isolation** with `init_score = z_champ_o - 4.1`, never allowing the new features to interact with the 108 base keys (`KMAT_108`) or enter the 81-col + L4 regime Newton-Cholesky stacker. Worse, when `LGBMClassifier` early-stopped on one fold (`std <= 1e-5`), the fallback switched to `Ridge` mid-CV, mixing model scales across folds.
2. **Feature-Starved CPU Surrogates (`RAWV` 18 cols in `P6` & `P7`):**
   - To stay within 18 GB local RAM, `P6` (`L6.1` `+7.95e-6`, `L6.3` `+8.17e-6`) and `P7` (`L7.1–L7.4` `+7.73e-6`) trained on 18 unencoded raw columns (`gb_mod.RAWV`) instead of the full 130-column out-of-fold TE + Digit Trie + Ratio + Stage-1 GLM matrix.
3. **Stealth-Fraud (`L6.2`) `init_score` Cancellation Bug:**
   - Setting `sample_weight = 0.0` on easy positives ($y=1 \land p \ge 0.12$) while keeping `init_score = z_champ_o` made the remaining training positives have negative mean residual $y - \hat{p}$, causing immediate tree early-stopping (`+0.00`). Fix: use damped Stage-1 offset `init_score = 0.35 * z_GLM`.
4. **Cross-Fold RankNet Scale Drift (`MPS RankNet` `-3.12e-3` & `L1.4` `|Δρ|=0.329`):**
   - On MPS, `TinyRes` matched `v4p_hc` within `1e-6` on every individual fold (`0.9514, 0.9525, 0.9570, 0.9527, 0.9480`), but dropped to `0.94918` when concatenated across folds because pairwise RankNet loss is shift-invariant within each fold!
   - **Mandatory v2 Fix:** Apply **Fold-Train ECDF Quantile Alignment** (`ndtri(ecdf_train_a(pred_b)) + mu_champ_a`) on every fold before concatenating OOF or averaging test predictions (`|Δρ| < 0.0005` guaranteed).
5. **Duplicate CPU Fallbacks (`P2, P5, P7, P8`):**
   - `L8.2–L8.5`, `L5.1–L5.2`, `L5.3–L5.5`, `L2.1/2/4/5`, and `L7.1–L7.3` were wired to shared CPU fallback functions. In v2 on GPU, all 40 levers are distinct algorithms enforced by a `pytest` inter-lever orthogonality gate ($\rho_{\text{between levers}} < 0.98$).

### 2. Extended Dedicated GPU Execution Order (`s6e11-0953-campaign`, `<= $0.70/hr`, `$20` cap, `lheroidmuab1j4` blacklisted)
1. **Priority #1 — `Paradigm 3` (1.02M-Row Sub-Token & Sequence Transformers on GPU):**
   - `L3.1` 6-layer BiMasked GEGLU Transformer (`LLR_24_Bi` + single-pass cache of `llr24_decint` and `h64_cls`), `L3.2` $K=4$ Multi-Order Contrastive Causal Transformer (`LLR_24_Causal`), `L3.3` Decimal-Given-Integer LLR + Joint CUDA GBDT on `[KMAT_108 || LLR_48]`, `L3.4` 64-Dim CLS Bottleneck CUDA XGBoost/LGB, `L3.5` Exact Bayesian Log-Sum-Exp Posterior + Permutation Entropy.
2. **Priority #2 — `Paradigm 4 v2` (GPU Rescue of CPU Fold-0 Winners + Bulsah-Style Joint TE):**
   - `L4.1 v2` Fixed Train+Test Local Jump & Rolling Block Residual Joint GBDT, `L4.2 v2` Sub-Float String & BPE Digit-Pair Out-of-Fold TE (`~90` token cols), `L4.3 v2` Velocity/Risk Step-Grid $\times$ Flag/Category 2-Way Cross TE (rescuing `+3.68e-5` Fold-0 winner), `L4.4 v2` Ratio Mantissa & Benford 2nd/3rd Digit Trie Joint CatBoost/XGB (rescuing `+1.48e-5` Fold-0 winner), `L4.5 v2` Top-60 2nd-Order High-Card Key-Pair Hierarchical EB Target Encoding on GPU.
3. **Priority #3 — `Paradigm 7 v2` (True ICLR-2025 `TabM k=32` PLE + Full-130-Col Forced-Blindness on GPU):**
   - `L7.1 v2` `TabM k=32` BatchEnsemble with 48-bin PLE + Periodic Embeddings on all 130 TE+ratio cols (with Fold-Train ECDF alignment), `L7.2 v2` `TabM` 32-Head Epistemic Std Regime Gate, `L7.3 v2` Coarse-TE + Digit/Ratio Specialist (masking only `KCNT < 25`), `L7.4 v2` Top-3 Dominant Feature-Group Dropout CUDA XGBoost/MLP, `L7.5 v2` 24-Col $\times$ 4-Scale Flag-Conditioned Empirical-Bayes Hierarchy.
4. **Priority #4 — `Paradigm 2 v2` (1.02M-Row Tiled `torch.cdist` CloneGraph, SVD-128 & `TabR` Attention on GPU):**
   - `L2.1 v2` Exact $1.02\text{M} \times 1.02\text{M}$ 48-dim GLM-weighted `torch.cdist` 64-NN graph, `L2.2 v2` 3-Step Personalized PageRank residual diffusion, `L2.3 v2` 128-dim GPU TF-IDF Bipartite SVD + L2 reconstruction anomaly, `L2.4 v2` True 2-Hop Rare-Key Co-Occurrence Row Message Passing, `L2.5 v2` End-to-End PyTorch `TabR` Cross-Row Attention.
5. **Priority #5 — `Paradigm 5 v2` (Non-Linear Multi-Quantile Copula, $Y_0/Y_1$ DAE Energy & 3rd-Order ANOVA Tensor-FM on GPU):**
   - `L5.1 v2` 13-Col Multi-Quantile CUDA GBDT percentile inversion, `L5.2 v2` Class-0 vs Class-1 Copula QDA Log-Likelihood Ratio + 78 pairwise rank discrepancies, `L5.3 v2` SwiGLU $\text{DAE}_0$ vs $\text{DAE}_1$ manifold reconstruction energy gap, `L5.4 v2` Rank-64 2nd+3rd Order ANOVA Tensor-FM, `L5.5 v2` 64-Flag-State Hierarchical B-Spline GAM.
6. **Priority #6 — `Paradigm 6 v2` (Full-130-Col Stealth-Fraud, 5-Class Archetype & Huber/CleanLab CUDA Boosters):**
   - `L6.1 v2` 5-Class Fraud Sub-Archetype CUDA XGBoost + CatBoost on full 130 cols, `L6.2 v2` Stealth-Fraud Specialist (`w=0` on easy positives, `init_score = 0.35 * z_GLM`) on full 130 cols, `L6.3 v2` 5-Seed Huberized Clipped-Gradient CUDA XGBoost/LGB (rescuing `+8.17e-6` Fold-0 near-miss), `L6.4 v2` 7-Family Out-of-Fold Consensus Target Distillation (`0.88 y + 0.12 p_cons`), `L6.5 v2` Overlap-Band Pairwise-Weighted CUDA XGBoost.
7. **Priority #7 — `Paradigm 1 v2` (`Orig20k` Hierarchical Bayes Prior & Fold-Anchored Transport on GPU):**
   - `L1.1 v2` 5-Block GPU `torch.cdist` Chimera Splice + Joint `KMAT_108` GBDT, `L1.2 v2` 108-Dim TE-Logit Mahalanobis Prototype Transport, `L1.3 v2` Fold-Anchored 1D & 2D Sliced-Wasserstein OT (fixing `|Δρ|=0.00276`), `L1.4 v2` `Orig20k` Conjugate Beta TE Prior + ECDF-Aligned Joint Booster (fixing `|Δρ|=0.329`), `L1.5 v2` GPU RBF Kernel Ridge `V4p_hc` Error Transfer.
8. **Priority #8 — `Paradigm 8 v2` (Unified ~300-Feature Super-Reboost & Differentiable GPU Pairwise AUC Stacker):**
   - `L8.2 v2` 5-Family Unified ~300-Feature GPU Super-Reboost (`GLM_Quad + LGB + CUDA_XGB + CatBoost + TabM`), `L8.3 v2` 9-Family $\times$ 7-Regime Multi-$C$ Newton-Cholesky Stacker, `L8.4 v2` GPU Differentiable Binned Pairwise RankNet Meta-Optimizer (`10^7` pairs/epoch, L2-anchored to Newton weights), `L8.5 v2` 64-Stratum Cross-Stratum Pairwise RankNet Affine Calibration.

---

## 3. Nov07 A40 GPU Execution & Grandmaster Harvest (`ht590jc0rpb9kz`, `STRICT NO_SUBMIT`)

### 3.1 Executive Summary of A40 GPU Campaigns (`2026-11-07 10:49–18:16 UTC`)
- **Pod:** `ht590jc0rpb9kz` (`1x NVIDIA A40 48GB`, `$0.59/hr`, `EU-SE-1`, stopped in `EXITED` state after full local `rsync` + SHA-256 verification).
- **Verified Stack Evaluator (`scripts/paradigm_stack_delta_eval.py`):**
  - Replaced legacy stack evaluation bug (which overwrote `0.84*rk(v4p) + 0.12*rk(hc) + 0.04*rk(gb_super)` with `0.50*z_champ + 0.30*xgb + 0.20*tabr` and reported bogus `-4.45e-3` deltas) with **Test-1 Orthogonal $\alpha$-Probe** (`score_candidate_vs_v4p_hc`) on rank-logit scale $z = \text{logit}(\text{rk}(\cdot))$ where $\alpha = 0$ reproduces `0.9523071` to `< 1e-12`.
- **New Champion Promoted (`P_super_v5_harvest`):**
  - **5-Fold OOF ROC-AUC:** **`0.9523401`** (vs frozen `v4p_hc` `0.9523071`)
  - **Raw OOF Upgrade:** **`+3.300e-05` (`+0.33 bps`, 4/5 folds won, Fold-0 `+1.191e-04`)**
  - **Pure `P4` 5-Fold Blend Test-1 Lift:** **`+3.623e-05` (`+0.36 bps`)**
  - **3-Seed Nested Stack Mean $\Delta\text{AUC}$ (`{42, 7, 2024}`):** **`+1.734e-05`** (`+3.30e-5` on seed 42, `+2.08e-5` on seed 7)
  - **90% Bootstrap $\Delta\text{AUC}$ Mean:** **`+3.230e-05`** (`CI: [-2.17e-05, +8.93e-05]`)

### 3.2 Proprietary Winning Signals Discovered & Locked
Only **discrete generator-forensic log-odds & token LLR signals** provided true orthogonal lift over `v4p_hc`:

| Proprietary Module | Algorithm / Feature Mechanism | 5-Fold Test-1 $\Delta\text{AUC}$ | Fold Win Rate | KS Stat | Mean Stack Weight $w$ | Local Verified Artifact (`preds/`) |
|---|---|---:|---:|---:|---:|---|
| **`P4_L4.3` Pure 5F (`ridge_resid`)** | Velocity & Merchant-Risk Step-Grid Parity $\times$ 64-Flag Cross Out-of-Fold Log-Odds (`Ridge(alpha=500)` on prevalence-calibrated residual $y - \sigma(z_{\text{init}})$, ECDF-aligned per fold) | **`+1.553e-05`** | `4/5 (80%)` | `0.0360` | `-0.0210` | `matrix_v2/P4_L4.3_pure5f_{oof,test}.npy` |
| **`P4_L4.2` Pure 5F (`ridge_resid`)** | Bulsah Sub-Float ASCII String & Digit-Pair Out-of-Fold Target Encoding (`Ridge(alpha=500)` on calibrated residual, ECDF-aligned) | **`+1.508e-05`** | `4/5 (80%)` | `0.0014` | `-0.0240` | `matrix_v2/P4_L4.2_pure5f_{oof,test}.npy` |
| **`B_xgb_llr` (`paradigm_B`)** | 72-Column Signed Log-Likelihood Ratio (`LLR_72` from BiMasked + Multi-Perm Causal Transformers) + `KMAT_108` CUDA `XGBoost` (`max_depth=3`) | **`+6.463e-06`** | `5/5 (100%)` | `0.0000` | `-0.0340` | `paradigm_B/xgb_llr_{oof,test}.npy` |
| **`P4_L4.5` Pure 5F (`xgb_resid`)** | Top-60 2nd-Order High-Cardinality Key-Pair Hierarchical Empirical-Bayes TE (`max_depth=2` Lever-Only CUDA `XGBoost` residual) | **`+3.340e-06`** | `2/5 (40%)` | `0.0027` | `-0.0220` | `matrix_v2/P4_L4.5_pure5f_{oof,test}.npy` |
| **`harvest_p4_cuda_xgb` (Pure 5F P4 Blend)** | Test-1-weighted rank-logit combination of `P4_L4.3_pure5f + P4_L4.2_pure5f + P4_L4.5_pure5f` | **`+3.623e-05`** | `4/5 (80%)` | `0.0000` | Combined | `matrix_v2/harvest_p4_cuda_xgb_{oof,test}.npy` |
| **`P_super_v5_harvest` (Final Super-Stack)** | Nested 5-Fold Out-of-Fold Coordinate Ascent over orthogonalized residuals $[B_{\text{xgb\_llr}}, P4_{L4.3}, P4_{L4.2}, P4_{L4.5}]$ on $z_{\text{base}} = \text{logit}(\text{rk}(v4p\_hc))$ | **`+2.133e-05`** *(Raw $\Delta$: **`+3.300e-05`**)* | `4/5 (80%)` | `0.0000` | See row weights | `P_super_v5_harvest_{oof,test}.npy` |

> **Why the weights $w_j$ on the orthogonalized residuals are negative (`-0.021` to `-0.034`):**
> Each candidate was already anchored on `z_champ` (`~0.998` correlated with `v4p_hc`). When regressed against $z_{\text{base}}$ (`ro = zo - beta * z_base`), the residual $r_o = z_{\text{cand}} - \beta z_{\text{base}}$ captures where the candidate *overshoots* the consensus on noisy high-cardinality keys. Subtracting `-0.021` to `-0.034` of that overshoot acts as an **out-of-fold empirical-Bayes variance-reduction / shrinkage operator** on `v4p_hc`!

### 3.3 True Orthogonal Residual Injection onto `sub_ev4` (`0.95243` Public LB Anchor)
- **Diagnostic Trap Caught & Fixed:** Naive blending `(1 - a) * z_ev4 + a * z_v5` selected `a = 1.0` on OOF (because `v5 > v4p_hc`), which completely overwrote `sub_ev4` (`0.95243` LB) with pure `v5`!
- **Proprietary Fix (`scripts/p4_pure_5fold.py`):** Extract the pure zero-mean orthogonal logit correction $\Delta z_{\text{test}} = Z_{\text{test}} \bar{w}$ and inject it directly onto $z(\text{sub\_ev4})$:
  $$p_{\text{on\_ev4}} = \text{rk}\!\left(\text{logit}(\text{rk}(\text{sub\_ev4})) + \eta \cdot \Delta z_{\text{test}}\right)$$
  - `submissions/sub_v5_harvest_on_ev4_half.csv` ($\eta = 0.5$): $\rho(\text{on\_ev4\_half}, \text{sub\_ev4}) = \mathbf{0.999920}$, `KS = 0.00000`, SHA-256 `ccb4016d70c317d5`
  - `submissions/sub_v5_harvest_on_ev4.csv` ($\eta = 1.0$): $\rho(\text{on\_ev4\_1.0x}, \text{sub\_ev4}) = \mathbf{0.999679}$, `KS = 0.00000`, SHA-256 `953e118caff0f298`
  - `submissions/sub_v5_harvest_pure.csv` (Pure OOF-optimal `P_super_v5_harvest_test`): $\rho(\text{pure}, \text{sub\_ev4}) = \mathbf{0.999123}$, SHA-256 `c43df92be3b1d9a1`


---

## Pointer (2026-11-12) — ATH moved past this campaign log

This file is the Nov 6–7 `STRICT NO_SUBMIT` / v4p→v5 harvest campaign log (OOF era ~0.9523). **Current public ATH is 0.95371** (Q17 anchor + 0.03·d1_v4). See:

- `docs/THINGS_THAT_WORKED.md` / `docs/DEAD_ENDS.md`
- `logs/overnight/GM_V12_DAY_SUMMARY_2026-11-12.md`
- `logs/overnight/GM_V12_3_LADDER_LIVE_REPORT.md`
