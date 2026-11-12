# GM V12.1 Forensic: Q17 Leak + knightfork + Fold0 Subfold LOFO

**UTC:** 2026-11-12T20:03:20.741334+00:00  
**`NO_SUBMIT=True`** — 0 uploads; **no RunPod pod touches this turn**  
**Elapsed:** 145.48s

## Step 1 — Pods
- **No pod operations executed** (explicit user hard-stop).
- `vjez42fl07jtrl`: already absent (404) at prior check; not touched
- `w102kdrd4pmsz8`: observed as current P730 successor; NOT inspected/terminated this turn
- Foreign `never touch` untouched.

## Step 2 — Q17 `anchor_095370` leakage & late kernels
### Prior audit (`GM_AMANATAR_Q17_AUDIT.md`) — executive excerpt
```
# GM Track-2 Audit: amanatar Q17 + detective + G26
**UTC:** 2026-11-12T16:32:36.735189+00:00  
**`NO_SUBMIT=True`** — 0 uploads  
**Elapsed:** 516.57s

## Executive verdict
- **`anchor_095370` cleared the dual LOFO gate** (S7m **5/5**, simple **4/5**, `d0>0`). Orth residual meanΔ ≈ **+0.035e-5** on S7m. **All 16 Q17 member streams failed** the dual gate (several hit S7m 5/5 alone — `gbn_bag5_w512`, `gb_bag5` — but simple ≤3/5).
- **HOLD all 4 submits** (`NO_SUBMIT=True`): anchor is already the public 0.95370 apex; do **not** burn a cand2↔anchor blend without a dedicated promotion plan.
- **F2−F4 > 0.004 on every stream** including known-clean public OOFs → structural fold difficulty, **not** cross-fold leak.
- **G26 live `test_5fold`:** **ABSENT** (still oof_5fold + test_f0 only).
- **Q17 `test_gpt2_seed2026` ≡ G26 `test_f0`** (`1−ρ = 0`) — not a full 5-fold test tensor.
- **Detective:** `model.pt` (~475MB) + `config.json`; data = orig20k CSV only — no OOF/test bank.

## Live dataset file lists
### amanatar/s6e11-rank1-q17-meta-bundle
```
name                       size  creationDate                
---------------------  --------  --------------------------  
rank1_q17_streams.npz  64574677  2026-11-12 14:04:47.476000
```
```

### Stream bank keys / solo AUCs
| stream | solo AUC | F0 | F1 | F2 | F3 | F4 |
|---|---:|---:|---:|---:|---:|---:|
| anchor_095370 | 0.9525514 | 0.9517749 | 0.9525318 | 0.9572653 | 0.9528037 | 0.9483810 |
| megayak_5f_lm | 0.9504517 | 0.9485056 | 0.9506818 | 0.9560850 | 0.9506043 | 0.9463809 |
| masaya_gpt2 | 0.9500480 | 0.9488823 | 0.9503172 | 0.9553120 | 0.9498742 | 0.9458540 |
| lm7_5fold | 0.9494440 | 0.9475445 | 0.9497671 | 0.9551662 | 0.9490652 | 0.9456760 |
| gpt2_seed2026 | 0.9501731 | 0.9491911 | 0.9503701 | 0.9552357 | 0.9502097 | 0.9458585 |
| tabpfn_slimypunk_tp1 | 0.9493429 | 0.9481552 | 0.9503329 | 0.9547382 | 0.9491845 | 0.9443034 |
| sadam_tabpfn_100k_tp3 | 0.9488896 | 0.9475724 | 0.9508788 | 0.9529789 | 0.9490339 | 0.9439834 |
| yekenot_realmlp | 0.9501145 | 0.9500496 | 0.9510404 | 0.9549457 | 0.9503984 | 0.9441386 |
| alex_glm_gen | 0.9505035 | 0.9494181 | 0.9506593 | 0.9555146 | 0.9514767 | 0.9454483 |
| daoviet_gpdvae | 0.9465294 | 0.9447321 | 0.9476896 | 0.9514630 | 0.9466615 | 0.9420998 |
| gbn_bag5_w512 | 0.9512429 | 0.9499157 | 0.9516058 | 0.9560761 | 0.9518553 | 0.9467612 |
| sameer_glm10f | 0.9516046 | 0.9508096 | 0.9520959 | 0.9563718 | 0.9520656 | 0.9466798 |
| sadam_v6_gb_xgb | 0.9507703 | 0.9506578 | 0.9504680 | 0.9558513 | 0.9510173 | 0.9458569 |
| sadam_v6_gb_lgb | 0.9509402 | 0.9506461 | 0.9506616 | 0.9560987 | 0.9513073 | 0.9459875 |
| sadam_v4_cat | 0.9501135 | 0.9491888 | 0.9505731 | 0.9550723 | 0.9510023 | 0.9447305 |
| gb_bag5 | 0.9515369 | 0.9504322 | 0.9515890 | 0.9562735 | 0.9523948 | 0.9469945 |
| megayak_shallow_tree | 0.9515043 | 0.9503825 | 0.9518021 | 0.9562093 | 0.9525071 | 0.9466198 |

### Possibility A vs B — OOF vs Test blend weights
- Ridge R² OOF=0.994034, Test=0.982663
- max|w_oof − w_test| (Ridge) = **9.116272e-01** ; OLS = **9.141872e-01**
- Identical at 1e-4? **False**
- Corr(test, apply w_oof on test streams) = **0.989263**
- **Verdict: Possibility B: OOF/test constructed differently (weights diverge)**

Top |Δw| streams:
- `megayak_shallow_tree`: w_oof=+0.5745, w_test=-0.3371, |Δ|=0.9116
- `gb_bag5`: w_oof=+0.3435, w_test=+1.1050, |Δ|=0.7614
- `sadam_v6_gb_xgb`: w_oof=+0.5859, w_test=+1.2160, |Δ|=0.6301
- `megayak_5f_lm`: w_oof=+0.4063, w_test=+0.8609, |Δ|=0.4546
- `sameer_glm10f`: w_oof=+0.3658, w_test=-0.0657, |Δ|=0.4315
- `sadam_v6_gb_lgb`: w_oof=-0.1337, w_test=-0.5405, |Δ|=0.4068
- `masaya_gpt2`: w_oof=+0.4427, w_test=+0.1026, |Δ|=0.3400
- `tabpfn_slimypunk_tp1`: w_oof=-0.0053, w_test=+0.2103, |Δ|=0.2156

### `test_anchor_095370` Spearman distances (`1−ρ`)
| ref | 1−ρ |
|---|---:|
| A_095340 | 0.00165884 |
| F_095330 | 0.00179762 |
| z_S7_ours | 0.00104637 |
| cand2 | 0.00098960 |
| areeb_095347 | 0.00115785 |
| s354 | 0.00107220 |

### knightfork LB 0.95357 — exact formula
- **Formula:** `L(preserved_095354_ladder) + 0.25 * (L(sovereign) - L(wide))`
- Public: 0.95354 → 0.95357 (+0.00003)
- Proxy OOF Δ≈1.7580980577e-05 with 4/5 folds up
- Inputs: goodpjw LM Ladder 0.95354 reference + Amanatar Sovereign vault + wide stack

### asterios / markbarney / modansaif
- **asterios:** Public top is a dense noisy band; nested CV on wide-stack OOF separates probes that survive private 80%. Method: Simulate public(~60k)/private(~240k) splits from goodpjw nested wide OOF; replay published ladder moves. Takeaway: Many tiny public gains are winner's-curse; prefer nested-CV positive moves (e.g. G26−MS) over probes that lose OOF.
- **markbarney:** Plain-language postmortem of ~85 own models / dead ends over 12 days; invites collaboration. Compute: Home lab + RTX 5070/4090 + DGX Sparks + ~$39 Vast.ai. Publishes failures intentionally; lists ideas not yet tested (other LMs, BERT-style).
- **modansaif:** pulled to `/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/external/nov12_late_kernels/modansaif` (Fraud-fusion reproduction / credited — pulled for completeness.)

## Step 3 — Fold-0 5-subfold stratified LOFO (fixed weights)
Base = `eval_S7matched_355[fold0]` · SF = `StratifiedKFold(5, shuffle=True, rs=42)` on 139,992 rows.

| move | pos/5 | meanΔe5 | [sf0..sf4] e-5 |
|---|---:|---:|---|
| MoveA_fixed_w0.036 | 4/5 | 0.0264 | [-0.031, 0.1087, 0.0156, 0.0078, 0.0311] |
| MoveB_fixed_w0.036 | 4/5 | 0.0544 | [-0.0078, 0.0388, 0.07, 0.0311, 0.14] |
| MoveC_fixed_B_plus_0.025_LMtrim | 5/5 | 0.0855 | [0.0854, 0.1552, 0.0622, 0.0311, 0.0933] |
| MoveD_q17_orth_anchor_w0.036 | 3/5 | 0.0358 | [-0.031, 0.0854, -0.0467, 0.1167, 0.0544] |
| MoveD_q17 FULL F0..F4 | 5/5 | 0.0154 | [0.0118, 0.0124, 0.0162, 0.0357, 0.0009] |

### Interpretation
- Prior V12 in-sample Move C (+0.218e-5) used a **2-parameter grid** on all of Fold 0. Fixed-weight subfold C is **5/5**, mean **0.0855e-5**.
- Move A fixed (0.036× analog of G26−MS): **4/5**, mean **0.0264e-5**.
- Move B fixed: **4/5**, mean **0.0544e-5**.
- Move D (orth Q17 anchor) subfold **3/5**; full5 **5/5** mean **0.0154e-5**.

**HOLD all 4 daily submissions** until promotion law is satisfied on debiased evidence.

---
*Generated by `scripts/gm_v12_1_forensic_q17_knightfork_and_f0_subfold.py` — no pod ops.*
