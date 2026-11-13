# GM V13.2 — Overnight Upstream + δ1_v14 Stage-B on ATH 0.95388

**STRICT NO_SUBMIT=True** · POD_TOUCHES=0 · elapsed ~319s (+95s corrected eval)  
**Script:** `scripts/gm_v13_2_overnight_check_and_stageB_v14.py`

## Step 1 — Upstream

- **G26 `gpt2_s2026_test_5fold` / f1–f4:** **ABSENT** (remote still only `gpt2_s2026_oof_5fold.npy` + `gpt2_s2026_test_f0.npy`, stamp 2026-11-12 09:14 UTC). ~15h later: **no new G26 files**.
- Nov-13-updated s6e11 datasets: **2**
  - `alexchilton/s6e11-oof-predictions` (2026-11-13 08:51) — downloaded → `external/nov13_recon/alexchilton_oof_predictions/`
  - `daoviet/s6e11-gp-dvae-orthogonal-apex` (2026-11-13 05:29) — refreshed → `external/nov13_recon/daoviet_gp_dvae_apex_nov13/`
- Public kernel board still capped at **0.95388** (`areebbb/…-bold-then-bolder`); no open **0.9539x+** artifact.

### alexchilton OOF screen (top)

| file | OOF AUC |
|---|---:|
| `oof_4188470eded4cdec.npy` | 0.95066 |
| `oof_82e9a79659bdf9d8.npy` | 0.95054 |
| `oof_21b4d1316c55b429.npy` | 0.95052 |

ATH OOF AUC ref: **0.95243** — alex members are mid-tier base learners, not ATH-level.

## Step 2 — δ1_v14 Stage-B on ATH 0.95388

- ATH OOF proxy: `bold MIX + 0.03·unitize(d1_v4)` · AUC=0.9524301 · test locked to LIVE `sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv`
- **X_v14 = 41 cols:** KMAT_top30 (from `X_v5` keys) + P4_L4.2/3/5_pure5f + contrasts `MS−LM`, `L7F−LM`, `GPD−MLP`, `Gmed−MS` + `std(MS,LM,L7F,GPD)` + `|LM_mean−Tree_mean|` + denoise(`G26−MS`) + denoise(`L7_s2026_K10−L7F`)
- Denoise dispersion lock: **disp(G26)=1.0000**, **disp(L7_s2026)=1.0000** (fold0-only real L7_s2026; other folds Ridge-filled)
- Boosters: `lgb+xgb+cat` · `max_depth=3, num_leaves=6, min_child=500, colsample=0.35, λ=80, lr=0.015, n=160` · **`init_score = z_ATH`**
- Artifacts: `preds/gm_v13_2_stageB/d1_v14_{oof,test}.npy`, `d1_blend_v4_v14_{oof,test}.npy`
- Nested gate bar: **≥3/5** main folds with subfold 5/5

### Dual-gate scorecard (inject on ATH)

| candidate | 5/5 | meanΔ (e-5) | min | 90% boot CI | nested SF | dual |
|---|---:|---:|---:|---|---:|---|
| `ATH+d1_v14_w0.02` | 3 | +0.056 | −1.13 | [−0.66, 0.75] | **0/5** | False |
| `ATH+d1_v14_w0.03` | 3 | −0.004 | −1.67 | [−1.08, 1.01] | **0/5** | False |
| `ATH+d1_v14_w0.035` | 3 | −0.027 | −1.95 | [−1.30, 1.17] | **0/5** | False |
| `ATH+d1_v14_w0.04` | 3 | −0.092 | −2.29 | [−1.57, 1.27] | **0/5** | False |
| `ATH+blend_v4v14_w0.02` | 5 | +1.216 | +0.37 | [0.66, 1.90] | **0/5** | False |
| `ATH+blend_v4v14_w0.03` | 4 | +1.499 | −0.13 | [0.56, 2.56] | **0/5** | False |
| `d1_v4_retune_w0.035_vs_ATH` *(corrected: +0.005·d1)* | 4 | +0.478 | −0.06 | [0.23, 0.73] | **0/5** | False |

Per-family at w=0.03: lgb 3/5 mean−0.07 · xgb 3/5 mean−1.21 · cat 2/5 mean−0.50 — all nested **0/5**.

### Diagnosis

- `ρ(δ1_v14, δ1_v4) ≈ 0.068` — mostly a **new** direction, but **null/negative** on ATH (not a useful residual).
- Ensemble is LGB-dominated (`ρ(d14,lgb)≈0.98`); XGB/Cat do not rescue nested stability.
- `0.5·v4+0.5·v14` blend that looks 5/5 at small w is **just diluted `d1_v4` energy** — still **nested 0/5**, same V13.1 pathology.
- Corrected `d1_v4` w:0.03→0.035 vs ATH is only **4/5** here (V13.1’s 5/5 was MIX-vs-MIX with a slightly different MIX; either way nested fails).

## Staging decision

**NOT STAGED.** No candidate cleared **main 5/5 ∧ nested ≥3/5**, and none beat corrected `d1_v4` retune on **both** meanΔ and nested SF.  
`submissions/sub_v13p2_d1_v14_verified_NOSUBMIT.csv` was **not** written.

## Bottom line

1. **G26 5-fold test still not on Kaggle** — highest external lever unchanged.  
2. Retraining Stage-B **directly on the 0.95388 ATH base did not produce a nested-stable residual** — the “obsolete-base” hypothesis for `δ1_v4` nested failure is **not** rescued by `δ1_v14` under this feature/regimen set.  
3. Public board still **0.95388**; no local dual-gate path to 0.9539x from this overnight pass.
