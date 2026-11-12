# S6E11 Dead Ends

Catalogue of approaches that were tried, measured, and **should not be re-run** without a new hypothesis.  
Policy context: **STRICT NO_SUBMIT** for ladder / local CPU recon. Sources: `logs/overnight/paradigm_ladder_progress.json`, `ladder_local_cpu_audit.json`, `local_testbag_tabpfn_recon.json`, `part10_upgrade_psuper.json`.

Locked CV: `StratifiedKFold(5, shuffle=True, random_state=42)`.  
Stack baseline referenced below: `S_fold0_base = 0.950841`, `P_super_champ` nested = **0.9521425**.

---

## 1. Paradigm ladder (A5000 overnight)

Gate rules (Fold-0 stack ΔS / standalone AUC): R1 ≥0.94935 or ΔS≥4e−5 · R2 ≥0.94980 or ΔS≥8e−5 · promote ΔS≥3e−5.

| ID | Approach | Result | Why dead | Artifact |
|----|----------|--------|----------|----------|
| L1 | **Rung 1** Multi-perm Bayes LLR K=8 + δ24 Ridge (24k FT GPT-2, 16k gate score) | FAIL · K-curve 0.850→0.854 · δ24=**0.904** · ΔS=**−0.00109** | Standalone ≪ Masaya Fold0 0.94888 (−0.095); stack **hurts** base | `paradigm_ladder_progress.json` → `rung1` |
| L2 | Rung 1 enrich at `SCORE_BS=1536` dual-prefix | CUDA OOM | bs≤256 required on A5000 | ladder log; fixed fallback in `run_paradigm_ladder.py` |
| L3 | `Ridge.decision_function` for δ24 | AttributeError crash after K=8 score | sklearn Ridge uses `.predict` | patched → `.predict` |
| L4 | Gate1 watcher treating `auc_sum_llr` as ready | False relaunch; killed Rung2 mid-pretrain | Require `auc_weighted_delta` only | watcher script |
| L5 | **Rung 2** Transductive LM + contrastive LLR | FAIL · K8=0.943 · δ24=**0.929** · ΔS=**−0.00110** | Beats R1 but still fails gate; stack still negative | `rung2` |
| L6 | Rung 2 score at bs=1536 | CUDA OOM after contrastive ckpt saved | Re-score from cache at bs=256 | `rung2_fold0_contrastive/` |
| L7 | **Rung 4** BiTabGPT (12 ep, bf16) | FAIL · AUC=**0.94943** · ΔS=**−1.7e−5** | Below Gate4 0.95080; tiny negative stack Δ | `rung4` |
| L8 | **Rung 5** full 5-fold + 100% refit of passers | No nested OOF / no refit npy | Champ builder needed missing Part7 notebook on pod; only logged `rung3` Fold-0 ΔS | `rung5.promoted[].nested_auc = null` |
| L9 | Ladder `*_score_full.npy` as honest 5-fold members | Not usable as OOF | Fold-0 gate scores / Masaya-filled proxies; **no test** vectors for r1/r2/r3/r4 | `preds/ladder/` |

**Rung 3 is a DEAD END (audit correction 2026-11-05):** Fold-0 ΔS=+7.3e−5 was measured against a **3-member fallback** stack (`S_fold0_base=0.950841`) after Part-7 notebook `FileNotFoundError` on the pod — **not** against the true 76-member `P_super_v3` Fold-0 (**0.951120**). Standalone LGB+δ **0.95011** is weaker than baseline `gb_s0` **0.95039**. Do not promote or re-inject Rung 3.

---

## 2. External / orthogonality recon (local CPU)

| ID | Approach | Result | Why dead | Artifact |
|----|----------|--------|----------|----------|
| E1 | **Sadam TabPFN** pairs (100k) into newton-cholesky stack | OOF AUC **0.94883** · ρ(gb_bag5)≈**0.971** · nested Δ≈**0** | Aligned to SKF(5,rs=42) but **no nested lift** | `local_testbag_tabpfn_recon.json` |
| E2 | Expect Part-9 `testbag` to be orthogonal to `gb_bag5` | Spearman ρ(testbag, bag5) ≈ **0.99993** | Nearly identical ranking; variance reduction only, not new signal | same |
| E3 | goodpjw Part 10 as “CV breakthrough” | LB 0.95239 · **no nested CV** | Public fusion only: rank-mean(Part9, amanatar) | Part10 notebook cells 104–105 |
| E4 | Soft-label / pseudo TE (goodpjw Part 9.3) | Fold0/1 drops vs Part6 booster | Documented reject in upstream notebook | Part10 nb §9.3 |
| E5 | 10-fold members mixed into 5-fold stack (Part 9.2) | Nested↑, public↓ | OOF inflation vs 100% refit test | Part10 nb §9.2 |
| E6 | Digit CNN + more public members (Part 9.4) | Nested↑ · public **down** | Public board did not reward | Part10 nb §9.4 |

---

## 3. Stacking upgrades on `P_super_champ` (0.9521425)

Baseline recipe (kept): Part-8B **74** cols · newton-cholesky **C=0.005** (0.952130) · **+12%** hill-climb · **+8%** `gb_super = 0.5·rk(gb_bag5)+0.5·rk(gb_gpt2_joint)`.

| ID | Approach | Nested | Δ vs 0.9521425 | Verdict |
|----|----------|--------|----------------|---------|
| S1 | Inject **`gb_gpt2_bag5`** as extra newton column | 0.952129 | **−1.36e−5** | **Dead** — collinear with bank / gb_super |
| S2 | Inject `gb_gpt2_bag5` + dual-orth(GLM,GPT2) | 0.952129 | **−1.33e−5** | **Dead** — orth does not rescue |
| S3 | Inject bag5 + `glm_quad` together | 0.952131 | **−1.12e−5** | **Dead** — bag5 drag dominates |
| S4 | Full bag5±orth + glm_quad±orth (78 cols) | 0.952133 | **−9.4e−6** | **Dead** |
| S5 | Add gpt2_bag5+glm_quad on **P7-65 only** (not 8B74) | 0.952099 | **−4.4e−5** vs P_super | **Dead** as a path to beat champ — wrong baseline bank |
| S6 | testbag wire hoping for nested gain | nested **unchanged** (OOF fixed) | 0 | Not a nested lever; test-only variance tweak |

**Not dead (small keep):**

- Swap joint→**`gb_gpt2_bag5` inside 8% `gb_super` only**: 0.9521428 (+0.3e−6).
- Add **`glm_quad` + orth(glm_quad \| GLM)** to 74-bank + bag5-super → **`P_super_v3` 0.9521490** (+6.5e−6). Files: `preds/P_super_v3_{oof,test}.npy`.

Do **not** put `gb_gpt2_bag5` into the newton design matrix again without a new residualization story that beats S1–S4.

---

## 3b. High-EV paradigm loop vs true `P_super_v3` (0.9521490)

Gate: Nested > 0.9521490 **and** Fold-0 stack > 0.9511199. Log: `logs/overnight/high_ev_paradigm_loop.json`.

| ID | Approach | Result | Verdict |
|----|----------|--------|---------|
| H1 | **`glm_cubic_synergy`** (top-1500 2-way + top-400 cubic + KCNT×z + tanh/log tails) | Fold0 solo 0.949995 > glm_quad 0.949937, but Fold0 **stack 0.9511107 < 0.9511199** | **Dead** — solo noise, no stack lift |
| H2 | **`gb_superquad`** from `z_Super=[GLM,glm_quad,gpt2]` + disagreement cols | Fold0 LGB 0.95064 > gb_s0; bag5 OOF 0.95147; **nested 0.9521469 < V3** | **Dead** — Fold0 gate OK, full nested regresses |
| H3 | Marginal BPE unigrams / Rung3 δ-LGB | Already in `gboost.py`; Rung3 fallback artifact | **Dead** (see §1) |

**Not dead (kept into `P_super_v4`):**

- **L1 inject** `masaya_stack` + dhananjay `18_blend_{lm50,pooled2}` → joint nested **0.9522321**.
- **L4 regime** `{r_rare,r_disp,r_tail}×5` families on L1-expanded bank → **`P_super_v4` nested 0.9522570** (`L1_plus_L4_regime`). Files: `preds/P_super_v4_{oof,test}.npy`.

Amanatar rank3 dataset ships **fusion CSVs only** (no 117 raw OOFs). Subh / amanatar / triad CSVs are test-only.

---

## 4. Ops / infra dead ends (ladder run)

| ID | Mistake | Effect | Fix / lesson |
|----|---------|--------|--------------|
| O1 | Score / enrich default `LADDER_SCORE_BS=1536` | Repeated OOMs on dual-prefix GPT-2 | Cap ≤256 with OOM fallback |
| O2 | Progress stub with `auc_sum_llr` before δ24 ready | Watcher killed healthy Rung2 | Gate on `auc_weighted_delta` only |
| O3 | Missing `goodpjw_part7` notebook on pod | Rung0/5 champ path failed; fallback 3-member stack | Sync Part7 nb before “full promote” |
| O4 | Continuing ladder after Rung1 FAIL without enrich metrics | K-curve delayed until bs fix | Enrich is mandatory post-Gate1 logging, not optional |

---

## 5. Still open (not dead — queue)

These are **candidates**, not proven lifts on public LB (NO_SUBMIT):

1. Hold / fuse **`P_super_v4`** (`L1_plus_L4_regime`, nested **0.9522570**) vs public fusions.
2. Amanatar-style **50/50** `rk(P_super_v4)+rk(amanatar)` (test (1−ρ)×1e4 ≈ 27 vs sovereign alone).
3. `0.5·v4 + 0.5·goodpjw_part9` / part10 rank-mean (v4↔part10 distance ≈ 21.5 e4).
4. Pull missing **OOF** for amanatar unbundled/triad members if ever published (dataset currently test-CSV only).

Amanatar 0.95237 formula (for fusion design, not to re-derive blindly):  
`0.50 × unbundled_48_newton + 0.50 × titan_triad_40/30/30`.

---

## 6. Quick “do not retry” checklist

- [ ] Re-fine-tune Fold-0 GPT-2 multi-perm LLR with same 24k/16k gate budget expecting ≥0.949
- [ ] Contrastive LLR as a stack member without beating Gate2 **and** positive ΔS
- [ ] BiTabGPT as currently coded (12 ep residual-on-logit)
- [ ] TabPFN 100k into newton stack expecting nested lift
- [ ] Adding `gb_gpt2_bag5` as a raw/orth column to the 74-member newton bank
- [ ] Treating testbag or Part10 public fusion as nested-CV evidence
- [ ] Dual-prefix score bs=1536 on A5000 without fallback

---

*Last updated: 2026-11-05 · Authors: paradigm ladder + Part10 P_super upgrade session.*
(Superseded chronology: see §11 for 2026-11-12.)

---

## 7. Own TE-GLM / pair-TE rehash (2026-11-06) — DEAD

Re-ran interaction-TE GLM + shallow LGB4 under SKF(5,rs=42) with multi-smoothing TE + screened pairs (`scripts/own_generator_research.py`).

| Member | Solo OOF | ρ vs V4p | Best blend Δ |
|--------|----------|----------|--------------|
| `own_teglm` | 0.949763 | 0.9807 | **+2.5e−7** @α=0.02 (Fold0 Δ **negative**) |
| `own_telgb` | 0.949012 | 0.9741 | **0** (α=0 preferred) |

Same family as Stage 3/4 TE + Rung3 residual trees. Does **not** move V4p. Artifacts: `logs/overnight/own_generator_research.json`, `preds/own_te{glm,lgb}_{oof,test}.npy`.

---

## 8. Nov07 A40 GPU 3-Paradigm Campaign (`A, B, C`) & `matrix_v2` Dead Ends (`2026-11-07`)

Evaluated on `ht590jc0rpb9kz` (NVIDIA A40 48GB) against frozen `P_super_v4p_hc` (`0.9523071` 5-Fold OOF, `0.9513936` Fold-0) using verified `score_candidate_vs_v4p_hc` (`scripts/paradigm_stack_delta_eval.py`).

| ID | Paradigm / Module | Result vs `v4p_hc` (`0.9523071`) | Why Dead / Root Cause | Artifact |
|----|-------------------|----------------------------------|-----------------------|----------|
| G1 | **Paradigm A1/A2/A3** (`1.02M` `torch.cdist` CloneGraph + `orig20k` Splice + `TabR` Cross-Row Attention) | Solo `tabr` = `0.88577` · Solo `xgb` = `0.95205` · Combo Stack $\Delta = \mathbf{-1.58\times 10^{-6}}$ · `A/tabr` `|\Delta\rho| = 0.0545` (FAIL) | Continuous Euclidean / Mahalanobis neighbor distances in synthetic tabular space carry zero information beyond exact discrete key matches (`KMAT_108`); `TabR` solo (`0.8858`) suffers severe OOF-vs-Test rank correlation drift (`|\Delta\rho| = 0.0545 > 0.0020`). | `preds/paradigm_A/` |
| G2 | **Paradigm B1 Solo** (`LLM_Solo` BiMasked + Causal Transformer without GBDT) | Solo OOF = `0.94695` · Test-1 $\Delta = +5.79\times 10^{-6}$ · `|\Delta\rho| = 0.0521` (FAIL) | Standalone neural sequence model over-correlates on full-refit test vs 5-fold OOF (`|\Delta\rho| = 0.0521`). Only useful when its 72 LLR features are fed into CUDA `XGBoost` (`B/xgb_llr`, which passed `|\Delta\rho| = 0.00153`). | `preds/paradigm_B/llm_solo_*` |
| G3 | **Paradigm C1/C2/C3** (`TabM k=32` PLE + Copula QDA + SwiGLU $\text{DAE}_0/\text{DAE}_1$ + Stealth Fraud Specialist) | Solo `tabm` = `0.94287` · Solo `stealth` = `0.94248` · Stack $\Delta = \mathbf{-2.90\times 10^{-6}}$ · `KS = 0.0645` (FAIL) | Continuous copula/DAE reconstruction energy and PLE embeddings cannot resolve discrete key Lookup tables; stealth down-weighting distorted tree calibration (`KS = 0.0645`). | `preds/paradigm_C/` |
| G4 | **Forensic Step 2 Continuous Feature Matrix** (`F_all_new` = 261 cols `[F_A(42), llr72(72), h64(64), copula_dae(57), stepgrid(26)]` fit via 5-fold CUDA IRLS / Newton-Cholesky with `init_score = z_v4p`) | Direct OOF $\Delta \le 0$ across all $L_2 \in [10^2, 10^6]$ · Best Ortho $\Delta = +6.3\times 10^{-7}$ (`0.06 bps`) | Linearly combining 261 mostly-continuous neural/manifold columns on top of `v4p_hc` adds pure estimation variance. | `scripts/forensic_audit_3min.py` |
| G5 | **`matrix_v2` Levers `P3_L3.3, L3.4, L3.5` & `P4_L4.4`** (when evaluated with `score_candidate_vs_v4p_hc` on Fold 0) | Fold-0 Test-1 $\Delta$: `L3.3 = -1.81e-5`, `L3.4 = -1.35e-6`, `L3.5 = -5.70e-7`, `L4.4 = -1.01e-5` | Only `P4_L4.3` (`+2.74e-5`), `P4_L4.2` (`+1.85e-5`), and `P4_L4.5` (`+1.38e-5`) were genuine Fold-0 winners. Mixing the 4 negative levers into a 69-column CUDA XGBoost (`harvest_v5.py`) drowned out the 3 winners (`+3.1e-7`). | `preds/matrix_v2/*_fold0.npz` |

---

## 9. Grandmaster 5-Playbook Orthogonal Tournament Dead Ends (`2026-11-07`)

Executed in `scripts/grandmaster_5playbook_tournament.py` (`logs/overnight/5playbook_tournament_state.json`).

| ID | Playbook | Fold-0 Solo AUC | Fold-0 Test-1 $\Delta\text{AUC}$ | Why Dead / Root Cause |
|----|----------|----------------:|---------------------------------:|-----------------------|
| PB1 | **`PB1_KMAT450`** (8-seed inner EB TE + 96 Bulsah keys + 180 2D/3D hierarchical keys $\rightarrow$ un-anchored Stage-1 `newton-cholesky` GLM `z_Stage1_450` $\rightarrow$ CUDA XGB/LGB) | `0.945782` | `+4.76\times 10^{-6}` (Gate `+1.5e-5` MISS) | Fitting an un-anchored Stage-1 linear GLM across `1,125` sparse 2D/3D columns severely overfitted `fold_a0` (`train AUC = 0.95084` vs `val AUC = 0.94442`), poisoning the `base_margin` of downstream boosters (`0.94578` vs `0.95139`). **Rule:** Never fit an un-anchored wide Stage-1 linear model on hundreds of 2D/3D interaction TE columns to replace `z_champ`. |
| PB2 | **`PB2_GroupbyCtx`** (`372` Unsupervised Contextual Groupby Aggregations `mean/std/Z/rank-pct` across `1.02M` rows) | `0.951395` | `-2.81\times 10^{-7}` | Unsupervised continuous group aggregations (`amount - mean(amount|cat)`) are already spanned by GBDT splits on `KMAT_108 + RAW`; resulting predictions had $\rho = 0.999995$ with `v4p_hc`. |
| PB3 | **`PB3_ResidDist_Cat`** (Out-of-Fold `V4p` Residual Distribution `mean/max/std(r|k)` + Stealth Fraud Key Support + GPU Ordered `CatBoost`) | `0.951378` | `-3.55\times 10^{-6}` | Encoding higher moments of $r_i = y_i - V_{4p}(x_i)$ per key overfits rare keys with 1–2 fraud events; GPU CatBoost with `base_margin` degraded Fold-0 ranking on every subfold (`win_rate = 0.0`). |
| PB4 | **`PB4_Transductive`** (Transductive Semi-Supervised Soft Pseudo-Label TE on `b0 + test` with $\gamma \in \{0.15, 0.30\}$) | `0.946726` | `-1.05\times 10^{-6}` ($\gamma=0.15$) / `-6.09\times 10^{-6}` ($\gamma=0.30$) | Soft pseudo-labels from `v4p_hc` reinforce existing `v4p_hc` biases on unseen keys rather than providing independent supervision, while Stage-1 refit drops solo AUC to `0.9467`. |
| PB5 | **`PB5_AUC_fobj`** (Custom Overlap-Band ROC-AUC Surrogate Objective `fobj` in CUDA XGBoost/LightGBM) | `0.951399` | `+1.57\times 10^{-6}` (Gate `+1.5e-5` MISS) | Custom Gaussian overlap weighting around $p \in [0.004, 0.25]$ preserved baseline calibration ($\rho = 0.999998$), but without new discrete TE features it produced only `+0.016 bps` (`+1.57e-6`). |

---

## 10. Modeling & Stacking Pitfalls Caught in Audit (Never Repeat)

1. **Uncalibrated Rank-Logit `base_margin` Early-Stop Collapse:**
   - Passing uncalibrated `to_z_ranklogit(v4p_hc)` (which has mean `0.0`, i.e. $\bar{p} = 0.50$ instead of true fraud prevalence $\bar{y} = 0.0167$) into `xgb.train(objective="binary:logistic", early_stopping_rounds=50)` causes every row on round 1 to have massive negative gradient $p - y \approx +0.483$, triggering immediate early stopping at `best_iteration = 1`!
   - **Mandatory Fix:** Always Platt-calibrate `base_margin` to true prevalence (`calibrate_logit_to_prevalence`) OR fit `Ridge(alpha=500)` / `reg:squarederror` directly on the calibrated residual $r_i = y_i - \sigma(z_{\text{cal}, i})$.
2. **Non-Negative Weight Search Failure on Anchor-Correlated Candidates:**
   - When a candidate `cand` is built on top of `z_champ` (`\rho \approx 0.998`), its raw score is collinear with `v4p_hc`, and its orthogonal component $r_{\text{orth}} = z_{\text{cand}} - \beta z_{\text{base}}$ has a **negative** optimal sign ($\alpha < 0$, acting as an empirical-Bayes shrinkage / denoising term). Constraining `w >= 0` on un-orthogonalized candidates yields `w = 0.0`.
3. **The `sub_ev4` Overwrite Trap (`\alpha = 1.0` on OOF Proxy):**
   - Tuning `(1 - a) * v4p_oof + a * v5_oof` naturally picks `a = 1.0` because `v5_oof` already contains `v4p_oof` plus the upgrade. Applying `a = 1.0` to `(1 - a) * sub_ev4 + a * v5_test` completely erases `sub_ev4` (`0.95243` LB)!
   - **Mandatory Fix:** Always inject the **pure zero-mean difference vector** $\Delta z_{\text{test}} = z(\text{v5\_test}) - z(\text{v4p\_hc\_test})$ onto $z(\text{sub\_ev4})$ (`sub_v5_harvest_on_ev4_half.csv`, $\rho = 0.999920$).

---

## 11. Nov 12 ATH plateau — v10/v11 stack probes & HiRes Compact-LM (`STRICT NO_SUBMIT`)

Locked public ATH: **cand2 = L(S7) + 0.03613·(G26−MS) → LB 0.95355**.  
Promotion law: inject only after **strict LOFO 5/5** on `eval_S7matched_355` (double-orth to `[L(S7), G26−MS]` + PCHIP). Sources: `logs/overnight/GM_V11_*.md`, `HIRES_LM_FOLD0_CANARY_FAIL.md`, `GM_MATAPANINO_AND_SEED2026_AUDIT.md`.

| ID | Approach | Result | Why dead | Artifact |
|----|----------|--------|----------|----------|
| N1 | **Bank / member swaps** on top of cand2 after removing `(G26−MS)` axis | 189→155 clean pairs → **0** strict 5/5 | Two-seed GPT-2 plateau saturated; residual bank swaps collinear with private Stage-B | `GM_V11_MACRO_LEAP_REPORT.md` |
| N2 | **α∈{0.036, 0.060}** retunes of `(G26−MS)` on cand2 | Both LB **0.95355** | No public lift; do not burn submits on α-grid | same |
| N3 | **v11.1 `orth(LM_trim)`** as live submit | LOFO **4/5** (meanΔ=+0.007e−5), simple **3/5** | Near-miss ≠ promote; **do not submit** `sub_v11p1_slot3_*_NOSUBMIT.csv` | `GM_V11_1_NONLINEAR_STAGEB_REPORT.md` |
| N4 | **v11.1 `G26_denoise` (James–Stein)** | LOFO **0/5** | Denoise ≈ null after double-orth | same |
| N5 | **v11.1 Wilcoxon / Stage-B_v11 nonlinear** | Wilcoxon **1/5**; StageB **3/5** | Fail 5/5 gate | same |
| N6 | **Teamlead modular dual** (all 4 modules) | **0/4** clear independent 5/5 | Same orth wipe after `[S7, G26−MS]` | `GM_V11_TEAMLEAD_MACRO_REPORT.md` |
| N7 | **cand2b** follow-up live probe | Flat vs cand2 | No second ATH | `GM_V10_2` / live probe logs |
| N8 | **HiRes-Compact-LM** (`V≈3232`, `T=72`, δ_cont@odd, value-token CE, no LLR BCE, 30 antithetic, Fold-0 canary) | Fold-0 OOF AUC **0.89416** ≪ L7_f0 **0.94761**; orth **−3.84e−5** | Short CE recipe (12rep/3ep) missed L7 discriminative floor; pod killed (~$0.44) | `preds/hires_lm_5f/`, `HIRES_LM_FOLD0_CANARY_FAIL.md` |
| N9 | **Expect Masaya/goodpjw kernel outputs to ship GPT-2 `.pt` / per-order `.npy`** | Kernel outs = aggregated OOF/test + HTML only | Weights not public; cannot finish G26 test from kernel artifacts | `external/masaya_kernel_out/`, Matapanino audit |
| N10 | **Budget <$3 full natural GPT-2 seed2026 F1–4** (`T≈224`, even pre-tokenized+bf16) | Est. **tens of GPU-h** / multi-$ | Outside hard budget; wait for G26 test upload or Compact-LM canary | `GM_MATAPANINO_AND_SEED2026_AUDIT.md` |
| N11 | **L7 5F rot10 on `smugcqcy2oqzeg` without save fix** | Crash: assert `oof.shape==(139992,10,2)` got `(699960,10,2)` | Save shape bug (full train vs `val_idx`), not training; do not re-run identical saver | `train_compact_lm_5f.py` / L7 status ticks |

### Still open (not dead — blocked on data / gated GPU)

1. **G26 `test_5fold` / `test_f1..f4`** — live dataset still only `oof_5fold` + `test_f0` (stamp 2026-11-12 09:14 UTC). Highest leverage if/when uploaded.
2. **Compact-LM `seed=2026` Fold-0 canary** (`V=1372`, `T=72`, 60rep+8FT, **fixed** `oof[val_idx]` save) ≤~$0.45 — only if F0 ≥ 0.94761.
3. **LM_trim** remains best *near-miss* (4/5); retry only with a new orth story that flips the failing fold — **not** as a blind submit.

### Quick “do not retry” (Nov 12 addendum)

- [ ] Submit any `*_NOSUBMIT.csv` / LM_trim 4/5 / G26_denoise 0/5 candidate
- [ ] Re-run HiRes-Compact-LM with same short CE recipe expecting ≥0.9476 Fold-0
- [ ] Full Matapanino natural GPT-2 5-fold under <$3 wall-clock budget
- [ ] L7 save path that writes full-train `(699960, K, 2)` then asserts val shape
- [ ] Treat Masaya presentation kernel as a source of checkpoints

---

## 12. Nov 12 late — V12.3/V12.4 on Q17+L2 base (`POD TOUCHES: 0`)

Locked public ATH: **L2 = L(Q17 `test_anchor_prob`) + 0.03·unitize(d1_v4) → LB 0.95371**.  
Sources: `logs/overnight/GM_V12_3_LADDER_LIVE_REPORT.md`, `GM_V12_4_FURTHER_LIFT_REPORT.md`, `GM_V12_DAY_SUMMARY_2026-11-12.md`.

| ID | Approach | Result | Why dead | Artifact |
|----|----------|--------|----------|----------|
| V1 | **L3** `P_super_v5` + `stageB_4fam` on L2 (OOF LOFO 5/5) | Public **0.95366** (−5e-5) | OOF lied; axes poison Q17/L2 base | L3 ref `58284886` |
| V2 | **L4** retune `d1=0.028, sov=0.012, G26=0.018` (OOF 5/5 mean+2.62e-5) | Public **0.95370** (−1e-5) | Complementary-fold OOF did not transfer | L4 ref `58285309` |
| V3 | Sameer apex+LM logit mix on L2 | F0 −7…−18e-5 OOF | Kills Fold-0 on this base | `GM_V12_4_*` |
| V4 | Further single-axis injects on SAFE=L2+sov+G26 | 0 singles / 0 pairs clear 5/5 beyond SAFE | Bank saturated for public lift | same |
| V5 | Bare cand2 α / micro-nudges (pre-Q17) | Plateau **0.95355** | Superseded by Q17+d1 | §11 N2/N7 |

### Still open (not dead)

1. Complete **L7 Compact-LM seed=2026 K=10** folds 1–4 (fold0 only local today).
2. Any **new private residual** with d1-like public-transfer DNA — not P_super/sov/G26 retunes.
3. External **0.95385** recipe if/when a public kernel/dataset appears.

### Quick “do not retry” (V12 addendum)

- [ ] Re-submit P_super_v5 / stageB_4fam on Q17 or L2
- [ ] Re-submit sov−wide + G26−MS weight grids on L2 expecting public > 0.95371
- [ ] Sameer LM logit mix on L2
- [ ] More bank blends of existing members without a new private signal

---

*Last updated: 2026-11-12 · ATH L2 0.95371 · V12 ladder + L4 no-transfer.*
