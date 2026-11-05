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
