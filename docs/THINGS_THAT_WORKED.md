# S6E11 Things That Worked

Catalogue of approaches that **cleared a measured gate** (nested OOF, Fold-0 canary, or public LB) and should be **reused as building blocks**, not rediscovered.  
Companion to `docs/DEAD_ENDS.md`. Policy: prefer **STRICT NO_SUBMIT** diagnostics; promote only after the stated gate.

Locked CV: `StratifiedKFold(5, shuffle=True, random_state=42)`.

---

## 0. Current ATH lock (2026-11-12 late)

| Item | Detail |
|------|--------|
| **Public ATH** | **0.95371** (L2) |
| Formula | `L(Q17 test_anchor_prob) + 0.03 · unitize(d1_v4)` |
| Prior plateau | cand2 = `L(S7) + 0.03613·(G26−MS)` → **0.95355** (superseded) |
| File | `submissions/sub_v12p3_slot2_anchor095370_plus_d1v4_5of5_LIVE.csv` |
| Gate for next inject | Strict LOFO **5/5** vs L2 **and** ingredient family with prior public transfer (d1-like); ban P_super/stageB/sov+G26 retunes on this base |
| Submits | Day quota **0** after L4; hold until new private signal |

Artifacts: `logs/overnight/GM_V12_3_LADDER_LIVE_REPORT.md`, `GM_V12_DAY_SUMMARY_2026-11-12.md`.

---

## 1. Early campaign (Nov 2) — public OOF ensembles

| Item | Detail |
|------|--------|
| Public OOF bank + rank/logit hillclimb | CV 0.95111 → public **0.95163** (first breakout) |
| Core rule / Stage 1 split proof | Adv AUC ≈ 0.50080; locked SKF(5, rs=42) |
| Stage 2 promotion gate | Reject unless OOF↑ **and** ≥4/5 folds↑ |
| TE control family | `te_xgb` / `te_xgb_smooth50` (~0.9493–0.9494 OOF) |
| Stage 5 shallow leaves | Best TE-LGBM at leaves=3 |

See also: `docs/AUDIT_REPORT.md` §What worked (Nov 2 snapshot).

---

## 2. Stacking ladder that compounded (Nov 5–7)

| Milestone | Nested / note | Keep |
|-----------|---------------|------|
| Part-8B 74-col newton-cholesky + hill + `gb_super` | `P_super_champ` ≈ 0.95214 | Bank recipe |
| `glm_quad` + orth into bank | `P_super_v3` **0.952149** | Keep |
| L1 public LM members + L4 regime | `P_super_v4` **0.952257** | Keep |
| `+12%` hill + `4%` GB_super multi-seed | `P_super_v4p_hc` **0.9523071** | **Step-0 champ** |
| Discrete residual heads (P4 L4.2/L4.3/L4.5 + `B_xgb_llr`) | `P_super_v5_harvest` **0.9523401** | Orth inject pattern |
| Wide + megayak + harvest residual | `P_super_v6_grandmaster` nested **0.9525275** (5/5 vs wide) | Residual-on-public formula |

**Generator law (validated):** after `KMAT_108`, continuous manifold methods add ≈0; lift comes from **discrete TE / digit / step-grid / token LLR**.

---

## 3. Target-first LMs that actually scored

| Member | OOF / note | Role |
|--------|------------|------|
| **Masaya GPT-2** (natural text, K=10) | Pooled **0.950027**; Fold0 **0.94888** | `MS` seed in ATH |
| **G26 GPT-2 seed2026** | OOF **0.950140** (5-fold); test **f0 only** public | `G26` seed; `G26−MS` orth axis |
| **0.5·G26 + 0.5·MS** | OOF **0.950439** | Two-seed rung |
| **Compact-LM L7** (`V=1372`, `T≈72`, seed7) | Band **0.9495–0.9499**; Fold0 ≈ **0.94761** | Public `goodpjw2008/s6e11-compact-lm-seed7`; second-seed candidate via seed2026 |
| **megayak compact** | Compact token format; NATURAL≈224 vs compact≈66–72 | Design source for L7 |

**Engineering that worked (Matapanino audit):**

- Pre-tokenize `(column, value)` BPE pieces once (`RowTokens`) — **no HF tokenizer in hot loop**.
- Target-first Bayes: score under True/False, log-sum-exp over orders + prior.
- Compact path: 1 learned col-token + value pieces + used-vocab LM head (`V=1372`) → trainable on A40 in hours, not days.
- Phase-0 **save shape unit test** (`oof=(n_val,K,2)`, `test=(299984,K_test,2)`) — mandatory before GPU (prevents `smugcqcy2oqzeg`-class crashes).

---

## 4. Nov 12 V12 ladder wins (zero-pod)

| Item | Detail |
|------|--------|
| **Pure Q17 anchor** (`test_anchor_prob` / `anchor_095370`) | Public **0.95370** (L1) — submit verified external apex as-is |
| **+0.03·unitize(d1_v4)** on that anchor | Public **0.95371** (L2) — only private residual that transferred (~55% of OOF) |
| Zero-pod recon + Kaggle CLI | Recovered Q17 bank without GPU; `kaggle competitions submit --competition` |

---

## 5. Ops / process wins

| Practice | Why it worked |
|----------|----------------|
| Fold-0 canary before Folds 1–4 | Killed HiRes at ~$0.44 instead of full 5F |
| Hard pod watchdog (`sleep 300m` + spend cap) | Bound runaway GPU cost |
| Ignore foreign / blacklisted pods (`lheroidmuab1j4`) | No accidental spend |
| **Hard zero-pod rule** when ordered | Prevents spend / state drift during local-only days |
| `NO_SUBMIT` + staged `*_NOSUBMIT.csv` | Preserves daily slots; 4/5 near-misses do not burn |
| Double-orth before LOFO | Stops false “5/5” from soft-orth optimism |
| Platt / prevalence-calibrated `base_margin` | Avoids early-stop collapse on rank-logit offsets |
| Inject **Δz residual**, never α=1 overwrite of LB anchor | Preserves ATH backbone |
| Trust **public transfer history**, not OOF 5/5 alone | L3/L4 both 5/5 OOF → public red |

---

## 6. Highest-leverage next actions (reuse these wins)

1. Finish **Compact-LM seed2026 K=10 folds 1–4** (fold0 local only today); re-test pure two-seed residual with subfold LOFO vs L2.
2. Recon for any new public **0.9538x** artifact before burning quota.
3. Do **not** re-blend Q17+d1 with P_super / stageB / sov+G26 without a new family.
4. Hold submits unless gate: LOFO 5/5 **and** prior public-transfer family (d1-like).

---

*Last updated: 2026-11-12 · ATH 0.95371 · Companion to `docs/DEAD_ENDS.md`.*
