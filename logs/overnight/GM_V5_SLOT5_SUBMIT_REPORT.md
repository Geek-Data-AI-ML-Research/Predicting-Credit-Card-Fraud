# GM V5 Overnight — Slot5 Submit Report

**Submitted:** `sub_gm_v5_overnight_slot5.csv`  
**Ref:** `58227983`  
**Message:** `GM V5 overnight Slot5: S5_positive_purge + d1_v5_super axis @ 2.00e-4 (NO bad_S3/ldr)`  
**Public LB:** **0.95328**  
**Status:** COMPLETE · 9 submissions remaining today

---

## Verdict

**−7e−5 vs current best (0.95335).** Overnight Slot5 did **not** improve the public LB. The move is a small regression relative to Ladder S5, consistent with (1) stripping the positive `bad_S3` / `bad_ldr` components that Ladder S5 used, and (2) adding a near-null `d1_v5_super` axis (OOF AUC ≈ 0.526).

---

## Public LB context

| Submission | Public | Δ vs best |
|---|---:|---:|
| **GM V5 overnight Slot5 (this)** | **0.95328** | **−7.0e−5** |
| Ladder S5 macro finale (best) | 0.95335 | — |
| Ladder S4 adaptive | 0.95334 | −1.0e−5 |
| Ladder S1 / S2 | 0.95333 | −2.0e−5 |
| Strike 9.5 | 0.95332 | −3.0e−5 |
| Ladder S3 | 0.95329 | −6.0e−5 |

---

## What was submitted

Construction (Phase 5 staging):

1. Base = `z_S5_positive_purge` (Ladder-S5 coords with **positive** `c_bad_S3` / `c_bad_ldr` zeroed; negative `c_bad_tree` kept).
2. Axis = `d1_v5_super` (NN residual on `z_ref_old`, Method-B cleaned, orth to `B_bad`).
3. Step = **2.00e−4** Spearman distance (largest overnight ladder rung).
4. Post-check: no positive reintroduction of `bad_S3` / `bad_ldr`.

File checks: n=299984, mean p=0.016700, identical to staged `slot5_p.npy`.

### Test-space telemetry

| Metric | Value |
|---|---|
| 1−ρ(z_slot5, z_S5_purged) | 2.00e−4 |
| cos(Δ, bad_S3 / ldr / tree / soft) | ~0 (purge lock held) |
| Spearman vs Ladder S5 probs | ρ=0.999799 · 1−ρ=2.01e−4 |
| Spearman vs Ladder S4 | ρ=0.999637 · 1−ρ=3.63e−4 |

---

## Overnight factory → why the axis was weak

### New Stage-B families (OOF)

| Family | OOF AUC |
|---|---:|
| lgb_stageB_extratrees_bag6 | 0.9523231 |
| xgb_stageB_monotone_lm_bag6 | 0.9523114 |
| cat_stageB_bag9 | 0.9523050 |
| xgb_stageB_tail_specialist_bag6 | 0.9523021 |
| lgb_stageB_colsub_deep_bag6 | 0.9523012 |
| mlp_stageB_bag5 | 0.9517065 |
| z_ref_old (base) | 0.9526612 |
| **d1_v5_super (residual)** | **0.5260154** |

Families sit ~3–4e−4 **below** `z_ref_old` OOF — useful diversity candidates, but nested non-negative blend on `z_ref_old` / `z_v8` selected almost nothing:

- `on_z_ref_old`: pos **0/5**, only `lgb_gpt2_3ep_fixed` @ w≈0.002
- `on_z_v8_old`: pos **0/5**, same family @ w≈0.004

So `d1_v5_super` is essentially a tiny GPT2-LGB residual → near-noise axis (AUC 0.526). Slot5 therefore ≈ **purged S5 + 2e−4 of noise-orthogonal step**.

### Phase-1 combo (kept, not the submit base)

- δ_v4_combo_NN vs z_ref_old: AUC 0.9526829, Δe5 ≈ [−0.49, +6.31, +0.20, +0.59, +2.88], **4/5**
- S5 raw coords had `c_bad_S3=+2.27`, `c_bad_ldr=+1.85` (helpful on prior LB S5=0.95335); overnight **purged those positives**.

---

## Interpretation

1. **LB drop −7e−5** aligns with distance from Ladder S5 (~2e−4 rank move) after removing poison-axis components that the public LB had rewarded.
2. **Overnight GBDT/MLP bag did not unlock a positive nested residual** on the old refs under w≥0 — portfolio gate failed (0/5).
3. Slot5 was the right “max overnight” probe; result says: do **not** climb the purged+d1_v5 ladder. Hold **0.95335** Ladder S5 as champion.

---

## Recommended next (not submitted)

- Keep Ladder S5 (`0.95335`) as public best.
- If probing further: submit **un-purged** S5 ± a residual that is **5/5 positive** OOF (not this d1_v5), or test overnight Slot1 (0.25e−4) only if needing a tiny purge-safety check — expect little upside.
- Re-open family selection: allow signed / fold-gated weights, or build residuals **orthogonal to z_ref but correlated with Strike e1** before another submit.

`NO_SUBMIT` overnight artifacts remain under `preds/gm_v5_overnight/` and `logs/overnight/GM_V5_OVERNIGHT_8HR_REPORT.md`.
