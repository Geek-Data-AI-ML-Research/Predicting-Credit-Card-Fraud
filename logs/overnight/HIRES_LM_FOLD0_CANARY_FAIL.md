# HiRes-Compact-LM Fold-0 Canary — FAIL (pod terminated)

**UTC window:** 2026-11-12 ~14:56–15:50  
**Pod:** `aix28sdso639me` (NVIDIA A40 Secure, CA-MTL-1, **$0.59/hr**) — removed after canary fail  
**Approx spend:** ~45 min × $0.59 ≈ **~$0.44** (under $3.50 hard cap; Fold-0 gate intended ~$0.35)

## Phase 0 — PASS ($0)
- Script: `scripts/train_hires_compact_lm_5f.py`
- Causal residual: δ_cont only on odd value positions (verified)
- Save shapes locked: `oof=(n_val,10,2)`, `test=(299984,6,2)`; wrong full-train shape rejected
- Real cache: `V=3232`, `NTOK=3272`, `T=72`, 35 feats
- Artifact: `preds/hires_lm_5f/PHASE0_PASS.json`

## Phase 1 — PASS
- Ignored foreign pod `lheroidmuab1j4` (not on this account / empty list)
- Provisioned A40 @ $0.59/hr (4090/A40 community stock unavailable under $0.69 at create time)
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- Watchdogs: local `sleep 300m` + in-pod 300m self-kill

## Phase 2 — Fold-0 Canary FAIL
Synced locally:
- `preds/hires_lm_5f/oof_0.npy` `(139992, 10, 2)`
- `preds/hires_lm_5f/test_0.npy` `(299984, 6, 2)`
- `preds/hires_lm_5f/fold0_canary.json`

| Metric | Value | Gate |
|--------|------:|------|
| Fold-0 OOF AUC (mean ΔLL) | **0.89416** | must beat L7_f0 **0.94761** |
| beat_L7_f0 | false | |
| Orth ΔF0 vs cand2-proxy | **−3.84e-5** | need > +0.5e-5 |
| pass | **false** | |

Per-order AUCs ~0.893–0.895 (stable across 10 antithetic orders; not a single bad perm).

### Runtime knobs used (after pace cut for budget)
- `REP_EPOCHS=12`, `EPOCHS=3`, `BATCH=128`, `SCORE_BS=160`
- Replica ~6 min; Fold-0 train ~20 min; OOF score ~8 min; TEST score ~10 min

### Likely failure modes (next iteration)
1. Generative CE alone may under-separate fraud at 1.67% with only 3 FT epochs / no label-aware head.
2. Scoring uses full-sequence LL; consider value-token-only LL (match train mask) for Bayes score.
3. `V=3232` below target band 3800–4096 — bump quantile bins / engineered cols.
4. Orth gate path on pod missed local `z_ref`/`d1` paths (local re-eval still failed AUC + orth).

## Status
- **Pod terminated. No Folds 1–4. No submission.**
- ATH cand2 `0.95355` unchanged / NO_SUBMIT holds.
