# GM V14 — Complete Hardened 5-Fold L7_s2026 Report

**STRICT NO_SUBMIT=True** · no live Kaggle submit from this pipeline.  
**Updated:** 2026-11-13 ~17:58 UTC (local 19:58).

## Live status (folds 1–4 resume)

| Field | Value |
|---|---|
| Pod | `li1uaxfpuiqf59` · RTX 4080 SUPER · **$0.28/hr** |
| SSH | `ssh root@87.197.127.62 -p 41134` |
| Python | **PID 257** — do not signal/restart |
| Watchdog | `sleep 21600` (360 m from ~15:31 UTC; kills bash **254** only if `ALL_FOLDS_DONE` missing) |
| Sync | `tmp/v14_sync_monitor.sh` → stage `tmp/l7_s2026_v14/preds/` → `preds/l7_s2026_5f/` (60 s; size-stable **or** next-fold gate; delete pod when `oof/test_{1..4}` all local) |
| Workdir rule | Use **repo `tmp/`** on pics volume — system `/tmp` often full |

### Fold AUC / wall (Bayes OOF)

| Fold | Seed | OOF AUC | Train | Score | Wall | Local `oof/test` |
|---|---|---|---|---|---|---|
| 0 | 2026 | **0.947493** | 70.3 m | 26.0 m | (prior run) | ✅ |
| 1 | 2027 | **0.949578** | 48.9 m | 26.8 m | **75.7 m** | ✅ (17:29) |
| 2 | 2028 | **0.954835** | 48.9 m | 26.8 m | **75.7 m** | ✅ (18:47) |
| 3 | 2029 | **0.949330** (OOF) | 50.3 m | **scoring now** | — | ⏳ |
| 4 | 2030 | — | — | — | — | ⏳ |

Observed pace ≈ **75.7 m/fold** (F1–F2). F3 test-score in flight (~13 GB VRAM / 100% GPU). ETA F4 finish ≈ **~1.5 h** from this update.

Fold-1 canary: solo AUC **0.94958 > 0.9485** — **PASS** (see V14.2).

---

## Step 1 — Recon

| Item | Status |
|---|---|
| Active pods (pre-launch) | **0** (`list-pods` empty; `p730` / `vjez42fl07jtrl` gone; foreign `lheroidmuab1j4` never touched) |
| Local `oof_0.npy` | `(139992, 10, 2)` float32 · 11.2 MB |
| Local `test_0.npy` | `(299984, 10, 2)` float32 · 24.0 MB |
| `ckpts/replica.pt` | 344.8 MB |
| `orders_10` / `master_perms_seed2026` | `(10, 24)` int64 — **NC=24**; bit-exact `RandomState(2026)` |
| `oof/test_{1,2}` | **ON DISK** locally |
| `oof/test_{3,4}` | pending |

## Step 2 — Kaggle upstream

| Check | Result |
|---|---|
| `goodpjw2008/s6e11-target-first-gpt2-seed-2026` | Still only F0 test dump — **no** full 5-fold test |
| Public board | Cap still **0.95388** (areebbb bold) |

## Step 3 — Preflight / parity (HARD GATE)

Script: `scripts/gm_v14_complete_hardened_pipeline.py`

| Flaw fix | Enforcement |
|---|---|
| K_oof=K_test=10 | Resume scores all 10 orders on OOF **and** Test |
| no `_orig_mod.` | `USE_COMPILE=0`; native SDPA |
| bf16 autocast | GPU scoring `torch.bfloat16` |
| orders | `orders_10.npy` bit-exact vs `master_perms_seed2026.npy` |
| disk | Resume saves fp16 model-only `fold_latest.pt` |
| parity | Fold0 128-row CPU recompute |

```
rho(llr)=0.99997335
max|Δ|=0.064293  (CPU amp mad_lim=0.10)
pass=True · replica.pt strict=True OK
```

## Step 4 — Pod resume — IN PROGRESS (see Live status)

Stock note: 4090 / L40S / A100 / A40 were **NONE** at create; Community 4080 SUPER placed first.

## Step 5 — Assemble + ATH 4-bar — PENDING

Blocked until `oof/test_{0..4}` local and pod deleted ($0.00/hr).

Then:

- `l7_s2026_oof_5fold_raw.npy` `(699960, 10, 2)`
- `l7_s2026_test_5fold_raw.npy` `(5, 299984, 10, 2)`
- `R_bayes_5f` / `R_ranklogit_5f`
- Moves 1–3 on ATH 0.95388 + NOSUBMIT staging

## Related reports

| Doc | Content |
|---|---|
| `GM_V14_1_WATCHDOG_CANARY_STEP5_PREP.md` | Canary gates + L7F/LM fold refs |
| `GM_V14_2_FOLD1_VERIFICATION.md` | F1 two-seed (+4.47 bps) + F0+F1 ATH Moves 1–3 (weak) |

### V14.2 headline (NO_SUBMIT)

- F1 Bayes=`0.949578` · 2-seed vs L7F **`+4.47` bps** (F0 **`+5.04` bps** ≡ prior “+50.4e-5”)
- F0+F1 ATH stack: Move1 2/2 but sf10 6/10; Move2/3 not nested-stable — **no Stage-5 claim yet**

## Guardrails

- **NO_SUBMIT=True**
- Foreign pod `lheroidmuab1j4` never touched
- Zero live Kaggle submit from V14
- Never touch / restart PID **257** while training
