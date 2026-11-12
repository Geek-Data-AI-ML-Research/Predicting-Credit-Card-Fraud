# S6E11 Day Summary — 2026-11-12

**POD TOUCHES: 0** (hard stop after earlier incident)

## Scoreboard (our team)

| Milestone | Public LB | Notes |
|---|---:|---|
| Start-of-day ATH | 0.95355 | cand2 = S7 + 0.036·(G26−MS) |
| After Q17 pure anchor (L1) | 0.95370 | amanatar rank1 bundle `test_anchor_prob` |
| **Day ATH (L2)** | **0.95371** | +0.03·unitize(d1_v4) on Q17 anchor |
| L3 P_super+stageB | 0.95366 | OOF lied; ban axes |
| L4 sov+G26 retune | 0.95370 | OOF lied; no ATH change |
| Public #1 (external) | 0.95385 | no public recipe recovered |

Best submission on account remains **L2** (Kaggle keeps best public). Team place ~4th at **0.95371**.

## What worked
1. Submit pure Q17 `anchor_095370` / `test_anchor_prob` when external LB is verified above ATH.
2. Tiny `d1_v4` (private Stage-B residual) on that anchor at **w=0.03** — only private inject that transferred.
3. Zero-pod local forensics + Kaggle CLI recon.

## What failed / banned
| Axis | Why |
|---|---|
| P_super_v5 + stageB_4fam on L2 | OOF 5/5 → public −5e-5 |
| sov−wide + G26−MS on L2 (even 5/5 OOF) | public −1e-5 vs ATH |
| Sameer apex+LM logit mix on L2 | F0 collapse OOF |
| Bare cand2 micro-nudges earlier | plateau at 0.95355 |

## Locked next-day plan
1. Do **not** re-blend Q17 + d1 + public sov/G26/P_super without new signal.
2. Priority: finish **L7 Compact-LM seed=2026 K=10** for folds 1–4 (only fold0 local today); re-test pure two-seed residual with subfold LOFO.
3. Recon for any new public 0.9538x artifact before burning quota.
4. Hold submits unless gate: LOFO 5/5 **and** ingredient family with prior public transfer (d1-like), not banned families.

## Key artifacts
| Path | Role |
|---|---|
| `submissions/sub_v12p3_slot2_anchor095370_plus_d1v4_5of5_LIVE.csv` | **ATH 0.95371** |
| `external/nov12_late_recon/amanatar_q17/rank1_q17_streams.npz` | Q17 bank |
| `preds/gm_v4_polish/d1_v4_{oof,test}.npy` | private residual that worked |
| `preds/l7_s2026_5f/{oof,test}_0.npy` | fold0 only Compact-LM s2026 |
| `logs/overnight/GM_V12_3_LADDER_LIVE_REPORT.md` | ladder table |
| `logs/overnight/GM_V12_4_FURTHER_LIFT_REPORT.md` | post-ATH research + L4 |
| `scripts/gm_v12_2_hardened_095385_recon_and_anchor_inversion.py` | zero-pod recon script |
