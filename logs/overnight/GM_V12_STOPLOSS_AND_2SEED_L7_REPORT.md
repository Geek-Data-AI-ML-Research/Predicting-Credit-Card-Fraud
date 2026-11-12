# GM V12 Stop-Loss & True 2-Seed L7 Fold-0 Upgrade Audit

**UTC:** 2026-11-12T19:39:30.393286+00:00  
**`NO_SUBMIT=True`** — all 4 daily submissions held  
**Elapsed:** 61.87s

## Step 1 — Rescue & terminate
- Local `oof_0` `[139992, 10, 2]`, `test_0` `[299984, 10, 2]` intact ✓
- Rescued `replica.pt` **344.8 MB**, `fold_0.pt` **1034.4 MB** → `preds/l7_s2026_5f/ckpts/`
- Terminated pod `nnts6ddqh5mprf` (never touched `lheroidmuab1j4`)
- GPU killed before terminate; L7 pod billing **$0.00/hr**
- Note: unrelated account pod `vjez42fl07jtrl` (`p730-gm-1112-191120`) still RUNNING @ $0.59/hr — **not** touched

## Step 2 — Apples-to-apples Fold-0 (K=4 vs 2-seed K=10)
### Flaw in prior Gate 3
- `cand2` / areeb ladder on **test** uses `lm7_test_f0.npy` = **seed=7, Fold-0 only, K=4**.
- Gate 3 residual used `L(s2026_K10) − L(s7_K10)` — **not** matching the K=4 test member.
- Recomputed OLD residual lift: **0.0031e-5** (w=0.04).
- True Move A vs **K=4**: **0.0134e-5** (w=0.12).

### Solo Fold-0 AUCs
| member | AUC |
|---|---:|
| L7_s7_K4 (orders 0..3) | 0.9473314 |
| L7_s7_K10 | 0.9475445 |
| L7_s2026 bayes K10 | 0.9474934 |
| L7_s2026 best red (`R3_ranklogit_mean`) | 0.9475174 |
| 0.5·L(s7_K10)+0.5·L(s2026) | 0.9480483 |
| 0.5·L(s7_K4)+0.5·L(s2026) | 0.9479780 |
| hybrid L7F+s2026 best γ=0.45 | 0.9480456 |
| corr(s2026, s7_K10) | 0.986728 |
| corr(s2026, s7_K4) | 0.985860 |

### Reduction table (s2026 K=10)
| reduction | AUC |
|---|---:|
| R1_mean_llr | 0.9475151 |
| R2_bayes_tau0.5 | 0.9474432 |
| R2_bayes_tau1.0 | 0.9474934 |
| R2_bayes_tau1.5 | 0.9475044 |
| R2_bayes_tau2.0 | 0.9475096 |
| R3_ranklogit_mean | 0.9475174 |
| R2_bayes_default | 0.9474934 |

### Full-stack Fold-0 lift on `eval_S7matched_355[f0]`
Base F0 AUC = **0.9516700**

| move | ΔAUC (e-5) | best w | notes |
|---|---:|---:|---|
| A: L(s2026_K10)−L(s7_K4) | 0.0134 | 0.12 | Compact-LM analog of G26−MS |
| B: 0.65·L(L7F)+0.35·L(s2026)−L(s7_K4) | 0.0438 | 0.15 | 5f seed7 + s2026 vs K4 |
| C: B + orth(LM_trim) | 0.2176 | wB=0.12, wL=0.08 | joint |
| OLD Gate3: L(s2026)−L(s7_K10) | 0.0031 | 0.04 | flawed comparator |

## Step 3 — Q17 / detective / G26 upstream
- G26 `test_5fold` live: **False**
### goodpjw2008/s6e11-target-first-gpt2-seed-2026
```
name                         size  creationDate                
------------------------  -------  --------------------------  
README.md                    1343  2026-11-12 09:14:45.794000  
gpt2_s2026_oof_5fold.npy  5599808  2026-11-12 09:14:46.075000  
gpt2_s2026_test_f0.npy    2400000  2026-11-12 09:14:45.915000  

```
### amanatar Q17 (prior audit)
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
### aphoristicartist/s6e11-detective-weights
```
name              size  creationDate                
-----------  ---------  --------------------------  
config.json        961  2026-11-10 23:11:40.103000  
```
### detective-weights
```
name              size  creationDate                
-----------  ---------  --------------------------  
config.json        961  2026-11-10 23:11:40.103000  
model.pt     497805283  2026-11-10 23:11:51.122000  

```
### Recent s6e11 kernels (dateCreated)
```
ref,title,author,lastRunTime,totalVotes
markbarney/s6e11-what-we-tried-and-what-did-not-work,S6E11 | What We Tried and What Did Not Work,Mark Barney,2026-11-12 19:26:19.613000,0
modansaif/s6e11-fraud-fusion-reproduction-credited,S6E11 Fraud Fusion Reproduction (credited),SAIF MODAN,2026-11-12 18:36:26.740000,0
bawanthabeliwaththa/the-honest-stack-cv-lb-and-the-0-9533-trap,"The Honest Stack: CV, LB, and the 0.9533 Trap",Bawantha Beliwaththa,2026-11-12 17:38:52.200000,0
knightfork/s6e11-top-15-cpu-blend-eda-lb-0-95357,S6E11 | Top 15 CPU Blend + EDA | LB 0.95357,Knight Fork,2026-11-12 13:52:53.687000,0
asteriospanoras/s6e11-how-much-of-the-public-top-is-real,S6E11 | How Much of the Public Top Is Real?,Asterios Panoras,2026-11-12 14:46:56.213000,1
deepakkumbhar16/beginners-friendly-blending-gb
```

## Staged NOSUBMIT candidates
- `sub_v12_cand1_MoveA_twoseed_K10_minus_K4_0.10e4_NOSUBMIT.csv` — 1−ρ(cand2)=0.1044e-4, w=0.0249 **NOSUBMIT**
- `sub_v12_cand2_MoveB_hybrid_minus_K4_0.10e4_NOSUBMIT.csv` — 1−ρ(cand2)=0.1026e-4, w=0.0399 **NOSUBMIT**

**HOLD all 4 daily submissions.** Do not upload.

---
*Generated by `scripts/gm_stoploss_and_f0_k4_to_2seed_k10_audit.py`*
