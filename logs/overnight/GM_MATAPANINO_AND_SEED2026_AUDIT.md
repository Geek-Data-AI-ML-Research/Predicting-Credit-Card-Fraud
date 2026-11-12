# GM Matapanino GPT-2 + G26 Upstream + Compact-LM seed2026 Audit

**Clock (UTC):** 2026-11-12T16:01:25.564893+00:00  
**`NO_SUBMIT=True`** — **all 4 daily submissions remain in reserve**  
**HOLD:** do **not** submit `sub_v11p1_slot3_pure_LMtrim_plus_G26denoise_0p10e4_NOSUBMIT.csv`  
(LM_trim LOFO **4/5** on S7m / **3/5** simple; G26_denoise **0/5** — promotion law forbids burn)

Runtime: 22.4s · $0.00 GPU

---

## Executive verdict

1. **Masaya kernel is a presentation wrapper**, not the trainer. Kernel outputs contain **OOF/test `.npy` + CSV + evidence HTML only** — **zero** `.pt` / `.bin` / `.safetensors` checkpoints and **zero** per-order score arrays.
2. **Verbatim training/scoring code lives in** `external/matapanino_gpt2_repo/` (cloned from `Matapanino/s6e11-gpt2-target-first`). It **already pre-tokenizes** each `(column, value)` BPE piece once via `RowTokens` — HF tokenizer is **not** called inside the train/score hot loop.
3. **G26 upstream unchanged for 5-fold test:** still only `gpt2_s2026_oof_5fold.npy` + `gpt2_s2026_test_f0.npy`. **No** `gpt2_s2026_test_5fold.npy` / `f1..f4` in the last 3 hours.
4. **Under <$3:** Architecture B (proven Compact-LM `V=1372,T=72`, seed=2026 Fold-0 canary ≈ **$0.25–0.45**) is the only path that fits the budget for a gated 5-fold second seed. Architecture A (full GPT-2 natural `T≈224`) remains **tens of GPU-hours** even after pre-tokenize+bf16; finishing G26 test folds 1–4 is the higher-leverage A-path if/when uploaded.

---

## Step 1 — Verbatim GPT-2 assets

### `external/gpt2_target_first_09501/`
- Kernel id: `masayakawamata/s6e11-gpt-2-target-first-classifier-cv-0-9500`
- Dataset source: `['masayakawamata/s6e11-gpt2-target-first-oof']`
- `enable_gpu`: `False` (CPU notebook — loads saved predictions)
- Code cells: **14** (load member → show recipe/AUC → write submission)
- Headline from markdown: CV **0.9500**; training cited ~**60 GPU hours**; notebook itself runs in under a minute on CPU

### Kernel output `masayakawamata/s6e11-gpt-2-target-first-classifier-cv-0-9500`
Downloaded to `external/masaya_kernel_out/`:
```
evidence/01_recipe.html
evidence/02_row_text.html
evidence/03_fold_auc.html
evidence/04_feature_orders.html
evidence/05_public_members.html
evidence/06_first_fold_decision.html
fold_ids.npy
manifest.json
oof.csv
oof.npy
s6e11-gpt-2-target-first-classifier-cv-0-9500.log
submission.csv
test.npy
```
- **Weight / checkpoint files:** **NONE**
- **Per-order `.npy`:** **NONE** (only aggregated `oof.npy` / `test.npy` log-odds)
- Manifest CV: **0.950027203793** (fold AUCs in `manifest.json`)

### `goodpjw2008/s6e11-lm-ladder-two-gpt-2-seeds-lb-0-95354` kernel output
```
s6e11-lm-ladder-two-gpt-2-seeds-lb-0-95354.log
submission.csv
```
No GPT-2 weights; ladder blend artifacts only.

### `external/matapanino_gpt2_repo/` (Matapanino / masaya training code)
Clone: `https://github.com/Matapanino/s6e11-gpt2-target-first` → present=True  
Files: `.gitignore, LICENSE, README.md, baseline_xgb.py, common.py, copy_finished_stages.sh, make_folds.py, make_predictions.py, make_queue.py, requirements.txt, run_queue.py, score_rows.py, stack_lr.py, tabtext.py, train_classifier.py, train_replica.py`

#### Exact serialization / `T`
- Natural target-first string: `is_fraud is {True|False}, col is value, ...<|endoftext|>`
- Values = **exact CSV strings** (never cast to float); blank → `None`
- **Sequence length `T` (BPE):** megayak measures natural rows at **≈224 GPT-2 tokens** (only ~40.5 of which are value-carrying). Matapanino uses this full natural format (not the compact `T=72` rewrite).
- Compact contrast (Architecture B): **`T≈66–72`** with 1 learned col-token + value BPE pieces; **`V=1372`** used-token LM head.

#### Tokenizer placement (bottleneck answer)
- **`RowTokens` pre-tokenizes once** per distinct `(column, value)` into integer piece lists; each row stores factorized codes.
- Train (`train_classifier.py`) and score (`score_rows.py`) **concatenate integer pieces** (`body_ids` / prefixes) — **no HF `tokenizer()` inside the step/score loop**.
- Proven by `check_matches_tokenizer()` (pieces == whole-string tokenize).
- Residual bottlenecks:
  1. **Attention `O(T²)` with T≈224** (≈10× tokens vs compact T=72 → ~100× attn FLOPs/row vs compact if same width — GPT-2-small width identical, so attention dominates).
  2. **Full 50,257-way `lm_head` softmax** (score_rows chunks `lm_head_tokens=4096` in fp32 for memory).
  3. **Orders:** post uses **K=30**; shared S6E11 predictions used **K=10**. Scoring ≈ linear in K and in rows×labels(2).
  4. Classifier batch remains **32** (recipe); score default `--batch-sequences 256` (memory knob; length-bucketed, no pad).
  5. Default **score precision fp32** (post); `bf16`/`tf32` optional but add rounding noise per README.

#### Recorded S6E11 wall (masaya presentation)
| Stage | GPU-hours (summed jobs) |
|------|------------------------:|
| Replica | 5.07 |
| Classifiers (5×8 ep) | 55.22 |
| Training total | **60.29** |
| Scoring (incl. extra OOF orders on early folds) | **61.62** |

Pooled OOF **0.950027**; G26 seed2026 OOF **0.950140** (prior recon).

---

## Step 2 — Live upstream (G26 + recent)

### `goodpjw2008/s6e11-target-first-gpt2-seed-2026` (G26)

Live file list (CLI):
```
name                         size  creationDate                
------------------------  -------  --------------------------  
README.md                    1343  2026-11-12 09:14:45.794000  
gpt2_s2026_oof_5fold.npy  5599808  2026-11-12 09:14:46.075000  
gpt2_s2026_test_f0.npy    2400000  2026-11-12 09:14:45.915000
```

- **`gpt2_s2026_test_5fold.npy` / `f1..f4`: ABSENT**
- Present today: `gpt2_s2026_oof_5fold.npy`, `gpt2_s2026_test_f0.npy`, `README.md`
- Dataset `lastUpdated` from user list: **2026-11-12 09:14:44 UTC** (not within the last 3 hours as of audit clock)
- `kaggle datasets status ...` → API 404 (endpoint not available on this CLI version); file list is authoritative

### goodpjw2008 datasets / kernels (abbrev)
```
ref                                              title                                                    size  lastUpdated                 downloadCount  voteCount  usabilityRating  
-----------------------------------------------  -------------------------------------------------  ----------  --------------------------  -------------  ---------  ---------------  
goodpjw2008/s6e11-oof-bank                       S6E11 OOF Bank: 70 Members, Same 5 Folds            259968203  2026-11-03 16:00:24.543000           2321          0  0.64705884       
goodpjw2008/s6e11-fraud-series-artifacts         S6E11 Fraud Series Artifacts                        135687080  2026-11-07 23:52:13.130000           1513          6  0.4375           
goodpjw2008/rsna-knee-2-5d-convnext-reader       RSNA Knee 2.5D ConvNeXt reader                      353235670  2026-10-05 00:09:57.943000          10272         44  0.6875           
goodpjw2008/s6e10-goodpjw2008-extra-members-oof  S6E10 goodpjw2008 extra members OOF                 444701211  2026-10-30 00:58:50.340000           6706          3  0.29411766       
goodpjw2008/s6e10-tabpfn-member                  S6E10 TabPFN-3.5 full-context member (OOF + test)    52874279  2026-10-04 06:03:29.540000            979          3  0.5294118        
goodpjw2008/wear-hasca-part2-gpu-outputs         WEAR@HASCA Part 2: GPU-stage outputs of our run    1166759225  2026-10-08 22:39:44.500000             68          0  0.5294118        
goodpjw2008/lmsys-gemma2-9b-lora                 LMSYS Gemma-2-9B LoRA adapter                        89852346  2026-05-22 01:23:44.340000             28          1  0.25             
goodpjw2008/s6e9-aux-members                     S6E9 aux members: LR-margin XGB, MLP, LGBM           74482029  2026-09-30 22:33:19.330000      
```
```
ref                                                        title                                               author       lastRunTime                 totalVotes  
---------------------------------------------------------  --------------------------------------------------  -----------  --------------------------  ----------  
goodpjw2008/s6e11-lm-ladder-two-gpt-2-seeds-lb-0-95354     S6E11 LM Ladder | Two GPT-2 Seeds | LB 0.95354      goodpjw2008  2026-11-12 09:17:28.553000          19  
goodpjw2008/s6e11-part-12-wide-stack-2nd-lm-lb-0-95298     S6E11 Part 12 | Wide Stack + 2nd LM | LB 0.95298    goodpjw2008  2026-11-08 00:06:34.660000          53  
goodpjw2008/house-prices-5-seed-stack-bag-11838            House Prices | 5-Seed Stack Bag | 11838             goodpjw2008  2026-05-19 03:30:21.720000           4  
goodpjw2008/s6e10-anatomy-of-a-public-0-96188-lb-0-96188   S6E10 | Anatomy of a Public 0.96188 | LB 0.96188    goodpjw2008  2026-10-27 01:39:40.150000          21  
goodpjw2008/wear-hasca-learned-links-counts-lb-0-92942     WEAR@HASCA | Learned Links + Counts | LB 0.92942    goodpjw2008  2026-10-04 12:47:48.817000          22  
goodpjw2008/biohub-private-0-925-hf-v3-flip3-tta-sub-voxe  Biohub Private 0.925 | HF V3 + flip3 TTA, Sub-voxe  goodpjw2008  2026-09-30 02:57:19.990000           8  
```

### Recent competition kernels (page-size 30)
Newest include CPU blends / EDA; **no new goodpjw G26 5-fold test upload** in the last ~3h window.
Notable: `amanatar/s6e11-rank1-q17-meta-bundle` dataset updated **2026-11-12 14:04 UTC**.

### s6e11 datasets by updated
```
ref                                                     title                                                  size  lastUpdated                 downloadCount  voteCount  usabilityRating  
------------------------------------------------------  -----------------------------------------------  ----------  --------------------------  -------------  ---------  ---------------  
amanatar/s6e11-rank1-q17-meta-bundle                    S6E11 Rank #1 Q17 Meta Bundle                      64592945  2026-11-12 14:04:45.277000              6          1  0.125            
daoviet/s6e11-gp-dvae-orthogonal-apex                   S6E11 GP and DVAE Orthogonal Apex                  11617844  2026-11-12 12:40:02.780000             47          0  0.29411766       
goodpjw2008/s6e11-target-first-gpt2-seed-2026           S6E11 Target-First GPT-2 Seed 2026                  7465705  2026-11-12 09:14:44.553000             47          1  0.375            
alexchilton/s6e11-oof-predictions                       S6E11 OOF + test predictions (15 models)          229776751  2026-11-12 08:22:15.010000           1198          0  0.5              
goodpjw2008/s6e11-compact-lm-seed7                      S6E11 Compact Target-First LM Seed 7               29140672  2026-11-12 08:04:21.997000            225          0  0.375            
amanatar/s6e11-sovereign-oof-test-and-orthogonal-vault  S6E11 Sovereign OOF+Test & Orthogonal Vault        65342999  2026-11-11 02:54:55.807000            273          0  0.64705884       
aphoristicartist/s6e11-detective-weights                s6e11-detective-weights                           462714314  2026-11-10 23:11:38.847000             46          0  0.375            
keshigami/model-vs-one-line-rule-six-kaggle-starters    Model vs one-line rule: 6 Kaggle starters              1675  2026-11-10 15:56:19.047000             12          0  1                
aphoristicartist/s6e11-detective-data                   s6e11-detective-data                                 665173  2026-11-10 15:55:57.537000             30          0  0.23529412       
dariushafshar/s6e11-golem-oof-library                   S6E11 OOF library - 9 models, frozen folds         76962207  2026-11-10 15:38:57.440000            151          0  0.9411765        
```


**3-hour newness check:** G26 dataset stamp **09:14 UTC**; audit after ~15:50 UTC → **no new G26 test-fold files**. Compact-LM seed7 dataset (`lm7_*_5fold`) remains the public Compact-LM 5-fold source.

---

## Step 3 — Side-by-side GPU feasibility (`<$3` discipline)

| Architecture | GPU / rate | Wall & $ (optimized estimate) | Cheaper K path | Expected OOF |
|---|---|---|---|---|
| A GPT-2 natural `T≈224` seed2026 F1–4 | RTX 4090 @ $0.34–0.39/hr (community) | train≈12h + score K10≈14h = **26h (~$9.4**) | K4 score path ≈**18h (~$6.5**) | OOF target **≥0.9500** (MS=0.95003; G26=0.95014) |
| A GPT-2 natural `T≈224` seed2026 F1–4 | A100 80GB @ ~$1.2–1.8/hr | train≈9h + score K10≈10h = **19h (~$26.4**) | K4 score path ≈**14h (~$18.8**) | OOF target **≥0.9500** (MS=0.95003; G26=0.95014) |
| A GPT-2 natural `T≈224` seed2026 F1–4 | L40S @ ~$0.79–1.09/hr | train≈10h + score K10≈12h = **22h (~$19.6**) | K4 score path ≈**15h (~$13.3**) | OOF target **≥0.9500** (MS=0.95003; G26=0.95014) |
| B Compact-LM `V=1372,T=72` seed2026 | A40 @ $0.59/hr | Fold0 canary (60rep+8FT+score) ≈**0.70h (~$0.41**) | Full 5F after canary ≈**3.5h (~$2.06**) | OOF target **0.9495–0.9499**; F0 gate ≥**0.94761** |
| B Compact-LM `V=1372,T=72` seed2026 | RTX 4090 @ $0.36/hr | Fold0 canary (60rep+8FT+score) ≈**0.45h (~$0.16**) | Full 5F after canary ≈**2.2h (~$0.79**) | OOF target **0.9495–0.9499**; F0 gate ≥**0.94761** |
| B Compact-LM `V=1372,T=72` seed2026 | L40S @ $0.89/hr | Fold0 canary (60rep+8FT+score) ≈**0.55h (~$0.49**) | Full 5F after canary ≈**2.8h (~$2.49**) | OOF target **0.9495–0.9499**; F0 gate ≥**0.94761** |

### Interpretation
- **Arch A** after pre-tokenize + bf16 SDPA + `batch_sequences=256` still lands roughly **~$5–$25+** for a full seed2026 F1–4+score depending on GPU — **outside <$3** unless only finishing **missing G26 test folds** on an already-trained seed (if weights existed; they are **not** in public kernel outputs).
- **Arch B** Fold-0 canary (`SEED=2026`, 60 rep + 8 FT, fix `save_npy_verified(oof[val_idx])`) is the **budget-fit** path to a second Compact-LM seed analogous to `G26−MS`, with hard gate **F0 AUC ≥ 0.94761**. Full 5F after pass stays near **~$1.3–$2.1** on A40/4090.
- Prior A40 crash on `smugcqcy2oqzeg` was the **1-line shape assert** on full `oof`, not a model failure — replica+fold1 already proved ~2.1h.

### Recommended next GPU dollar (still NO_SUBMIT)
1. **Do not** burn a submission on LM_trim 4/5.
2. Prefer **wait/pull G26 test 5-fold** if goodpjw uploads (highest stack leverage per $0).
3. Else **Compact-LM seed2026 Fold-0 canary ≤$0.45** with fixed saver → only then Folds 1–4.

---

## NO_SUBMIT ledger
- Submissions remaining today: **4**
- Staged but **blocked**: `sub_v11p1_slot3_pure_LMtrim_plus_G26denoise_0p10e4_NOSUBMIT.csv`
- ATH hold: **cand2 = 0.95355**

## Artifacts touched this audit
- `external/matapanino_gpt2_repo/` (clone)
- `external/masaya_kernel_out/` (kernel output pull)
- `external/goodpjw_lm_ladder_out/` (kernel output pull)
- This report

---
*Generated by `scripts/gm_audit_matapanino_09501_and_upstream.py` · `NO_SUBMIT=True`*
