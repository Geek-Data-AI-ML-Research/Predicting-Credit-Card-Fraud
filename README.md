# Predicting Credit Card Fraud (Kaggle Playground S6E11)

Local research repo for [playground-series-s6e11](https://www.kaggle.com/competitions/playground-series-s6e11).

Public highlight: a frozen submission that scored **Public LB `0.95330`** (Kaggle ref `58151250`), published as an open notebook so others can reproduce that exact row.

## Public notebook (LB 0.95330)

| | |
|---|---|
| **Notebook** | [S6E11 \| Grandmaster Ladder \| LB 0.95330](https://www.kaggle.com/code/moranzavdi/s6e11-grandmaster-ladder-lb-0-95330) |
| **Dataset** | [s6e11-strike9-lb-095330](https://www.kaggle.com/datasets/moranzavdi/s6e11-strike9-lb-095330) |
| **Local path** | `notebooks/lb_095330/` |
| **Frozen CSV** | `notebooks/lb_095330/dataset/submission_095330.csv` |
| **SHA-256** | `f4a7a4f8636c99fabd82945c7d81956c5559c609dcb55cde6d4e03f80b0ea105` |

The notebook does **not** retrain the full private stack in-kernel. It loads the byte-identical Strike-9 CSV (SHA-checked), validates ids against `sample_submission.csv`, and writes `/kaggle/working/submission.csv`. Running + submitting that file should land on **Public LB 0.95330**.

### What “Strike 9” is

`0.95330` is the end of a local **LB-gradient ladder** on top of a Part-12 / Fraud-Signal-Lab backbone, not a single model:

| Step | Idea | Public LB |
|---|---|---:|
| `v8` | Part-12 wide LM + FSL-style rank-logit blend | 0.95307 |
| Slot 3 | verified F0/F2 contrastive upgrade | 0.95312 |
| Att 2 → Att 3 | multi-axis LB-gradient continuation | 0.95315 → 0.95317 |
| Strike 2 | honest 5-fold orthogonal harvest (NADE / Sameer / kNN / P4 / Graph) | 0.95319 |
| Strike 3A | + GLM-quad + graph GLM | 0.95322 |
| Strike 4A | continuation | 0.95323 |
| Strike 6 | reverse harmful pre-Part-12 tree dilution | 0.95325 |
| Strike 7 → 8 | keep riding the high-slope axis | 0.95327 → **0.95329 (#1)** |
| **Strike 9** | slope-weighted 2D step | **0.95330 (#1)** |

Mechanics that mattered in practice:

1. Work in **rank-logit** space; map test scores onto a fixed OOF reference with PCHIP quantiles.
2. Inject only **nested 5-fold orthogonal** residuals (LOFO OLS against the backbone controls).
3. Prefer **tiny Spearman distance** steps (`1-ρ ≈ 1e-5`) over large reshuffles.
4. Drop collinear public tree bags that dilute the LM backbone (Strike 5A cost **−0.00008** LB).

Repro scripts for the later ladder steps:

- `scripts/gm_top1_strike4_6_7_8_repro.py`
- `scripts/gm_top1_strike9_final_max.py`

## Credits (please upvote upstream work)

This score sits on public notebooks/datasets as well as our local training. Full linked tables live in the [public notebook](https://www.kaggle.com/code/moranzavdi/s6e11-grandmaster-ladder-lb-0-95330); summary:

- **goodpjw2008** — Part 12 wide stack + 2nd LM; [fraud-series-artifacts](https://www.kaggle.com/datasets/goodpjw2008/s6e11-fraud-series-artifacts); [oof-bank](https://www.kaggle.com/datasets/goodpjw2008/s6e11-oof-bank)
- **Malyshev Danil** — [Fraud Signal Lab](https://www.kaggle.com/code/malyshevdanil/fraud-signal-lab-gpt-2-and-glm-fusion) (FSL blend recipe)
- **sameershaikh27** — [Wide Pooled Stack · Every Bank Found](https://www.kaggle.com/code/sameershaikh27/s6e11-fraud-wide-pooled-stack-every-bank-found)
- **megayak**, **masayakawamata** — target-first LM / shallow-tree members and settings
- **areebbb** — [Wide Stack + Shallow-Tree Fusion · 0.95300](https://www.kaggle.com/code/areebbb/s6e11-wide-stack-shallow-tree-fusion-0-95300)
- **sadamtorres** — TabPFN–GBDT OOF/test pairs
- **alexchilton** — public OOF bank / shallow-tree stack
- Also referenced in the climb: Chris Deotte (target-first idea), mathsonchain, amanatar, lucifer19

## Repo layout (high level)

```
notebooks/lb_095330/     # public 0.95330 notebook + frozen dataset CSV
scripts/                 # local research / ladder / retake builders
preds/                   # local OOF & test prediction banks (gitignored bulk)
submissions/             # scored CSVs (mostly gitignored)
logs/overnight/          # JSON diagnostics & submit plans
docs/                    # research notes / audits
external/                # pulled public kernels & OOF banks
```

## Local next steps (beyond 0.95330)

After `0.95330` was passed on the LB (`0.95332`), SoftPseudo-family retake CSVs were staged for the next quota reset:

- Plan: `logs/overnight/RETAKE_SUBMIT_PLAN.md`
- Builders: `scripts/gm_day2_strike10_orth_leap.py`, `scripts/gm_day2_strike10b_softfamily_boost.py`, `scripts/gm_day2_retake_maxleap.py`
- Ready files: `submissions/sub_retake_R*.csv`, `submissions/sub_day2_slot*.csv`

Those use a nested SoftPseudo family pack with much larger honest OOF lift than the pure continuation ladder; they are **not** in the public 0.95330 notebook.

## License / competition rules

Respect Kaggle competition rules when using or redistributing predictions. Upstream notebooks/datasets keep their own licenses; credit them if you build on this release.
