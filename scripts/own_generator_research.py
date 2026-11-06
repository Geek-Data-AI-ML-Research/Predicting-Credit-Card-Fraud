#!/usr/bin/env python3
"""Own generator research: cross-fitted interaction-TE GLM + shallow LGBM.

Hypotheses (from own forensics, not notebooks):
- DGP ~ additive log-odds in per-value risk + a few pairwise interactions
  (merchant x amount, velocity x hours, amount x merchant_risk, channel x billing...)
- Digit keys are smoothed coarsenings of value; multi-smoothing TE logits let LR
  choose shrinkage per column.
Gate: solo OOF, then blend vs P_super_v4p OOF (pooled + per-fold + Fold0).
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-7
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

tr = pd.read_csv(ROOT/"data/train.csv"); te = pd.read_csv(ROOT/"data/test.csv")
y = tr["is_fraud"].values.astype(float)
N, NT = len(tr), len(te)
both = pd.concat([tr.drop(columns=["is_fraud"]), te], ignore_index=True)

NUMS = ["amount_usd","hours_since_last_txn","distance_from_home_km","account_balance_usd","velocity_score","merchant_risk_score"]
CATS = ["merchant_category","card_type","auth_method","channel","device_type"]
INTS = ["time_of_day_hour","day_of_week","cvv_retry_count","txn_count_last_24h","prior_disputes","card_age_months","customer_age"]
FLAGS = ["is_foreign_transaction","is_new_merchant","used_vpn","ip_country_mismatch","billing_shipping_mismatch","is_ai_generated_scam_attempt"]

def factorize(s):
    codes, _ = pd.factorize(s, sort=False)
    return codes.astype(np.int64)

codes = {}          # name -> (codes_all, n_codes)
def add_code(name, arr):
    arr = np.asarray(arr)
    if arr.dtype.kind in "biuf" and arr.min() >= 0 and np.all(arr == arr.astype(np.int64)):
        a = arr.astype(np.int64); codes[name] = (a, int(a.max())+1)
    else:
        codes[name] = (factorize(pd.Series(arr)), int(pd.Series(arr).nunique()))

for c in CATS: add_code("cat_"+c, both[c].values)
for c in INTS: add_code("int_"+c, both[c].values)
for f in FLAGS: add_code("flg_"+f, (both[f]==True).astype(int).values)
flagcount = sum((both[f]==True).astype(int).values for f in FLAGS)
add_code("flagcount", flagcount)

# quantile bins from train only
qbins = {}
for c in NUMS:
    edges = np.unique(np.quantile(tr[c].values, np.linspace(0,1,11)))
    edges[0], edges[-1] = -np.inf, np.inf
    qbins[c] = np.clip(np.digitize(both[c].values, edges[1:-1]), 0, 9)
    add_code("q_"+c, qbins[c])
    add_code("exact_"+c, both[c].values)

# digit keys
for c in NUMS:
    cents = np.round(both[c].values*100).astype(np.int64)
    intpart = cents//100
    lead = pd.Series(intpart).astype(str).str[0].astype(int).values
    add_code("dig_cents_"+c, cents % 100)
    add_code("dig_last_"+c, cents % 10)
    add_code("dig_units_"+c, intpart % 10)
    add_code("dig_intpart_"+c, np.clip(intpart, 0, 999))
    add_code("dig_lead_"+c, lead)

# pair keys (own screen winners + structural)
def pair(name, a, b):
    ca, na = codes[a]; cb, nb = codes[b]
    codes[name] = (ca*nb + cb, int(na*nb))
pair("pair_merch_amountq", "cat_merchant_category", "q_amount_usd")
pair("pair_amountq_riskq", "q_amount_usd", "q_merchant_risk_score")
pair("pair_velq_hoursq", "q_velocity_score", "q_hours_since_last_txn")
pair("pair_merch_riskq", "cat_merchant_category", "q_merchant_risk_score")
pair("pair_channel_billing", "cat_channel", "flg_billing_shipping_mismatch")
pair("pair_cardage_custage", "q_card_age_months" if "q_card_age_months" in codes else "int_card_age_months", "int_customer_age")
pair("pair_txn_velq", "int_txn_count_last_24h", "q_velocity_score")
pair("pair_device_hour", "cat_device_type", "int_time_of_day_hour")
pair("pair_auth_flagcount", "cat_auth_method", "flagcount")
pair("pair_hoursq_amountx", "q_hours_since_last_txn", "exact_amount_usd")
pair("pair_velq_hoursx", "q_velocity_score", "exact_hours_since_last_txn")
pair("pair_foreign_ip", "flg_is_foreign_transaction", "flg_ip_country_mismatch")

MAIN = [k for k in codes if not k.startswith("pair_")]
PAIRS = [k for k in codes if k.startswith("pair_")]
log("keys:", len(MAIN), "main,", len(PAIRS), "pairs,", {k: codes[k][1] for k in PAIRS})

def te_matrix(idxs_ref_list):
    """TE logit features: for each (ref_idx, target_idx) produce smoothed TE from ref.
    Returns dict name->array over concatenated target idx order of last call... simpler: function below per split."""
    pass

def te_feats(ref_idx, tgt_idx, keys, ms):
    """ms: list of smoothing values; returns (len(tgt), len(keys)*len(ms)) logit TE."""
    p = y[ref_idx].mean()
    out = np.empty((len(tgt_idx), len(keys)*len(ms)), dtype=np.float32)
    yr = y[ref_idx]
    for j, k in enumerate(keys):
        c_all, nc = codes[k]
        cr = c_all[ref_idx]
        n = np.bincount(cr, minlength=nc).astype(np.float64)
        s = np.bincount(cr, weights=yr, minlength=nc)
        ct = c_all[tgt_idx]
        for mi, m in enumerate(ms):
            r = (s[ct] + m*p) / (n[ct] + m)
            out[:, j*len(ms)+mi] = np.log(np.clip(r, EPS, 1-EPS)/(1-np.clip(r, EPS, 1-EPS)))
    return out

def raw_feats(idx):
    df = both.iloc[idx]
    cols = []
    for c in NUMS + ["card_age_months","customer_age","txn_count_last_24h","cvv_retry_count","prior_disputes","time_of_day_hour"]:
        cols.append(np.log1p(np.clip(df[c].values.astype(float), 0, None)))
    for f in FLAGS: cols.append((df[f]==True).astype(float).values)
    cols.append(flagcount[idx].astype(float))
    # flag pairwise products
    fl = np.column_stack([(df[f]==True).astype(float).values for f in FLAGS])
    for i in range(len(FLAGS)):
        for j in range(i+1, len(FLAGS)):
            cols.append(fl[:, i]*fl[:, j])
    return np.column_stack(cols).astype(np.float32)

RAW_TEST = raw_feats(np.arange(N, N+NT))

skf = StratifiedKFold(5, shuffle=True, random_state=42)
inner = StratifiedKFold(5, shuffle=True, random_state=17)

oof_lr = np.zeros(N); test_lr = np.zeros(NT)
oof_lgb = np.zeros(N); test_lgb = np.zeros(NT)
fold_aucs = {"lr": [], "lgb": []}

for fold, (itr, iva) in enumerate(skf.split(tr, y)):
    # inner OOF TE for training rows
    Xtr_parts, ytr_parts = [], []
    for iitr, iiva in inner.split(itr, y[itr]):
        ref, tgt = itr[iitr], itr[iiva]
        Xtr_parts.append(np.hstack([te_feats(ref, tgt, MAIN, [20., 200.]), te_feats(ref, tgt, PAIRS, [50.])]))
        ytr_parts.append(y[tgt])
    order = np.argsort(np.concatenate([itr[iiva] for _, iiva in inner.split(itr, y[itr])]))
    Xtr = np.vstack(Xtr_parts)[order]
    ytr = np.concatenate(ytr_parts)[order]
    Xva = np.hstack([te_feats(itr, iva, MAIN, [20., 200.]), te_feats(itr, iva, PAIRS, [50.])])
    Xte = np.hstack([te_feats(itr, np.arange(N, N+NT), MAIN, [20., 200.]), te_feats(itr, np.arange(N, N+NT), PAIRS, [50.])])
    Xtr_full = np.hstack([Xtr, raw_feats(itr)])
    Xva_full = np.hstack([Xva, raw_feats(iva)])
    Xte_full = np.hstack([Xte, RAW_TEST])
    log(f"fold{fold} X={Xtr_full.shape}")

    sc = StandardScaler().fit(Xtr_full)
    lr = LogisticRegression(C=0.1, max_iter=2000)
    lr.fit(sc.transform(Xtr_full), ytr)
    p_va = lr.predict_proba(sc.transform(Xva_full))[:, 1]
    oof_lr[iva] = p_va; test_lr += lr.predict_proba(sc.transform(Xte_full))[:, 1]/5
    a = roc_auc_score(y[iva], p_va); fold_aucs["lr"].append(a)

    m = lgb.LGBMClassifier(n_estimators=700, learning_rate=0.05, num_leaves=4, max_depth=2,
                           subsample=0.9, colsample_bytree=0.8, reg_lambda=5.0, random_state=42, n_jobs=8, verbose=-1)
    m.fit(Xtr_full, ytr)
    p2 = m.predict_proba(Xva_full)[:, 1]
    oof_lgb[iva] = p2; test_lgb += m.predict_proba(Xte_full)[:, 1]/5
    a2 = roc_auc_score(y[iva], p2); fold_aucs["lgb"].append(a2)
    log(f"fold{fold}: LR={a:.6f} LGB4={a2:.6f}")

np.save(ROOT/"preds/own_teglm_oof.npy", oof_lr); np.save(ROOT/"preds/own_teglm_test.npy", test_lr)
np.save(ROOT/"preds/own_telgb_oof.npy", oof_lgb); np.save(ROOT/"preds/own_telgb_test.npy", test_lgb)

def r01(a): return (rankdata(a.astype(float))-1)/(len(a)-1)
base = np.load(ROOT/"preds/P_super_v4p_oof.npy").astype(float)
report = {"solo": {}, "folds": fold_aucs}
folds_idx = [iva for _, iva in skf.split(tr, y)]
for name, oof in [("own_teglm", oof_lr), ("own_telgb", oof_lgb)]:
    solo = roc_auc_score(y, oof)
    rho = float(np.corrcoef(r01(base), r01(oof))[0,1])
    best = (roc_auc_score(y, base), 0.0)
    per_alpha = {}
    for alpha in [0.02,0.05,0.08,0.12,0.18,0.25,0.35,0.5]:
        auc = roc_auc_score(y, (1-alpha)*r01(base)+alpha*r01(oof))
        per_alpha[str(alpha)] = auc
        if auc > best[0]: best = (auc, alpha)
    # per-fold delta at best alpha
    deltas = []
    for iva in folds_idx:
        b = roc_auc_score(y[iva], base[iva]); c = roc_auc_score(y[iva], (1-best[1])*r01(base[iva])+best[1]*r01(oof[iva]))
        deltas.append(float(c-b))
    report["solo"][name] = {"auc": float(solo), "rho_vs_v4p": rho, "best_blend_auc": float(best[0]),
                            "best_alpha": best[1], "blend_gain": float(best[0]-roc_auc_score(y, base)),
                            "per_fold_delta_at_best_alpha": deltas, "fold0_delta": deltas[0], "per_alpha": per_alpha}
    log(name, report["solo"][name])
report["base_v4p_oof"] = float(roc_auc_score(y, base))
(ROOT/"logs/overnight/own_generator_research.json").write_text(json.dumps(report, indent=2))
log("DONE", report)
