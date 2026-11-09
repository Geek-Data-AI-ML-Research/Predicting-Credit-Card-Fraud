#!/usr/bin/env python3
"""Repair two broken Stage B/C members from post_gpt2_3ep_campaign.py and re-screen them on top of v8.

- lgb_gpt2_3ep_leaves4_bag6: OOF was predicted without the Platt base margin (test had it).
  Rebuild OOF as expit(logit(oof) + bm_b) with bm_b recomputed per fold.
- gpt2_rev_2ep: OOF is a standardised fold-wise LLR, test was a mean of per-fold ranks.
  Rebuild test as mean of per-fold LLR with the OOF affine map.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paradigm_stack_delta_eval import fast_auc  # noqa: E402

PRED = ROOT / "preds"
LOG = ROOT / "logs/overnight"


def rk(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    return (rankdata(x, method="average") - 0.5) / len(x)


def zrk(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    return logit(np.clip(rankdata(x, method="average") / len(x), 1e-6, 1.0 - 1e-6))


def calibrate_logit_to_prevalence(z_raw_a, y_a, *z_others):
    lr = LogisticRegression(C=100.0, solver="lbfgs", max_iter=200)
    lr.fit(np.asarray(z_raw_a, dtype=np.float64).reshape(-1, 1), np.asarray(y_a, dtype=np.int32).ravel())
    return tuple(lr.decision_function(np.asarray(z, dtype=np.float64).reshape(-1, 1)) for z in (z_raw_a, *z_others))


def bayes(ll, prior):
    t = np.logaddexp.reduce(ll[:, :, 1].astype(np.float64), axis=1)
    f = np.logaddexp.reduce(ll[:, :, 0].astype(np.float64), axis=1)
    return t - f + np.log(prior / (1 - prior))


def qmap_to_oof(x_test, x_oof):
    x_test = np.asarray(x_test, float).ravel()
    o = np.sort(np.asarray(x_oof, float).ravel())
    m, n = len(x_test), len(o)
    idx = np.minimum((((np.arange(m, dtype=np.float64) + 0.5) * n) / m).astype(np.int64), n - 1)
    out = np.empty(m, dtype=np.float64)
    out[np.argsort(x_test, kind="mergesort")] = o[idx]
    return out


def screen_on_v8(y, folds, z8, c_oof, grid):
    """Nested 1-D weight on the orthogonal, unit-scaled residual of zrk(c) vs zrk(v8)."""
    zc = zrk(c_oof)
    beta = float(np.dot(zc, z8) / np.dot(z8, z8))
    r = zc - beta * z8
    r = r / r.std()
    base = fast_auc(y, z8)
    oof = np.empty(len(y))
    ws, wins = [], 0
    for a, b in folds:
        best = max(grid, key=lambda w: fast_auc(y[a], z8[a] + w * r[a]))
        ws.append(best)
        oof[b] = z8[b] + best * r[b]
        wins += fast_auc(y[b], oof[b]) > fast_auc(y[b], z8[b])
    return dict(beta=beta, rho=float(np.corrcoef(zc, z8)[0, 1]), nested=float(fast_auc(y, oof)),
                delta=float(fast_auc(y, oof) - base), weights=ws, fold_wins=int(wins))


def main():
    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    out = {}

    # ---- lgb ----
    v5_o = np.load(PRED / "P_super_v5_harvest_oof.npy").ravel().astype(np.float64)
    v5_t = np.load(PRED / "P_super_v5_harvest_test.npy").ravel().astype(np.float64)
    z5, z5t = zrk(v5_o), zrk(v5_t)
    lo = np.load(PRED / "gpt2_3ep/lgb_gpt2_3ep_leaves4_bag6_oof.npy").ravel()
    lt = np.load(PRED / "gpt2_3ep/lgb_gpt2_3ep_leaves4_bag6_test.npy").ravel()
    lo_fix = np.empty_like(lo)
    for a, b in folds:
        _, bm_all, _ = calibrate_logit_to_prevalence(z5[a], y[a], z5, z5t)
        lo_fix[b] = expit(logit(np.clip(lo[b], 1e-12, 1 - 1e-12)) + bm_all[b])
    out["lgb"] = dict(
        old_auc=float(fast_auc(y, lo)), fixed_auc=float(fast_auc(y, lo_fix)),
        fold_auc=[float(fast_auc(y[b], lo_fix[b])) for _, b in folds],
        ks_raw=float(ks_2samp(lo_fix, lt).statistic), mean_oof=float(lo_fix.mean()), mean_test=float(lt.mean()),
    )
    np.save(PRED / "gpt2_3ep/lgb_gpt2_3ep_fixed_oof.npy", lo_fix)
    np.save(PRED / "gpt2_3ep/lgb_gpt2_3ep_fixed_test.npy", lt)
    print("lgb", json.dumps(out["lgb"], indent=1), flush=True)

    # ---- gpt2_rev ----
    gpt_o = np.load(PRED / "gpt2_3ep/gpt2_3ep_oof.npy").ravel()
    gpt_t = np.load(PRED / "gpt2_3ep/gpt2_3ep_test.npy").ravel()
    raw_o = np.full(len(y), np.nan)
    zt = []
    for f, (a, b) in enumerate(folds):
        prior = float(y[a].mean())
        raw_o[b] = bayes(np.load(PRED / f"gpt2_rev/oof_{f}.npy"), prior)
        zt.append(bayes(np.load(PRED / f"gpt2_rev/test_{f}.npy"), prior))
    zt = np.mean(zt, axis=0)
    mu, sd = raw_o.mean(), raw_o.std()
    ref_mu, ref_sd = gpt_o.mean(), gpt_o.std()
    ro = (raw_o - mu) / sd * ref_sd + ref_mu
    rt = (zt - mu) / sd * ref_sd + ref_mu
    old_o = np.load(PRED / "gpt2_3ep/gpt2_rev_2ep_oof.npy").ravel()
    old_t = np.load(PRED / "gpt2_3ep/gpt2_rev_2ep_test.npy").ravel()
    out["rev"] = dict(
        oof_matches_old=float(np.corrcoef(ro, old_o)[0, 1]),
        test_rank_corr_old=float(np.corrcoef(rk(rt), rk(old_t))[0, 1]),
        auc=float(fast_auc(y, ro)),
        ks_raw_old=float(ks_2samp(old_o, old_t).statistic), ks_raw_fixed=float(ks_2samp(ro, rt).statistic),
        ks_gpt2_3ep=float(ks_2samp(gpt_o, gpt_t).statistic),
    )
    np.save(PRED / "gpt2_3ep/gpt2_rev_2ep_fixed_oof.npy", ro)
    np.save(PRED / "gpt2_3ep/gpt2_rev_2ep_fixed_test.npy", rt)
    bi_o = rk(0.5 * rk(gpt_o) + 0.5 * rk(ro))
    bi_t = rk(0.5 * rk(gpt_t) + 0.5 * rk(rt))
    old_bi_t = np.load(PRED / "gpt2_3ep/gpt2_bidir_test.npy").ravel()
    out["rev"]["bidir_test_rank_corr_old"] = float(np.corrcoef(bi_t, rk(old_bi_t))[0, 1])
    np.save(PRED / "gpt2_3ep/gpt2_bidir_fixed_oof.npy", bi_o)
    np.save(PRED / "gpt2_3ep/gpt2_bidir_fixed_test.npy", bi_t)
    print("rev", json.dumps(out["rev"], indent=1), flush=True)

    # ---- screen on v8 (seed-42 folds) ----
    v8_o = np.load(PRED / "P_super_v8_part12_rect_oof.npy").ravel()
    z8 = zrk(v8_o)
    grid = np.round(np.arange(-0.06, 0.0601, 0.005), 4)
    out["v8_base"] = float(fast_auc(y, z8))
    out["screen"] = {}
    for name, c in [("lgb_fixed", lo_fix), ("xgb_gpt2_3ep", np.load(PRED / "gpt2_3ep/xgb_gpt2_3ep_d2_d3_bag6_oof.npy").ravel()),
                    ("rev_fixed", ro), ("gpt2_3ep", gpt_o)]:
        s = screen_on_v8(y, folds, z8, c, grid)
        out["screen"][name] = s
        print(f"screen {name}: rho={s['rho']:.5f} nested={s['nested']:.7f} Δ={s['delta']:+.2e} "
              f"wins={s['fold_wins']}/5 w={s['weights']}", flush=True)
    print(f"v8 base OOF {out['v8_base']:.7f}")
    (LOG / "fix_gpt2_members.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
