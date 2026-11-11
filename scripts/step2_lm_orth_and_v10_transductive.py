#!/usr/bin/env python3
"""Step 2: LM double-orth of soft_diff/gb_soft + v10_transductive_bulsah.

STRICT NO_SUBMIT. Local CPU only. Memory-capped (OMP=2).
Artifacts -> preds/v10_transductive_bulsah/ + logs/overnight/STEP2_*.md/json
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["CATBOOST_THREAD_COUNT"] = "2"
os.environ["LIGHTGBM_EXEC"] = os.environ.get("LIGHTGBM_EXEC", "")
os.environ["NO_SUBMIT"] = "1"

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import ks_2samp, spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from gm_top3_strike3_expanded7f import (
    HARV,
    P12,
    PRED,
    SUB,
    calibrate_logit_to_prevalence,
    fast_auc,
    pchip_quantile_map,
    zrk,
)
from recon_and_forensic_audit import (
    N_KNOTS,
    build_z_ref,
    nested_delta,
    resid_diagnostics,
)

OUT = ROOT / "logs" / "overnight"
ART = PRED / "v10_transductive_bulsah"
OUT.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)
T0 = time.time()
STATUS = OUT / "STEP2_STATUS.json"
TARGET_RHO = [0.2e-4, 0.5e-4, 1.0e-4]
FLAGS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]
CONT8 = [
    "amount_usd",
    "account_balance_usd",
    "velocity_score",
    "merchant_risk_score",
    "hours_since_last_txn",
    "distance_from_home_km",
    "txn_count_last_24h",
    "card_age_months",
]


def log(msg: str):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)
    try:
        STATUS.write_text(
            json.dumps(
                {
                    "t_s": round(time.time() - T0, 1),
                    "msg": msg,
                    "rss_gb": rss_gb(),
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                indent=2,
            )
        )
    except Exception:
        pass


def rss_gb() -> float:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # mac: bytes
    except Exception:
        return -1.0


def free_gb() -> float:
    try:
        # mac vm_stat page size 16384
        import subprocess

        out = subprocess.check_output(["vm_stat"], text=True)
        ps = 16384
        d = {}
        for line in out.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                d[k.strip()] = int(v.strip().rstrip("."))
        free = d.get("Pages free", 0) + d.get("Pages inactive", 0) + d.get("Pages speculative", 0)
        return free * ps / 1e9
    except Exception:
        return -1.0


def fold_orth_multi(zo, zt, Xo, Xt, folds):
    """5-fold LOFO OLS residual vs multi-col controls; unit-scale per fold train."""
    ro = np.zeros(len(zo), dtype=np.float64)
    rtf = []
    for a, b in folds:
        ols = LinearRegression().fit(Xo[a], zo[a])
        ua = zo[a] - ols.predict(Xo[a])
        mu, sd = float(ua.mean()), float(ua.std() + 1e-12)
        ro[b] = (zo[b] - ols.predict(Xo[b]) - mu) / sd
        rtf.append((zt - ols.predict(Xt) - mu) / sd)
    rt_raw = np.mean(rtf, 0)
    rt = pchip_quantile_map(rt_raw, ro, n_knots=N_KNOTS)
    return ro, rt, rt_raw


def weight_for_spearman_dist(z_base_t, r_t, target_dist, w_hi=0.25):
    """Binary-search w>=0 s.t. 1-spearman(z_base+w*r, z_base) ~= target_dist."""

    def dist(w):
        if abs(w) < 1e-15:
            return 0.0
        rho = float(spearmanr(z_base_t + w * r_t, z_base_t).correlation)
        return 1.0 - rho

    lo, hi = 0.0, float(w_hi)
    if dist(hi) < target_dist:
        # extend
        for _ in range(8):
            hi *= 1.6
            if dist(hi) >= target_dist:
                break
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        if dist(mid) < target_dist:
            lo = mid
        else:
            hi = mid
    w = 0.5 * (lo + hi)
    return float(w), float(dist(w))


def nested_at_w(y, z_base, r, folds, w):
    base = float(fast_auc(y, z_base))
    sc = float(fast_auc(y, z_base + w * r))
    pos = sum(
        1
        for _, b in folds
        if float(fast_auc(y[b], (z_base + w * r)[b])) > float(fast_auc(y[b], z_base[b]))
    )
    return {
        "delta_1e5": round((sc - base) * 1e5, 2),
        "w": round(float(w), 6),
        "pos": int(pos),
        "base": base,
    }


def load_gpt2_med_5f(y, folds, nte):
    """Assemble full 5-fold OOF + mean test from gpt2_med fold dumps (Bayes LLR)."""
    oof = np.full(len(y), np.nan, dtype=np.float64)
    tests = []
    for fi, (a, b) in enumerate(folds):
        op = PRED / f"gpt2_med_4ep/oof_{fi}.npy"
        tp = PRED / f"gpt2_med_4ep/test_{fi}.npy"
        if not op.exists() or not tp.exists():
            return None, None
        oraw = np.load(op)
        traw = np.load(tp)
        if oraw.shape[0] != len(b):
            return None, None
        prior = float(y[a].mean())
        z_o = (
            np.logaddexp.reduce(oraw[:, :, 1].astype(np.float64), axis=1)
            - np.logaddexp.reduce(oraw[:, :, 0].astype(np.float64), axis=1)
            + np.log(prior / (1.0 - prior))
        )
        # test dumps are (nte, 4, 2) — mean across available order dims
        if traw.ndim == 3:
            z_t = np.mean(traw[:, :, 1].astype(np.float64) - traw[:, :, 0].astype(np.float64), axis=1)
        else:
            z_t = traw.astype(np.float64).ravel()
        oof[b] = z_o
        tests.append(z_t)
    if np.isnan(oof).any():
        return None, None
    return oof, np.mean(tests, 0)


def load_lm_bank(y, folds, ntr, nte, test_ids):
    """Load all available 5-fold LM vectors; PCHIP-align test into each OOF rank-logit space."""
    bank = {}

    def add(name, xo, xt):
        xo = np.asarray(xo, dtype=np.float64).ravel()
        xt = np.asarray(xt, dtype=np.float64).ravel()
        if len(xo) != ntr or len(xt) != nte or not np.isfinite(xo).all() or not np.isfinite(xt).all():
            log(f"  skip LM {name}: shape/finite fail {getattr(xo,'shape',None)} {getattr(xt,'shape',None)}")
            return
        zo = zrk(xo)
        zt = pchip_quantile_map(zrk(xt), zo, n_knots=N_KNOTS)
        bank[name] = (zo, zt)
        log(f"  LM {name}: oof_auc={fast_auc(y, zo):.7f}")

    # lm12 / part12 wide stack
    add("lm12_wide", np.load(P12 / "wide_lm_stack_nested.npy"), np.load(P12 / "wide_lm_stack_test.npy"))
    if (P12 / "lm_oof.npy").exists() and (P12 / "lm_test.npy").exists():
        add("lm12_raw", np.load(P12 / "lm_oof.npy"), np.load(P12 / "lm_test.npy"))

    # masaya gpt2
    masa = ROOT / "external/banks/masayakawamata_s6e11-gpt2-target-first-oof"
    if (masa / "oof.npy").exists():
        add("masaya_gpt2", np.load(masa / "oof.npy"), np.load(masa / "test.npy"))

    # gpt2_rev
    for tag, op, tp in [
        ("gpt2_rev", PRED / "gpt2_3ep/gpt2_rev_2ep_fixed_oof.npy", PRED / "gpt2_3ep/gpt2_rev_2ep_fixed_test.npy"),
        ("gpt2_3ep", PRED / "gpt2_3ep/gpt2_3ep_oof.npy", PRED / "gpt2_3ep/gpt2_3ep_test.npy"),
        ("lm_gpt2_3ep", PRED / "lm_gpt2_3ep/lm_oof_5f.npy", PRED / "lm_gpt2_3ep/lm_test_5f.npy"),
    ]:
        if op.exists() and tp.exists():
            add(tag, np.load(op), np.load(tp))

    # gpt2_medium full 5f if assemblable
    gmo, gmt = load_gpt2_med_5f(y, folds, nte)
    if gmo is not None:
        add("gpt2_medium", gmo, gmt)
    else:
        log("  gpt2_medium full-5f unavailable — will use B_xgb_llr as LM-adjacent control")

    # B_xgb_llr (paradigm B)
    bxo, bxt = PRED / "paradigm_B/xgb_llr_oof.npy", PRED / "paradigm_B/xgb_llr_test.npy"
    if bxo.exists() and bxt.exists():
        add("B_xgb_llr", np.load(bxo), np.load(bxt))

    return bank


def part1_lm_orth(y, folds, z_ref_o, z_st9_t, s9, lm_bank):
    log("PART1: LM double-orthogonalization of soft_diff + gb_soft_bag3")
    # drop idf_p3 + clean035 permanently (not loaded)
    comps = [
        ("soft_diff", PRED / "glm_softpseudo_clean_diff_oof.npy", PRED / "glm_softpseudo_clean_diff_test.npy"),
        ("gb_soft_bag3", PRED / "gb_softpseudo_clean_bag3_oof.npy", PRED / "gb_softpseudo_clean_bag3_test.npy"),
    ]
    # control matrix: [1 handled by LinearRegression intercept] + z_ref + all LMs
    names = list(lm_bank.keys())
    Xo = np.column_stack([z_ref_o] + [lm_bank[n][0] for n in names])
    Xt = np.column_stack([z_st9_t] + [lm_bank[n][1] for n in names])
    log(f"  controls: z_ref + {names}  (p={Xo.shape[1]})")

    rows = []
    arts = {}
    for name, op, tp in comps:
        xo = np.load(op).ravel()
        xt = np.load(tp).ravel()
        zo = zrk(xo)
        zt = pchip_quantile_map(zrk(xt), zo, n_knots=N_KNOTS)
        # baseline: orth vs z_ref only (for comparison)
        ro1, rt1, rtr1 = fold_orth_multi(zo, zt, z_ref_o.reshape(-1, 1), z_st9_t.reshape(-1, 1), folds)
        nest1 = nested_delta(y, z_ref_o, ro1, folds)
        diag1 = resid_diagnostics(ro1, rt1, rtr1, s9, expit(zt))
        diag1["spearman_dist_st9_test_1e4"] = round(
            (1.0 - float(spearmanr(z_st9_t + rt1, z_st9_t).correlation)) * 1e4, 3
        )

        # full LM orth
        ro, rt, rtr = fold_orth_multi(zo, zt, Xo, Xt, folds)
        # residual corr check vs controls on OOF
        corrs = {
            "z_ref": float(np.corrcoef(ro, z_ref_o)[0, 1]),
        }
        for i, n in enumerate(names):
            corrs[n] = float(np.corrcoef(ro, Xo[:, i + 1])[0, 1])
        nest_free = nested_delta(y, z_ref_o, ro, folds)
        diag = resid_diagnostics(ro, rt, rtr, s9, expit(z_st9_t + rt))
        step_rows = []
        for td in TARGET_RHO:
            w, got = weight_for_spearman_dist(z_st9_t, rt, td)
            nest_w = nested_at_w(y, z_ref_o, ro, folds, w)
            step_rows.append(
                {
                    "target_1m_rho": td,
                    "target_1e4": round(td * 1e4, 4),
                    "achieved_1e4": round(got * 1e4, 4),
                    **nest_w,
                    "dispersion_ratio_raw": diag["dispersion_ratio_raw"],
                    "ks_raw": diag["ks_raw"],
                }
            )
            log(
                f"  {name}_orth_LM @1-ρ={td*1e4:.2f}e-4: w={w:.5f} Δ={nest_w['delta_1e5']}e-5 "
                f"{nest_w['pos']}/5 disp={diag['dispersion_ratio_raw']} KS={diag['ks_raw']}"
            )
        row = {
            "name": f"{name}_orth_LM",
            "base_vs_zref_only": {**nest1, **{k: diag1[k] for k in ("dispersion_ratio_raw", "ks_raw", "spearman_dist_st9_test_1e4")}},
            "free_grid": nest_free,
            "corr_vs_controls": {k: round(v, 6) for k, v in corrs.items()},
            "dispersion_ratio_raw": diag["dispersion_ratio_raw"],
            "ks_raw": diag["ks_raw"],
            "steps": step_rows,
        }
        rows.append(row)
        arts[name] = (ro, rt, rtr)
        np.save(ART / f"{name}_orth_LM_oof.npy", ro.astype(np.float32))
        np.save(ART / f"{name}_orth_LM_test.npy", rt.astype(np.float32))
        np.save(ART / f"{name}_orth_LM_test_raw.npy", rtr.astype(np.float32))
        log(
            f"  {name}: freeΔ={nest_free['delta_1e5']}e-5@{nest_free['w']} {nest_free['pos']}/5 | "
            f"vs_zref_only Δ={nest1['delta_1e5']} | max|corr|={max(abs(v) for v in corrs.values()):.2e}"
        )
    del Xo, Xt
    gc.collect()
    return rows, arts


def bitmask64(df: pd.DataFrame) -> np.ndarray:
    flags = df[FLAGS].to_numpy(np.float64)
    return (flags * (2 ** np.arange(flags.shape[1]))).sum(1).astype(np.int32)


def exact_key(arr) -> np.ndarray:
    """Exact float/string key as pandas StringDtype-friendly object array."""
    if np.issubdtype(np.asarray(arr).dtype, np.floating):
        # exact float string without scientific noise for typical decimals
        return pd.Series(arr).map(lambda x: format(float(x), ".12g")).to_numpy()
    return pd.Series(arr).astype(str).to_numpy()


def _groupby_sum_count(values: np.ndarray, codes: np.ndarray):
    n = int(codes.max()) + 1 if len(codes) else 0
    sm = np.bincount(codes, weights=values, minlength=n).astype(np.float64)
    ct = np.bincount(codes, minlength=n).astype(np.float64)
    return sm, ct


def eb_loo_mean(values, codes, m: float):
    """Empirical-Bayes shrunk leave-one-out group mean (int codes)."""
    values = np.asarray(values, dtype=np.float64)
    codes = np.asarray(codes, dtype=np.int32)
    sm, ct = _groupby_sum_count(values, codes)
    global_mu = float(np.mean(values))
    loo_sum = sm[codes] - values
    loo_cnt = ct[codes] - 1.0
    out = (loo_sum + m * global_mu) / (loo_cnt + m)
    out = np.where(loo_cnt > 0, out, global_mu)
    return out


def eb_loo_disp(values, codes, m: float):
    values = np.asarray(values, dtype=np.float64)
    codes = np.asarray(codes, dtype=np.int32)
    sm, ct = _groupby_sum_count(values, codes)
    sm2, _ = _groupby_sum_count(values * values, codes)
    loo_cnt = ct[codes] - 1.0
    loo_sum = sm[codes] - values
    loo_sum2 = sm2[codes] - values * values
    mean = np.divide(loo_sum, loo_cnt, out=np.zeros_like(loo_sum), where=loo_cnt > 0)
    var = np.divide(loo_sum2, loo_cnt, out=np.zeros_like(loo_sum2), where=loo_cnt > 0) - mean * mean
    var = np.clip(var, 0, None)
    global_var = float(np.var(values))
    out = (loo_cnt * var + m * global_var) / (loo_cnt + m)
    out = np.where(loo_cnt > 1, out, global_var)
    return np.sqrt(out)


def m_estimate_te(keys_tr, y_tr, keys_va, keys_te, m: float, prior: float):
    """Smoothed TE: (sum_y + m*prior)/(cnt + m) fit on train only."""
    g = pd.Series(keys_tr)
    ys = pd.Series(y_tr.astype(np.float64))
    sm = ys.groupby(g).sum()
    ct = ys.groupby(g).count()
    te = (sm + m * prior) / (ct + m)

    def map_keys(k):
        return pd.Series(k).map(te).fillna(prior).to_numpy(np.float64)

    return map_keys(keys_tr), map_keys(keys_va), map_keys(keys_te)


def build_v10_features(tr, te, orig, z_anchor_tr, z_anchor_te, y):
    """Build compact 1.02M-support transductive + bulsah feature matrix (float32)."""
    log("PART2: building transductive + bulsah features on train+test+orig20k")
    ntr, nte = len(tr), len(te)
    # align orig columns
    common = [c for c in tr.columns if c in orig.columns and c != "is_fraud" and c != "id"]
    o = orig[common].copy()
    o["is_fraud"] = orig["is_fraud"].to_numpy() if "is_fraud" in orig.columns else -1
    # concat
    tr2 = tr.copy()
    te2 = te.copy()
    tr2["_split"] = 0
    te2["_split"] = 1
    o["_split"] = 2
    # ensure risk_flag_count
    for df in (tr2, te2, o):
        df["risk_flag_count"] = df[FLAGS].sum(axis=1).astype(np.float64)
        df["flag_bitmask_64"] = bitmask64(df)

    all_df = pd.concat([tr2, te2, o], axis=0, ignore_index=True)
    n_all = len(all_df)
    log(f"  concat n={n_all} (tr={ntr} te={nte} orig={len(o)}) free={free_gb():.2f}GB")

    z_anchor_all = np.full(n_all, np.nan, dtype=np.float64)
    z_anchor_all[:ntr] = z_anchor_tr
    z_anchor_all[ntr : ntr + nte] = z_anchor_te
    z_anchor_all[ntr + nte :] = float(np.mean(z_anchor_tr))  # orig prior

    feat_cols = []
    feat_names = []

    def push(arr, name):
        a = np.asarray(arr, dtype=np.float64).ravel()
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        feat_cols.append(a.astype(np.float32))
        feat_names.append(name)

    # ---- key support features (top 1D + 2D) ----
    cont_keys = [
        "amount_usd",
        "account_balance_usd",
        "velocity_score",
        "merchant_risk_score",
        "hours_since_last_txn",
        "distance_from_home_km",
    ]
    cat_keys = ["merchant_category", "card_type", "auth_method", "device_type", "flag_bitmask_64"]
    # 1D keys: 6 cont + 5 cat = 11
    # 2D: select high-value crosses (24) -> ~35 total
    key_specs = [(c,) for c in cont_keys + cat_keys]
    crosses = [
        ("amount_usd", "merchant_category"),
        ("amount_usd", "card_type"),
        ("amount_usd", "flag_bitmask_64"),
        ("velocity_score", "merchant_category"),
        ("velocity_score", "auth_method"),
        ("velocity_score", "flag_bitmask_64"),
        ("merchant_risk_score", "merchant_category"),
        ("merchant_risk_score", "device_type"),
        ("merchant_risk_score", "flag_bitmask_64"),
        ("hours_since_last_txn", "merchant_category"),
        ("hours_since_last_txn", "auth_method"),
        ("distance_from_home_km", "merchant_category"),
        ("distance_from_home_km", "flag_bitmask_64"),
        ("account_balance_usd", "card_type"),
        ("account_balance_usd", "flag_bitmask_64"),
        ("merchant_category", "card_type"),
        ("merchant_category", "auth_method"),
        ("merchant_category", "flag_bitmask_64"),
        ("card_type", "auth_method"),
        ("card_type", "device_type"),
        ("auth_method", "device_type"),
        ("auth_method", "flag_bitmask_64"),
        ("device_type", "flag_bitmask_64"),
        ("velocity_score", "merchant_risk_score"),
    ]
    key_specs.extend(crosses)
    key_specs = key_specs[:35]
    log(f"  key specs: {len(key_specs)}")

    risk_vars = ["velocity_score", "merchant_risk_score", "risk_flag_count", "amount_usd"]

    for spec in key_specs:
        if len(spec) == 1:
            kname = spec[0]
            keys = exact_key(all_df[spec[0]].to_numpy())
        else:
            kname = "x".join(spec)
            keys = (
                pd.Series(exact_key(all_df[spec[0]].to_numpy())).astype(str)
                + "|"
                + pd.Series(exact_key(all_df[spec[1]].to_numpy())).astype(str)
            ).to_numpy()

        codes, _ = pd.factorize(keys, sort=False)
        codes = codes.astype(np.int32)
        _, ct = _groupby_sum_count(np.ones(len(codes), dtype=np.float64), codes)
        cnt = ct[codes]
        push(np.log1p(cnt), f"logcnt_{kname}")
        push((cnt == 1).astype(np.float32), f"rare1_{kname}")
        push((cnt == 2).astype(np.float32), f"rare2_{kname}")

        for rv in risk_vars:
            vals = all_df[rv].to_numpy(np.float64)
            sm, ctv = _groupby_sum_count(vals, codes)
            loo_m = np.divide(sm[codes] - vals, ctv[codes] - 1.0, out=np.full(len(vals), float(vals.mean())), where=(ctv[codes] > 1))
            push(loo_m, f"loo_mean_{rv}_{kname}")

        # Out-of-row transductive anchor expectations (EB m=5,20)
        for m in (5.0, 20.0):
            push(eb_loo_mean(z_anchor_all, codes, m), f"zanch_eb{int(m)}_{kname}")
            push(eb_loo_disp(z_anchor_all, codes, m), f"zanch_disp{int(m)}_{kname}")

        del keys, codes
        if len(feat_names) % 40 == 0:
            log(f"  ... feats={len(feat_names)} rss={rss_gb():.2f}GB free={free_gb():.2f}GB")

    # ---- Bulsah mantissa / step-grid ----
    log("  bulsah mantissa + step-grid + ratios")
    for c in CONT8:
        v = all_df[c].to_numpy(np.float64)
        # fractional cents / mantissa
        cents = np.round(v * 100.0)
        frac = v - np.floor(v)
        push(frac, f"frac_{c}")
        push(cents % 10, f"cent1_{c}")
        push(cents % 100, f"cent2_{c}")
        # first/last digit bigrams via string of rounded value
        s = pd.Series(np.round(v, 6)).astype(str).str.replace(".", "", regex=False).str.replace("-", "", regex=False)
        first2 = s.str[:2].apply(lambda x: float(x) if x.isdigit() else -1.0).to_numpy(np.float64)
        last2 = s.str[-2:].apply(lambda x: float(x) if x.isdigit() else -1.0).to_numpy(np.float64)
        push(first2, f"first2_{c}")
        push(last2, f"last2_{c}")
        # step-grid parity
        step = np.rint(v / (0.01 if "amount" in c or "balance" in c else 0.1))
        push(step % 2, f"step2_{c}")
        push(step % 5, f"step5_{c}")
        push(step % 10, f"step10_{c}")

    # 8 pairwise ratios
    amt = all_df["amount_usd"].to_numpy(np.float64)
    bal = all_df["account_balance_usd"].to_numpy(np.float64)
    vel = all_df["velocity_score"].to_numpy(np.float64)
    risk = all_df["merchant_risk_score"].to_numpy(np.float64)
    hrs = all_df["hours_since_last_txn"].to_numpy(np.float64)
    dist = all_df["distance_from_home_km"].to_numpy(np.float64)
    txn = all_df["txn_count_last_24h"].to_numpy(np.float64)
    age = all_df["card_age_months"].to_numpy(np.float64)
    ratios = {
        "amt_bal": amt / (bal + 1),
        "vel_risk": vel / (risk + 1e-6),
        "amt_vel": amt / (vel + 1),
        "bal_risk": bal / (risk + 1e-6),
        "dist_hrs": dist / (hrs + 1),
        "amt_txn": amt / (txn + 1),
        "bal_age": bal / (age + 1),
        "risk_hrs": risk / (hrs + 1e-6),
    }
    bm = all_df["flag_bitmask_64"].to_numpy(np.int32)
    for rn, r in ratios.items():
        push(np.log1p(np.abs(r)), f"log_{rn}")
        push(r - np.floor(r), f"frac_{rn}")
        step = np.rint(r * 1000.0)
        push(step % 2, f"parity_{rn}")
        # cross with bitmask (mod 64 already)
        push((step.astype(np.int64) % 10) * 64 + (bm % 64), f"cross_{rn}_bm")

    X_all = np.column_stack(feat_cols)
    del feat_cols
    gc.collect()
    log(f"  base feature matrix {X_all.shape} ~{X_all.nbytes/1e9:.2f}GB free={free_gb():.2f}GB")

    # Split back
    X_tr = X_all[:ntr]
    X_te = X_all[ntr : ntr + nte]
    # orig features discarded after support (already baked into LOO/counts)
    del X_all, all_df
    gc.collect()

    # ---- 8-seed inner-fold smoothed TE for key cats × bitmask (m=10,30) ----
    # Done inside model loop per outer fold to avoid leakage; here we only package keys
    te_key_mats = {
        "mc_bm": (
            (
                tr["merchant_category"].astype(str)
                + "|"
                + pd.Series(bitmask64(tr)).astype(str)
            ).to_numpy(),
            (
                te["merchant_category"].astype(str)
                + "|"
                + pd.Series(bitmask64(te)).astype(str)
            ).to_numpy(),
        ),
        "ct_bm": (
            (tr["card_type"].astype(str) + "|" + pd.Series(bitmask64(tr)).astype(str)).to_numpy(),
            (te["card_type"].astype(str) + "|" + pd.Series(bitmask64(te)).astype(str)).to_numpy(),
        ),
        "am_bm": (
            (tr["auth_method"].astype(str) + "|" + pd.Series(bitmask64(tr)).astype(str)).to_numpy(),
            (te["auth_method"].astype(str) + "|" + pd.Series(bitmask64(te)).astype(str)).to_numpy(),
        ),
        "dt_bm": (
            (tr["device_type"].astype(str) + "|" + pd.Series(bitmask64(tr)).astype(str)).to_numpy(),
            (te["device_type"].astype(str) + "|" + pd.Series(bitmask64(te)).astype(str)).to_numpy(),
        ),
    }

    meta = {"feature_names": feat_names, "n_features_base": len(feat_names), "te_keys": list(te_key_mats.keys())}
    (ART / "feature_names.json").write_text(json.dumps(meta, indent=2))
    return X_tr, X_te, te_key_mats, feat_names


def ecdf_align(ref, *arrs):
    """Per-fold ECDF alignment of predictions onto ref distribution via PCHIP."""
    out = []
    for a in arrs:
        out.append(pchip_quantile_map(a, ref, n_knots=min(N_KNOTS, max(200, len(ref) // 50))))
    return out


def fit_v10_models(y, folds, X_tr, X_te, te_key_mats, z_ref_o, z_st9_t, lm_bank):
    log("PART2: fitting v10 residual learners (ridge / lgb / catboost)")
    import lightgbm as lgb
    from catboost import CatBoostRegressor

    ntr, nte = len(y), X_te.shape[0]
    z_cal_o = calibrate_logit_to_prevalence(z_ref_o, z_ref_o, y)
    # test: calibrate St9-mapped z into same prevalence space
    z_cal_t = calibrate_logit_to_prevalence(z_st9_t, z_ref_o, y)
    p_cal_o = expit(z_cal_o)

    families = {
        "v10_ridge_resid": (np.zeros(ntr), np.zeros(nte)),
        "v10_lgb_d3_bag5": (np.zeros(ntr), np.zeros(nte)),
        "v10_cb_d4_bag3": (np.zeros(ntr), np.zeros(nte)),
    }
    te_acc = {k: [] for k in families}

    lgb_seeds = [42, 137, 2026, 7, 99]
    cb_seeds = [42, 137, 2026]
    te_ms = [10.0, 30.0]
    te_seeds = list(range(8))

    for fi, (a, b) in enumerate(folds):
        log(f"  fold {fi}/5  free={free_gb():.2f}GB rss={rss_gb():.2f}GB")
        if free_gb() >= 0 and free_gb() < 1.2:
            log("  WARN low free RAM — forcing gc")
            gc.collect()

        prior = float(y[a].mean())
        # TE columns for this fold (8-seed bag of M-estimate)
        te_blocks_a, te_blocks_b, te_blocks_t = [], [], []
        for kn, (ktr, kte) in te_key_mats.items():
            for m in te_ms:
                bag_a, bag_b, bag_t = [], [], []
                for sd in te_seeds:
                    rs = np.random.RandomState(1000 * fi + 17 * sd + int(m))
                    idx = rs.choice(len(a), size=int(0.9 * len(a)), replace=False)
                    aa = a[idx]
                    # fit TE on 90% subsample; map onto fold-train / fold-val / test
                    _, va, _ = m_estimate_te(ktr[aa], y[aa], ktr[a], kte, m, prior)
                    _, vb, vt = m_estimate_te(ktr[aa], y[aa], ktr[b], kte, m, prior)
                    bag_a.append(va)
                    bag_b.append(vb)
                    bag_t.append(vt)
                te_blocks_a.append(np.mean(bag_a, 0))
                te_blocks_b.append(np.mean(bag_b, 0))
                te_blocks_t.append(np.mean(bag_t, 0))

        Xa = np.column_stack([X_tr[a]] + te_blocks_a).astype(np.float32)
        Xb = np.column_stack([X_tr[b]] + te_blocks_b).astype(np.float32)
        Xt = np.column_stack([X_te] + te_blocks_t).astype(np.float32)
        del te_blocks_a, te_blocks_b, te_blocks_t
        gc.collect()

        sc = StandardScaler()
        Xa_s = sc.fit_transform(np.nan_to_num(Xa, 0.0)).astype(np.float32)
        Xb_s = sc.transform(np.nan_to_num(Xb, 0.0)).astype(np.float32)
        Xt_s = sc.transform(np.nan_to_num(Xt, 0.0)).astype(np.float32)

        resid_a = y[a].astype(np.float64) - expit(z_cal_o[a])

        # --- Ridge ---
        ridge = Ridge(alpha=500.0)
        ridge.fit(Xa_s, resid_a)
        pa = ridge.predict(Xa_s)
        pb = ridge.predict(Xb_s)
        pt = ridge.predict(Xt_s)
        pb_al, pt_al = ecdf_align(pa, pb, pt)
        families["v10_ridge_resid"][0][b] = pb_al
        te_acc["v10_ridge_resid"].append(pt_al)

        # --- LightGBM bag5 with init_score = z_cal ---
        # predict(raw_score=True) returns boosted residual margin (init NOT included on new data)
        params = dict(
            objective="binary",
            metric="auc",
            learning_rate=0.025,
            num_leaves=7,
            max_depth=3,
            feature_fraction=0.25,
            bagging_fraction=0.85,
            bagging_freq=1,
            min_child_samples=150,
            lambda_l2=30.0,
            verbosity=-1,
            num_threads=2,
        )
        lgb_b, lgb_t, lgb_a = [], [], []
        for sd in lgb_seeds:
            dtr = lgb.Dataset(
                Xa,
                label=y[a],
                init_score=z_cal_o[a],
                free_raw_data=False,
            )
            booster = lgb.train({**params, "seed": sd}, dtr, num_boost_round=350)
            lgb_a.append(booster.predict(Xa, raw_score=True))
            lgb_b.append(booster.predict(Xb, raw_score=True))
            lgb_t.append(booster.predict(Xt, raw_score=True))
            del booster, dtr
        pa = np.mean(lgb_a, 0)
        pb = np.mean(lgb_b, 0)
        pt = np.mean(lgb_t, 0)
        pb_al, pt_al = ecdf_align(pa, pb, pt)
        families["v10_lgb_d3_bag5"][0][b] = pb_al
        te_acc["v10_lgb_d3_bag5"].append(pt_al)
        del lgb_a, lgb_b, lgb_t
        gc.collect()

        # --- CatBoost bag3 on residual target (baseline = z_cal via residualization) ---
        cb_b, cb_t, cb_a = [], [], []
        for sd in cb_seeds:
            model = CatBoostRegressor(
                loss_function="RMSE",
                depth=4,
                l2_leaf_reg=25,
                learning_rate=0.03,
                iterations=400,
                random_seed=sd,
                thread_count=2,
                verbose=False,
            )
            model.fit(Xa, resid_a)
            cb_a.append(model.predict(Xa))
            cb_b.append(model.predict(Xb))
            cb_t.append(model.predict(Xt))
            del model
        pa = np.mean(cb_a, 0)
        pb = np.mean(cb_b, 0)
        pt = np.mean(cb_t, 0)
        pb_al, pt_al = ecdf_align(pa, pb, pt)
        families["v10_cb_d4_bag3"][0][b] = pb_al
        te_acc["v10_cb_d4_bag3"].append(pt_al)
        del cb_a, cb_b, cb_t, Xa, Xb, Xt, Xa_s, Xb_s, Xt_s
        gc.collect()
        log(f"  fold {fi} done rss={rss_gb():.2f}GB")

    for k in families:
        families[k] = (families[k][0], np.mean(te_acc[k], 0))
        np.save(ART / f"{k}_oof.npy", families[k][0].astype(np.float32))
        np.save(ART / f"{k}_test.npy", families[k][1].astype(np.float32))
        log(f"  saved {k}")

    return families, z_cal_o, z_cal_t


def audit_family(name, oof, test, y, z_ref_o, z_st9_t, s9, folds, lm_bank):
    """Orth vs z_ref + LM controls; full forensic gate table."""
    names = list(lm_bank.keys())
    Xo = np.column_stack([z_ref_o] + [lm_bank[n][0] for n in names])
    Xt = np.column_stack([z_st9_t] + [lm_bank[n][1] for n in names])
    # oof/test are residual-scale already; still re-orth against controls
    zo, zt = oof.astype(np.float64), test.astype(np.float64)
    ro, rt, rtr = fold_orth_multi(zo, zt, Xo, Xt, folds)
    nest = nested_delta(y, z_ref_o, ro, folds)
    diag = resid_diagnostics(ro, rt, rtr, s9, expit(z_st9_t + rt))
    # spearman of injected direction
    # find w at 0.5e-4 for reporting 1-rho of unit residual inject at nest w
    w = nest["w"]
    rho_dist = (1.0 - float(spearmanr(z_st9_t + w * rt, z_st9_t).correlation)) * 1e4
    diag["spearman_dist_st9_test_1e4"] = round(rho_dist, 3)
    steps = []
    for td in TARGET_RHO:
        ww, got = weight_for_spearman_dist(z_st9_t, rt, td)
        steps.append({"target_1e4": round(td * 1e4, 4), "achieved_1e4": round(got * 1e4, 4), **nested_at_w(y, z_ref_o, ro, folds, ww)})
    row = {
        "name": name,
        **nest,
        "dispersion_ratio_raw": diag["dispersion_ratio_raw"],
        "ks_raw": diag["ks_raw"],
        "spearman_dist_st9_test_1e4": diag["spearman_dist_st9_test_1e4"],
        "steps": steps,
        "gate_pass": (
            nest["delta_1e5"] >= 0.5
            and nest["pos"] >= 4
            and 0.85 <= diag["dispersion_ratio_raw"] <= 1.15
            and diag["ks_raw"] < 0.015
        ),
    }
    np.save(ART / f"{name}_orth_LM_oof.npy", ro.astype(np.float32))
    np.save(ART / f"{name}_orth_LM_test.npy", rt.astype(np.float32))
    return row


def write_report(part1, part2, base_auc, elapsed):
    md = [
        "# Step 2 — LM Orth + v10 Transductive Bulsah (NO_SUBMIT)",
        "",
        f"Elapsed **{elapsed:.1f}s**. z_ref_o AUC **{base_auc:.7f}**.",
        "",
        "## Part 1 — soft_diff / gb_soft double-orth vs LM bank",
        "",
        "Dropped permanently: `idf_p3`, `clean035`.",
        "",
    ]
    for r in part1:
        md.append(f"### {r['name']}")
        md.append("")
        md.append(
            f"- Free-grid nested: Δ={r['free_grid']['delta_1e5']}e-5 w={r['free_grid']['w']} "
            f"{r['free_grid']['pos']}/5 | disp={r['dispersion_ratio_raw']} KS={r['ks_raw']}"
        )
        md.append(
            f"- vs z_ref only (pre-LM): Δ={r['base_vs_zref_only']['delta_1e5']}e-5 "
            f"{r['base_vs_zref_only']['pos']}/5"
        )
        md.append(f"- max|corr| vs controls: {max(abs(v) for v in r['corr_vs_controls'].values()):.2e}")
        md.append("")
        md.append("| target 1-ρ e-4 | achieved | w | ΔOOF e-5 | pos | disp | KS |")
        md.append("|---:|---:|---:|---:|---:|---:|---:|")
        for s in r["steps"]:
            md.append(
                f"| {s['target_1e4']} | {s['achieved_1e4']} | {s['w']} | {s['delta_1e5']} | "
                f"{s['pos']}/5 | {s['dispersion_ratio_raw']} | {s['ks_raw']} |"
            )
        md.append("")

    md += ["## Part 2 — v10_transductive_bulsah audit", ""]
    md.append("| member | ΔOOF e-5 | w | pos | disp | KS | 1-ρ e-4 | gate |")
    md.append("|---|---:|---:|---:|---:|---:|---:|:---:|")
    for r in part2:
        md.append(
            f"| {r['name']} | {r['delta_1e5']} | {r['w']} | {r['pos']}/5 | "
            f"{r['dispersion_ratio_raw']} | {r['ks_raw']} | {r['spearman_dist_st9_test_1e4']} | "
            f"{'PASS' if r['gate_pass'] else 'fail'} |"
        )
    md += ["", f"Artifacts: `{ART}`", ""]
    (OUT / "STEP2_LM_ORTH_V10.md").write_text("\n".join(md))
    (OUT / "step2_lm_orth_v10.json").write_text(
        json.dumps({"part1": part1, "part2": part2, "base_auc": base_auc, "elapsed_s": elapsed}, indent=2)
    )
    log("wrote STEP2_LM_ORTH_V10.md")


def main():
    log("STEP2 START (STRICT NO_SUBMIT)")
    log(f"free_gb≈{free_gb():.2f}")
    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    test_ids = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])["id"].to_numpy()
    ntr, nte = len(y), len(test_ids)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(ntr), y))

    log("building z_ref / St9 anchors...")
    z_ref_o, z_st9_t, s9, base_auc = build_z_ref(y, test_ids, folds)
    log(f"z_ref_o AUC={base_auc:.7f}")

    log("loading LM bank...")
    lm_bank = load_lm_bank(y, folds, ntr, nte, test_ids)
    log(f"LM members loaded: {list(lm_bank.keys())}")

    part1, arts = part1_lm_orth(y, folds, z_ref_o, z_st9_t, s9, lm_bank)

    # Part 2
    log("loading frames for v10...")
    usecols = None  # all
    tr = pd.read_csv(ROOT / "data/train.csv")
    te = pd.read_csv(ROOT / "data/test.csv")
    orig = pd.read_csv(ROOT / "original/credit_card_fraud_2026.csv")
    # drop id from te for feature parity where needed
    X_tr, X_te, te_key_mats, feat_names = build_v10_features(
        tr, te, orig, z_ref_o, z_st9_t, y
    )
    del tr, te, orig
    gc.collect()

    families, z_cal_o, z_cal_t = fit_v10_models(
        y, folds, X_tr, X_te, te_key_mats, z_ref_o, z_st9_t, lm_bank
    )
    del X_tr, X_te
    gc.collect()

    part2 = []
    for name, (oof, test) in families.items():
        row = audit_family(name, oof, test, y, z_ref_o, z_st9_t, s9, folds, lm_bank)
        part2.append(row)
        log(
            f"  AUDIT {name}: Δ={row['delta_1e5']}e-5 {row['pos']}/5 disp={row['dispersion_ratio_raw']} "
            f"KS={row['ks_raw']} gate={row['gate_pass']}"
        )

    # combination: equal-weight orth residuals of the three
    Uo = np.column_stack([np.load(ART / f"{n}_orth_LM_oof.npy") for n in families])
    Ut = np.column_stack([np.load(ART / f"{n}_orth_LM_test.npy") for n in families])
    # re-orth the sum vs controls
    combo_o = Uo.mean(1)
    combo_t = Ut.mean(1)
    row = audit_family("v10_combo3", combo_o, combo_t, y, z_ref_o, z_st9_t, s9, folds, lm_bank)
    part2.append(row)
    log(
        f"  AUDIT v10_combo3: Δ={row['delta_1e5']}e-5 {row['pos']}/5 disp={row['dispersion_ratio_raw']} "
        f"KS={row['ks_raw']} gate={row['gate_pass']}"
    )

    write_report(part1, part2, base_auc, time.time() - T0)
    log("STEP2_COMPLETE")
    print("STEP2_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        (OUT / "STEP2_FAIL.txt").write_text(traceback.format_exc())
        raise
