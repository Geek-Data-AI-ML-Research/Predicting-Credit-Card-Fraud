#!/usr/bin/env python3
"""Step 3b: Surgical no-zanch probe (<5 min). STRICT NO_SUBMIT.

1) v10_ridge_bulsah_only = Bulsah cols + support-gated TE (no zanch, no loo_mean)
2) PCHIP-align soft_diff_orth_LM test→oof (disp=1, KS≈0)
3) Audit zanch_* train/test std+KS; optional PCHIP eb20 mean-only rescue
4) Stage clean Slot-1/2 CSVs (no kaggle submit)
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
os.environ["NO_SUBMIT"] = "1"

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import ks_2samp, spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from gm_top3_strike3_expanded7f import (
    PRED,
    SUB,
    TARGET_MEAN,
    calibrate_logit_to_prevalence,
    fast_auc,
    pchip_quantile_map,
    zrk,
)
from recon_and_forensic_audit import N_KNOTS, build_z_ref, nested_delta, resid_diagnostics
from step2_lm_orth_and_v10_transductive import (
    ART,
    bitmask64,
    fold_orth_multi,
    load_lm_bank,
    m_estimate_te,
    nested_at_w,
    weight_for_spearman_dist,
)

OUT = ROOT / "logs" / "overnight"
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
STATUS = OUT / "STEP3B_STATUS.json"
NO_SUBMIT = True


def log(msg: str):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)
    try:
        STATUS.write_text(
            json.dumps(
                {"t_s": round(time.time() - T0, 1), "msg": msg, "NO_SUBMIT": NO_SUBMIT, "ts": time.strftime("%Y-%m-%d %H:%M:%S")},
                indent=2,
            )
        )
    except Exception:
        pass


def fold_deltas(y, z_base, r, folds, w):
    out = []
    for _, b in folds:
        out.append(
            round(
                (float(fast_auc(y[b], (z_base + w * r)[b])) - float(fast_auc(y[b], z_base[b]))) * 1e5,
                2,
            )
        )
    return out, sum(1 for x in out if x > 0)


def classify_cols(names):
    bulsah, zanch, zanch_eb20_mean, loo, support = [], [], [], [], []
    for i, n in enumerate(names):
        if n.startswith("zanch_"):
            zanch.append(i)
            if n.startswith("zanch_eb20_") and "disp" not in n:
                zanch_eb20_mean.append(i)
        elif n.startswith("loo_mean_"):
            loo.append(i)
        elif n.startswith("logcnt_") or n.startswith("rare"):
            support.append(i)
        elif any(
            n.startswith(p)
            for p in ("frac_", "cent", "first2_", "last2_", "step", "log_", "parity_", "cross_")
        ):
            bulsah.append(i)
    return {
        "bulsah": np.array(bulsah, dtype=np.int32),
        "zanch": np.array(zanch, dtype=np.int32),
        "zanch_eb20_mean": np.array(zanch_eb20_mean, dtype=np.int32),
        "loo": np.array(loo, dtype=np.int32),
        "support": np.array(support, dtype=np.int32),
    }


def support_gate_te(te_tr, te_te, keys_tr, keys_te):
    """te * (test_cnt>=2) / sqrt(1+|log((tr_freq+eps)/(te_freq+eps))|)."""
    eps = 1e-5
    g_tr = pd.Series(keys_tr)
    g_te = pd.Series(keys_te)
    # joint freqs on train / test separately (support existence)
    ct_tr = g_tr.groupby(g_tr).transform("count").to_numpy(np.float64)
    # map test key counts
    te_counts = g_te.value_counts()
    ct_te_on_tr = g_tr.map(te_counts).fillna(0).to_numpy(np.float64)
    ct_te = g_te.map(te_counts).fillna(0).to_numpy(np.float64)
    # train freqs for keys appearing in test rows
    tr_counts = g_tr.value_counts()
    ct_tr_on_te = g_te.map(tr_counts).fillna(0).to_numpy(np.float64)

    def gate(te, ct_local_tr, ct_local_te):
        # only keep signal when key appears >=2 times in TEST
        mask = (ct_local_te >= 2).astype(np.float64)
        ratio = np.log((ct_local_tr + eps) / (ct_local_te + eps))
        denom = np.sqrt(1.0 + np.abs(ratio))
        return te * mask / denom

    return gate(te_tr, ct_tr, ct_te_on_tr), gate(te_te, ct_tr_on_te, ct_te)


def build_te_keys(tr, te):
    mats = {}
    # categorical × bitmask
    for name, col in [
        ("mc_bm", "merchant_category"),
        ("ct_bm", "card_type"),
        ("am_bm", "auth_method"),
        ("dt_bm", "device_type"),
    ]:
        mats[name] = (
            (tr[col].astype(str) + "|" + pd.Series(bitmask64(tr)).astype(str)).to_numpy(),
            (te[col].astype(str) + "|" + pd.Series(bitmask64(te)).astype(str)).to_numpy(),
        )
    # bulsah step-grid × bitmask (2D) — cheap discrete keys from continuous
    bm_tr = bitmask64(tr)
    bm_te = bitmask64(te)
    for col, step in [("velocity_score", 0.1), ("merchant_risk_score", 0.02), ("amount_usd", 1.0)]:
        s_tr = (np.rint(tr[col].to_numpy(np.float64) / step).astype(np.int64) % 50)
        s_te = (np.rint(te[col].to_numpy(np.float64) / step).astype(np.int64) % 50)
        mats[f"step_{col}_bm"] = (
            (pd.Series(s_tr).astype(str) + "|" + pd.Series(bm_tr).astype(str)).to_numpy(),
            (pd.Series(s_te).astype(str) + "|" + pd.Series(bm_te).astype(str)).to_numpy(),
        )
    return mats


def fit_bulsah_ridge(X_tr, X_te, bulsah_idx, te_mats, y, folds, z_cal_o):
    """Ridge on bulsah cols + support-gated TE; also LR residual check."""
    ntr, nte = len(y), X_te.shape[0]
    oof = np.zeros(ntr)
    te_folds = []
    oof_lr = np.zeros(ntr)
    te_lr = []
    te_ms = [10.0, 30.0]
    # 2 seeds for speed (<5min)
    te_seeds = [0, 3]

    Xp_tr = np.asarray(X_tr[:, bulsah_idx], dtype=np.float32)
    Xp_te = np.asarray(X_te[:, bulsah_idx], dtype=np.float32)

    for fi, (a, b) in enumerate(folds):
        prior = float(y[a].mean())
        blocks_a, blocks_b, blocks_t = [], [], []
        for kn, (ktr, kte) in te_mats.items():
            for m in te_ms:
                bag_a, bag_b, bag_t = [], [], []
                for sd in te_seeds:
                    rs = np.random.RandomState(1000 * fi + 17 * sd + int(m))
                    idx = rs.choice(len(a), size=int(0.85 * len(a)), replace=False)
                    aa = a[idx]
                    _, va, vt = m_estimate_te(ktr[aa], y[aa], ktr[a], kte, m, prior)
                    _, vb, vt2 = m_estimate_te(ktr[aa], y[aa], ktr[b], kte, m, prior)
                    va_g, vt_g = support_gate_te(va, vt, ktr[a], kte)
                    vb_g, _ = support_gate_te(vb, vt2, ktr[b], kte)
                    bag_a.append(va_g)
                    bag_b.append(vb_g)
                    bag_t.append(vt_g)
                blocks_a.append(np.mean(bag_a, 0))
                blocks_b.append(np.mean(bag_b, 0))
                blocks_t.append(np.mean(bag_t, 0))

        Xa = np.column_stack([Xp_tr[a]] + blocks_a).astype(np.float32)
        Xb = np.column_stack([Xp_tr[b]] + blocks_b).astype(np.float32)
        Xt = np.column_stack([Xp_te] + blocks_t).astype(np.float32)
        resid_a = y[a].astype(np.float64) - expit(z_cal_o[a])

        sc = StandardScaler()
        Xa_s = sc.fit_transform(np.nan_to_num(Xa, 0.0))
        Xb_s = sc.transform(np.nan_to_num(Xb, 0.0))
        Xt_s = sc.transform(np.nan_to_num(Xt, 0.0))

        ridge = Ridge(alpha=500.0).fit(Xa_s, resid_a)
        pa, pb, pt = ridge.predict(Xa_s), ridge.predict(Xb_s), ridge.predict(Xt_s)
        pb = pchip_quantile_map(pb, pa, n_knots=min(N_KNOTS, 1500))
        pt = pchip_quantile_map(pt, pa, n_knots=min(N_KNOTS, 1500))
        oof[b] = pb
        te_folds.append(pt)

        # LogisticRegression probability residual vs calibrated prevalence
        lr = LogisticRegression(C=0.5, solver="lbfgs", max_iter=200)
        lr.fit(Xa_s, y[a])
        la = lr.predict_proba(Xa_s)[:, 1] - expit(z_cal_o[a])
        lb = lr.predict_proba(Xb_s)[:, 1] - expit(z_cal_o[b])
        lt = lr.predict_proba(Xt_s)[:, 1] - float(np.mean(expit(z_cal_o)))
        lb = pchip_quantile_map(lb, la, n_knots=min(N_KNOTS, 1500))
        lt = pchip_quantile_map(lt, la, n_knots=min(N_KNOTS, 1500))
        oof_lr[b] = lb
        te_lr.append(lt)

        del Xa, Xb, Xt, Xa_s, Xb_s, Xt_s
        gc.collect()
        log(f"  bulsah ridge fold {fi}/5 done")

    return (
        oof,
        np.mean(te_folds, 0),
        oof_lr,
        np.mean(te_lr, 0),
    )


def logit_safe(p):
    from scipy.special import logit

    return logit(np.clip(p, 1e-6, 1 - 1e-6))


def audit_member(name, oof, test, y, z_ref_o, z_st9_t, s9, folds, lm_bank, save_stem=None):
    names = list(lm_bank.keys())
    Xo = np.column_stack([z_ref_o] + [lm_bank[n][0] for n in names])
    Xt = np.column_stack([z_st9_t] + [lm_bank[n][1] for n in names])
    ro, rt, rtr = fold_orth_multi(oof.astype(np.float64), test.astype(np.float64), Xo, Xt, folds)
    nest = nested_delta(y, z_ref_o, ro, folds)
    diag = resid_diagnostics(ro, rt, rtr, s9, expit(z_st9_t + rt))
    fd, pos = fold_deltas(y, z_ref_o, ro, folds, nest["w"] if abs(nest["w"]) > 1e-12 else 0.02)
    # force pos at nest w
    if abs(nest["w"]) > 1e-12:
        fd, pos = fold_deltas(y, z_ref_o, ro, folds, nest["w"])
    row = {
        "name": name,
        **nest,
        "fold_deltas_1e5": fd,
        "pos_at_w": pos,
        "dispersion_ratio_raw": diag["dispersion_ratio_raw"],
        "ks_raw": diag["ks_raw"],
        "gate_passonly": (
            nest["pos"] >= 4
            and nest["delta_1e5"] >= 0.5
            and 0.95 <= diag["dispersion_ratio_raw"] <= 1.05
            and diag["ks_raw"] < 0.005
        ),
    }
    if save_stem:
        np.save(ART / f"{save_stem}_oof.npy", ro.astype(np.float32))
        np.save(ART / f"{save_stem}_test.npy", rt.astype(np.float32))
        np.save(ART / f"{save_stem}_test_raw.npy", rtr.astype(np.float32))
    log(
        f"  {name}: Δ={row['delta_1e5']}e-5 w={row['w']} {row['pos']}/5 "
        f"disp={row['dispersion_ratio_raw']} KS={row['ks_raw']} passonly={row['gate_passonly']}"
    )
    return row, ro, rt, rtr


def pchip_align_residual(ro, rt_raw):
    """Map raw test residual onto OOF residual distribution → disp=1, KS≈0."""
    rt = pchip_quantile_map(rt_raw, ro, n_knots=N_KNOTS)
    # exact unit match
    rt = (rt - float(np.mean(rt))) / (float(np.std(rt) + 1e-12)) * float(np.std(ro))
    rt = rt + float(np.mean(ro))
    disp = float(np.std(rt) / (np.std(ro) + 1e-12))
    ks = float(ks_2samp(ro, rt).statistic)
    return rt, disp, ks


def write_sub(path, test_ids, z_test, z_ref_o, y):
    p = expit(calibrate_logit_to_prevalence(z_test, z_ref_o, y, target_mean=TARGET_MEAN))
    assert len(p) == 299984 and np.isfinite(p).all()
    pd.DataFrame({"id": test_ids, "is_fraud": p}).to_csv(path, index=False)
    log(f"  staged {path.name} mean={float(p.mean()):.6f} (NO_SUBMIT)")
    return {"path": str(path), "mean": float(p.mean()), "n": len(p)}


def main():
    log("STEP3B START surgical no-zanch probe (NO_SUBMIT)")
    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    test_ids = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])["id"].to_numpy()
    ntr, nte = len(y), len(test_ids)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(ntr), y))

    z_ref_o, z_st9_t, s9, base = build_z_ref(y, test_ids, folds)
    log(f"z_ref_o={base:.7f}")
    lm_bank = load_lm_bank(y, folds, ntr, nte, test_ids)

    names = json.loads((ART / "feature_names.json").read_text())["feature_names"]
    cols = classify_cols(names)
    log(
        f"cols bulsah={len(cols['bulsah'])} zanch={len(cols['zanch'])} "
        f"eb20_mean={len(cols['zanch_eb20_mean'])} loo={len(cols['loo'])} (loo EXCLUDED)"
    )

    X_tr = np.memmap(ART / "X_tr_f32.dat", dtype=np.float32, mode="r", shape=(ntr, len(names)))
    X_te = np.memmap(ART / "X_te_f32.dat", dtype=np.float32, mode="r", shape=(nte, len(names)))

    # ----- 3) zanch train/test bias audit -----
    log("zanch_* train-vs-test variance audit")
    zanch_rows = []
    bad_std = bad_ks = 0
    for j in cols["zanch"]:
        xo = np.asarray(X_tr[:, j], dtype=np.float64)
        xt = np.asarray(X_te[:, j], dtype=np.float64)
        sr = float(np.std(xt) / (np.std(xo) + 1e-12))
        ks = float(ks_2samp(xo, xt).statistic)
        if sr < 0.92 or sr > 1.08:
            bad_std += 1
        if ks > 0.01:
            bad_ks += 1
        zanch_rows.append({"feat": names[j], "std_ratio": round(sr, 4), "ks": round(ks, 6)})
    log(f"  zanch: {bad_std}/{len(cols['zanch'])} bad std_ratio; {bad_ks}/{len(cols['zanch'])} KS>0.01")

    # eb20 mean-only after per-col PCHIP
    eb_ok = []
    for j in cols["zanch_eb20_mean"]:
        xo = np.asarray(X_tr[:, j], dtype=np.float64)
        xt = np.asarray(X_te[:, j], dtype=np.float64)
        xt_al = pchip_quantile_map(xt, xo, n_knots=1500)
        sr = float(np.std(xt_al) / (np.std(xo) + 1e-12))
        ks = float(ks_2samp(xo, xt_al).statistic)
        if 0.97 <= sr <= 1.03 and ks < 0.005:
            eb_ok.append(j)
    log(f"  zanch_eb20_mean PCHIP-clean: {len(eb_ok)}/{len(cols['zanch_eb20_mean'])}")

    # ----- 1) bulsah-only ridge + support-gated TE -----
    log("fit v10_ridge_bulsah_only (+ support-gated TE)")
    tr = pd.read_csv(ROOT / "data/train.csv")
    te = pd.read_csv(ROOT / "data/test.csv")
    te_mats = build_te_keys(tr, te)
    del tr, te
    gc.collect()
    z_cal_o = calibrate_logit_to_prevalence(z_ref_o, z_ref_o, y)
    ridge_o, ridge_t, lr_o, lr_t = fit_bulsah_ridge(
        X_tr, X_te, cols["bulsah"], te_mats, y, folds, z_cal_o
    )
    np.save(ART / "v10_ridge_bulsah_only_oof.npy", ridge_o.astype(np.float32))
    np.save(ART / "v10_ridge_bulsah_only_test.npy", ridge_t.astype(np.float32))

    row_b, ro_b, rt_b, rtr_b = audit_member(
        "v10_ridge_bulsah_only",
        ridge_o,
        ridge_t,
        y,
        z_ref_o,
        z_st9_t,
        s9,
        folds,
        lm_bank,
        save_stem="v10_bulsah_clean_orth_LM",
    )
    row_lr, _, _, _ = audit_member(
        "v10_lr_bulsah_only", lr_o, lr_t, y, z_ref_o, z_st9_t, s9, folds, lm_bank
    )

    # optional: PCHIP eb20 means + bulsah ridge residual blend if eb_ok
    row_zb = None
    if len(eb_ok) >= 5 and row_b["gate_passonly"]:
        # already passonly — skip heavy
        pass
    elif len(eb_ok) >= 8:
        log(f"testing PCHIP zanch_eb20_mean ({len(eb_ok)}) + bulsah residual")
        # quick ridge on eb20 means alone → orth → blend with bulsah if improves disp
        oof_z = np.zeros(ntr)
        te_z = []
        Ze_tr = np.asarray(X_tr[:, eb_ok], dtype=np.float32)
        Ze_te = np.column_stack(
            [pchip_quantile_map(np.asarray(X_te[:, j], dtype=np.float64), np.asarray(X_tr[:, j], dtype=np.float64), n_knots=800) for j in eb_ok]
        ).astype(np.float32)
        for fi, (a, b) in enumerate(folds):
            resid_a = y[a].astype(np.float64) - expit(z_cal_o[a])
            sc = StandardScaler().fit(np.nan_to_num(Ze_tr[a], 0.0))
            ridge = Ridge(alpha=500.0).fit(sc.transform(np.nan_to_num(Ze_tr[a], 0.0)), resid_a)
            pa = ridge.predict(sc.transform(np.nan_to_num(Ze_tr[a], 0.0)))
            pb = pchip_quantile_map(ridge.predict(sc.transform(np.nan_to_num(Ze_tr[b], 0.0))), pa, n_knots=800)
            pt = pchip_quantile_map(ridge.predict(sc.transform(np.nan_to_num(Ze_te, 0.0))), pa, n_knots=800)
            oof_z[b] = pb
            te_z.append(pt)
        te_z = np.mean(te_z, 0)
        # blend 50/50 with bulsah before orth
        blend_o = 0.5 * ridge_o + 0.5 * oof_z
        blend_t = 0.5 * ridge_t + 0.5 * te_z
        row_zb, ro_zb, rt_zb, rtr_zb = audit_member(
            "v10_bulsah_plus_pchip_eb20",
            blend_o,
            blend_t,
            y,
            z_ref_o,
            z_st9_t,
            s9,
            folds,
            lm_bank,
        )
    else:
        log("eb20 PCHIP-clean subset too small / skipped blend")

    # ----- 2) PCHIP-align soft_diff -----
    log("PCHIP-align soft_diff_orth_LM")
    ro_s = np.load(ART / "soft_diff_orth_LM_oof.npy").astype(np.float64)
    rtr_s = np.load(ART / "soft_diff_orth_LM_test_raw.npy").astype(np.float64)
    # prefer raw; if missing shape ok
    rt_s, disp_s, ks_s = pchip_align_residual(ro_s, rtr_s)
    np.save(ART / "soft_diff_orth_LM_test_pchip.npy", rt_s.astype(np.float32))
    log(f"  soft_diff PCHIP: disp={disp_s:.6f} KS={ks_s:.6f}")

    soft_steps = []
    for td in [0.20e-4, 0.30e-4, 0.35e-4, 0.50e-4]:
        w, got = weight_for_spearman_dist(z_st9_t, rt_s, td)
        nest = nested_at_w(y, z_ref_o, ro_s, folds, w)
        fd, pos = fold_deltas(y, z_ref_o, ro_s, folds, w)
        soft_steps.append(
            {
                "target_1e4": round(td * 1e4, 4),
                "achieved_1e4": round(got * 1e4, 4),
                "w": round(w, 6),
                **nest,
                "fold_deltas_1e5": fd,
                "pos": pos,
                "disp": round(disp_s, 6),
                "ks": round(ks_s, 6),
            }
        )
        log(
            f"  soft_diff_pchip @1-ρ={td*1e4:.2f}e-4: Δ={nest['delta_1e5']} {pos}/5 folds={fd}"
        )

    # ----- stage CSVs -----
    staged = []
    # Slot1: soft_diff @ 0.35e-4
    w1, _ = weight_for_spearman_dist(z_st9_t, rt_s, 0.35e-4)
    nest1 = nested_at_w(y, z_ref_o, ro_s, folds, w1)
    fd1, pos1 = fold_deltas(y, z_ref_o, ro_s, folds, w1)
    meta1 = write_sub(
        SUB / "sub_strike10_slot1_softdiff_pchip_0p35.csv",
        test_ids,
        z_st9_t + w1 * rt_s,
        z_ref_o,
        y,
    )
    meta1.update(
        {
            "portfolio": "soft_diff_pchip",
            "target_1e4": 0.35,
            "w": w1,
            "delta_1e5": nest1["delta_1e5"],
            "pos": pos1,
            "folds": fd1,
            "disp": disp_s,
            "ks": ks_s,
        }
    )
    staged.append(meta1)

    # Slot2: soft + bulsah only if PassOnly clears
    slot2 = None
    use_b = row_b if row_b["gate_passonly"] else (row_zb if row_zb and row_zb["gate_passonly"] else None)
    if use_b and use_b["name"] == "v10_ridge_bulsah_only":
        ro2, rt2 = ro_b, rt_b
    elif use_b and row_zb:
        ro2, rt2 = ro_zb, rt_zb
    else:
        ro2 = rt2 = None

    if ro2 is not None:
        # equal-risk blend then target 0.45e-4
        ro_c = 0.5 * ro_s + 0.5 * ro2
        rt_c_raw = 0.5 * rt_s + 0.5 * rt2
        sd = float(np.std(ro_c) + 1e-12)
        ro_c = (ro_c - float(np.mean(ro_c))) / sd
        rt_c, disp_c, ks_c = pchip_align_residual(ro_c, rt_c_raw / sd)
        w2, _ = weight_for_spearman_dist(z_st9_t, rt_c, 0.45e-4)
        nest2 = nested_at_w(y, z_ref_o, ro_c, folds, w2)
        fd2, pos2 = fold_deltas(y, z_ref_o, ro_c, folds, w2)
        meta2 = write_sub(
            SUB / "sub_strike10_slot2_clean_combo_0p45.csv",
            test_ids,
            z_st9_t + w2 * rt_c,
            z_ref_o,
            y,
        )
        meta2.update(
            {
                "portfolio": f"soft+{use_b['name']}",
                "target_1e4": 0.45,
                "w": w2,
                "delta_1e5": nest2["delta_1e5"],
                "pos": pos2,
                "folds": fd2,
                "disp": disp_c,
                "ks": ks_c,
                "bulsah_included": True,
            }
        )
        staged.append(meta2)
        slot2 = meta2
        log(f"  Slot2 INCLUDED bulsah ({use_b['name']})")
    else:
        log("  Slot2 SKIPPED — bulsah PassOnly gate FAILED (no zanch-masked combo)")
        # still write a soft-only @0.45 for reference under different name? User said only if gate passes.
        # Write a marker JSON instead of CSV.
        (OUT / "STEP3B_SLOT2_SKIPPED.json").write_text(
            json.dumps({"reason": "bulsah_passonly_failed", "bulsah": row_b, "eb20_blend": row_zb}, indent=2)
        )

    # report
    md = [
        "# Step 3b — Surgical no-zanch probe (NO_SUBMIT)",
        "",
        f"Elapsed **{time.time()-T0:.1f}s**. Holding `sub_strike10A` — PassOnly lift was zanch-masked.",
        "",
        "## Why hold Strike10A",
        "",
        "Dropping `zanch_*` killed ridge pure-FE (+6.03→0). Portfolio C `disp=0.958` was soft_diff (`disp=1.028`) masking PassOnly (`disp=0.914`).",
        "",
        "## zanch_* train/test bias",
        "",
        f"- bad std_ratio (<0.92 or >1.08): **{bad_std}/{len(cols['zanch'])}**",
        f"- KS>0.01: **{bad_ks}/{len(cols['zanch'])}**",
        f"- eb20 mean-only PCHIP-clean (0.97–1.03, KS<0.005): **{len(eb_ok)}/{len(cols['zanch_eb20_mean'])}**",
        "",
        "## Bulsah-only / soft_diff PCHIP",
        "",
        "| member | ΔOOF e-5 | pos | disp | KS | PassOnly gate |",
        "|---|---:|---:|---:|---:|:---:|",
        f"| v10_ridge_bulsah_only | {row_b['delta_1e5']} | {row_b['pos']}/5 | {row_b['dispersion_ratio_raw']} | {row_b['ks_raw']} | {row_b['gate_passonly']} |",
        f"| v10_lr_bulsah_only | {row_lr['delta_1e5']} | {row_lr['pos']}/5 | {row_lr['dispersion_ratio_raw']} | {row_lr['ks_raw']} | {row_lr['gate_passonly']} |",
    ]
    if row_zb:
        md.append(
            f"| bulsah+pchip_eb20 | {row_zb['delta_1e5']} | {row_zb['pos']}/5 | {row_zb['dispersion_ratio_raw']} | {row_zb['ks_raw']} | {row_zb['gate_passonly']} |"
        )
    md += [
        f"| soft_diff_orth_LM PCHIP | (see steps) |  | {disp_s:.4f} | {ks_s:.6f} | n/a |",
        "",
        "### soft_diff PCHIP step grid",
        "",
        "| 1-ρ e-4 | w | ΔOOF e-5 | folds | pos | disp | KS |",
        "|---:|---:|---:|---|---:|---:|---:|",
    ]
    for s in soft_steps:
        md.append(
            f"| {s['target_1e4']} | {s['w']} | {s['delta_1e5']} | {s['fold_deltas_1e5']} | {s['pos']}/5 | {s['disp']} | {s['ks']} |"
        )
    md += ["", "## Staged (NOT submitted)", ""]
    for s in staged:
        md.append(
            f"- `{Path(s['path']).name}`: Δ={s.get('delta_1e5')}e-5 pos={s.get('pos')}/5 "
            f"disp={s.get('disp')} KS={s.get('ks')} @1-ρ={s.get('target_1e4')}e-4"
        )
    if slot2 is None:
        md.append("- `sub_strike10_slot2_clean_combo_0p45.csv`: **NOT WRITTEN** (bulsah failed PassOnly alone)")
    md += ["", f"`NO_SUBMIT={NO_SUBMIT}`. Await confirmation before Slot-1 probe.", ""]
    (OUT / "STEP3B_SURGICAL.md").write_text("\n".join(md))
    payload = {
        "zanch_audit": {"bad_std": bad_std, "bad_ks": bad_ks, "n": len(cols["zanch"]), "eb20_pchip_clean": len(eb_ok)},
        "zanch_rows_sample": sorted(zanch_rows, key=lambda r: r["ks"], reverse=True)[:20],
        "bulsah": row_b,
        "lr_bulsah": row_lr,
        "eb20_blend": row_zb,
        "soft_steps": soft_steps,
        "soft_pchip": {"disp": disp_s, "ks": ks_s},
        "staged": staged,
        "slot2_written": slot2 is not None,
        "NO_SUBMIT": NO_SUBMIT,
        "elapsed_s": time.time() - T0,
    }
    (OUT / "step3b_surgical.json").write_text(json.dumps(payload, indent=2))
    log(f"wrote STEP3B_SURGICAL.md elapsed={time.time()-T0:.1f}s")
    log("STEP3B_COMPLETE")
    print("STEP3B_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        (OUT / "STEP3B_FAIL.txt").write_text(traceback.format_exc())
        raise
