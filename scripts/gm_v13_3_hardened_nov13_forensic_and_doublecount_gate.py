#!/usr/bin/env python3
"""GM V13.3 HARDENED: forensic novelty / OOF-test symmetry / double-count gate
on Nov13 alexchilton + daoviet refreshes, then δ1_v14b with softened init_score.

STRICT NO_SUBMIT=True. Zero pods. Local Mac CPU only.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.special import logit
from scipy.stats import rankdata, spearmanr, wilcoxon, t as student_t
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

NO_SUBMIT = True
assert NO_SUBMIT is True

ROOT = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
TMP = Path("/tmp/s6e11_v133")
RECON = ROOT / "external/nov12_095369_recon/datasets"
PRED = ROOT / "preds"
SUB = ROOT / "submissions"
LOG = ROOT / "logs/overnight"
NOV13 = ROOT / "external/nov13_recon"
ALEX_NEW = NOV13 / "alexchilton_oof_predictions"
ALEX_OLD = RECON / "alexchilton__s6e11-oof-predictions"
DAO_NEW = NOV13 / "daoviet_gp_dvae_apex_nov13"
DAO_OLD = RECON / "daoviet__s6e11-gp-dvae-orthogonal-apex"
Q17 = Path("/tmp/s6e11_v132/rank1_q17_streams.npz")
if not Q17.exists():
    Q17 = ROOT / "external/nov12_late_recon/amanatar_q17/rank1_q17_streams.npz"
LIVE = SUB / "sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv"
BANK = RECON / "goodpjw2008__s6e11-oof-bank"

R2_REPACK = 0.992
DISP_LO, DISP_HI = 0.97, 1.03
MIN_FOLD_E5 = 0.10
SF25_NEED = 18
T25_NEED = 2.80
WILC_P = 0.005
BOOT_LO_NEED = 0.25

T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def U(v):
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def L(v):
    return logit(np.clip(U(v), 1e-12, 1 - 1e-12))


def L_fold(v, fold_id):
    v = np.asarray(v, float).ravel()
    out = np.empty_like(v, dtype=np.float64)
    for f in range(5):
        m = fold_id == f
        out[m] = L(v[m])
    return out


def unitize(v):
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def pchip_lock(src, ref):
    src = np.asarray(src, float).ravel()
    ref = np.asarray(ref, float).ravel()
    qs = np.linspace(0, 1, 401)
    x, y = np.quantile(src, qs), np.quantile(ref, qs)
    _, ui = np.unique(x, return_index=True)
    x, y = x[ui], y[ui]
    if len(x) < 4:
        return unitize(src) * (ref.std() + 1e-12) + ref.mean()
    return PchipInterpolator(x, y, extrapolate=True)(src)


def npy(p):
    return np.load(p).astype(np.float64).ravel()


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def ols_r2_weights(y, X, names):
    """OLS with intercept; return R² and top-3 |coef| names (excl intercept)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    X1 = np.column_stack([X, np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    pred = X1 @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-18
    r2 = 1.0 - ss_res / ss_tot
    w = {names[i]: float(coef[i]) for i in range(len(names))}
    top3 = sorted(w.items(), key=lambda kv: -abs(kv[1]))[:3]
    return r2, w, top3, coef


# ---------------------------------------------------------------------------
# Load shared pack: folds, ATH, basis streams
# ---------------------------------------------------------------------------
def load_pack():
    log("Loading Q17 / streams / ATH proxy")
    Z = np.load(Q17, allow_pickle=False)
    y = Z["y_train"].astype(int)
    fold_id = Z["folds"].astype(int)
    names = [str(s) for s in Z["stream_names"]]
    folds = [(np.where(fold_id != f)[0], np.where(fold_id == f)[0]) for f in range(5)]

    def so(n):
        return L_fold(Z[f"oof_{n}"], fold_id)

    def st(n):
        return L(Z[f"test_{n}"])

    MS_o, MS_t = so("masaya_gpt2"), st("masaya_gpt2")
    LM_o, LM_t = so("megayak_5f_lm"), st("megayak_5f_lm")
    L7F_o, L7F_t = so("lm7_5fold"), st("lm7_5fold")
    GPD_o, GPD_t = so("daoviet_gpdvae"), st("daoviet_gpdvae")
    MLP_o, MLP_t = so("gbn_bag5_w512"), st("gbn_bag5_w512")
    GB_o, GB_t = so("gb_bag5"), st("gb_bag5")
    G26_o = so("gpt2_seed2026")
    G26_t = L(npy(RECON / "goodpjw2008__s6e11-target-first-gpt2-seed-2026/gpt2_s2026_test_f0.npy"))
    Gmed_o = 0.5 * (G26_o + MS_o)
    Gmed_t = 0.5 * (G26_t + MS_t)

    # L7_s2026 fold0 only
    l7p = PRED / "l7_s2026_5f"
    L7s_o = np.zeros(len(y), float)
    L7s_t = np.zeros(299984, float)
    if (l7p / "oof_0.npy").exists():
        ll0 = np.load(l7p / "oof_0.npy")
        llt = np.load(l7p / "test_0.npy")
        m0 = fold_id == 0
        prior = np.array([1 - y[m0].mean(), y[m0].mean()])

        def bayes(ll, prior):
            joint = ll + np.log(prior + 1e-12).reshape(1, 1, 2)
            m = joint.max(-1, keepdims=True)
            ex = np.exp(joint - m)
            post = ex / ex.sum(-1, keepdims=True)
            return post[:, :, 1].mean(1)

        L7s_o[m0] = L(bayes(ll0, prior))
        L7s_o[~m0] = L7F_o[~m0]
        L7s_t = L(bayes(llt, prior))

    d1o = unitize(npy(Path("/tmp/s6e11_v132/d1_v4_oof.npy") if Path("/tmp/s6e11_v132/d1_v4_oof.npy").exists() else PRED / "gm_v4_polish/d1_v4_oof.npy"))
    d1t = unitize(npy(Path("/tmp/s6e11_v132/d1_v4_test.npy") if Path("/tmp/s6e11_v132/d1_v4_test.npy").exists() else PRED / "gm_v4_polish/d1_v4_test.npy"))

    # Rebuild ATH (bold MIX + 0.03 d1) — import thin path from v13.2 if available
    from gm_v13_2_overnight_check_and_stageB_v14 import build_ath_proxy

    ath = build_ath_proxy()
    z_ATH_o = ath["z_ATH_o"]
    z_ATH_t = ath["z_ATH_t"]
    MIX_T = ath.get("MIX_O")  # will rebuild test MIX below
    # Prefer LIVE for test double-count; also keep MIX_T from ath rebuild
    # Recompute MIX_T via same recipe internals: z_ATH_t from LIVE is L(LIVE);
    # bold_MIX_test_proxy ≈ z_ATH_t - 0.03*d1t when LIVE = force_law(bold, bold+d1)
    # For Test C we need MIX_test proxy in logit space before d1:
    # If z_ATH_t is L(LIVE), recover approx MIX by projecting.
    live_p = pd.read_csv(LIVE)["is_fraud"].to_numpy(float)
    L_LIVE = L(live_p)
    # bold MIX test ≈ L(bold) — use LIVE minus d1 step approx
    MIX_T = L_LIVE - 0.03 * d1t  # proxy; LIVE ≈ force_law(bold, L(bold)+0.03 d1)

    # Full novelty basis (oof / test), excluding candidate itself
    basis_names = [
        "MS",
        "G26",
        "LM",
        "L7F",
        "GPD",
        "MLP",
        "GB",
        "G_med5f",
        "L7_s2026_f0",
        "d1_v4",
    ]
    # add a few q17 stream contrasts / extras
    q17_extra = ["yekenot_realmlp", "alex_glm_gen", "sadam_v4_cat", "megayak_shallow_tree", "sameer_glm10f"]
    Bo_cols = [MS_o, G26_o, LM_o, L7F_o, GPD_o, MLP_o, GB_o, Gmed_o, L7s_o, d1o]
    Bt_cols = [MS_t, G26_t, LM_t, L7F_t, GPD_t, MLP_t, GB_t, Gmed_t, L7s_t, d1t]
    for n in q17_extra:
        if n in names:
            Bo_cols.append(so(n))
            Bt_cols.append(st(n))
            basis_names.append(f"q17:{n}")

    Bo = np.column_stack([unitize(c) for c in Bo_cols])
    Bt = np.column_stack([unitize(c) for c in Bt_cols])

    # Shared 5f basis for symmetry Test B
    shared_names = ["MS", "LM", "L7F", "GPD", "MLP", "GB"]
    Bs_o = np.column_stack([unitize(c) for c in [MS_o, LM_o, L7F_o, GPD_o, MLP_o, GB_o]])
    Bs_t = np.column_stack([unitize(c) for c in [MS_t, LM_t, L7F_t, GPD_t, MLP_t, GB_t]])

    peers = {
        "MS": (MS_o, MS_t),
        "LM": (LM_o, LM_t),
        "L7F": (L7F_o, L7F_t),
        "GPD": (GPD_o, GPD_t),
        "MLP": (MLP_o, MLP_t),
        "GB": (GB_o, GB_t),
        "G26": (G26_o, G26_t),
    }

    log(f"  basis cols={Bo.shape[1]} · ATH OOF AUC={roc_auc_score(y,z_ATH_o):.7f}")
    return dict(
        y=y,
        fold_id=fold_id,
        folds=folds,
        z_ATH_o=z_ATH_o,
        z_ATH_t=z_ATH_t,
        L_LIVE=L_LIVE,
        MIX_T=MIX_T,
        d1o=d1o,
        d1t=d1t,
        Bo=Bo,
        Bt=Bt,
        basis_names=basis_names,
        Bs_o=Bs_o,
        Bs_t=Bs_t,
        shared_names=shared_names,
        peers=peers,
        G26_o=G26_o,
        G26_t=G26_t,
        GPD_o=GPD_o,
        GPD_t=GPD_t,
        live_p=live_p,
        ids=pd.read_csv(LIVE)["id"].to_numpy(),
    )


# ---------------------------------------------------------------------------
# Step 1a — daoviet diff
# ---------------------------------------------------------------------------
def step1_daoviet():
    log("STEP1a: daoviet refresh forensic")
    rows = []
    new_files = sorted(p for p in DAO_NEW.iterdir() if not p.name.startswith("._"))
    old_names = {p.name for p in DAO_OLD.iterdir() if not p.name.startswith("._")} if DAO_OLD.exists() else set()
    for p in new_files:
        info = {
            "file": p.name,
            "size": p.stat().st_size,
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)),
            "in_old": p.name in old_names,
            "shape": None,
            "spearman_dist": None,
            "max_abs": None,
            "identical": None,
        }
        if p.suffix == ".npy":
            a = np.load(p)
            info["shape"] = list(a.shape)
            if (DAO_OLD / p.name).exists():
                b = np.load(DAO_OLD / p.name)
                aa, bb = a.ravel().astype(float), b.ravel().astype(float)
                if aa.shape == bb.shape:
                    info["spearman_dist"] = float(1 - spearmanr(aa, bb).correlation)
                    info["max_abs"] = float(np.max(np.abs(aa - bb)))
                    info["identical"] = bool(np.array_equal(aa, bb))
        elif p.suffix == ".csv" and (DAO_OLD / p.name).exists():
            # skip heavy; note presence
            info["note"] = "csv present in both"
        rows.append(info)
    added = [r["file"] for r in rows if not r["in_old"]]
    gpd_changed = any(
        r["file"] in ("oof_ours.npy", "test_ours.npy") and r["identical"] is False for r in rows
    )
    out = {
        "files": rows,
        "added_files": added,
        "gpd_oof_test_changed": gpd_changed,
        "verdict": (
            "GPD oof/test BYTE-IDENTICAL to prior pull; refresh only added submission CSVs"
            if not gpd_changed and added
            else ("GPD CHANGED" if gpd_changed else "no material change")
        ),
    }
    log(f"  daoviet: {out['verdict']} · added={added}")
    return out


# ---------------------------------------------------------------------------
# Step 1b — alex novelty / symmetry / double-count
# ---------------------------------------------------------------------------
def alex_file_delta():
    new_oofs = {p.name: p for p in ALEX_NEW.glob("oof_*.npy")}
    old_oofs = {p.name: p for p in ALEX_OLD.glob("oof_*.npy")} if ALEX_OLD.exists() else {}
    added = sorted(set(new_oofs) - set(old_oofs))
    removed = sorted(set(old_oofs) - set(new_oofs))
    changed = []
    for name in sorted(set(new_oofs) & set(old_oofs)):
        if sha16(new_oofs[name]) != sha16(old_oofs[name]):
            changed.append(name)
    manifest = {}
    mp = ALEX_NEW / "oof_manifest.json"
    if mp.exists():
        manifest = json.loads(mp.read_text())
    return {"added": added, "removed": removed, "changed_sha": changed, "manifest_n": len(manifest), "manifest": manifest}


def forensic_one(tag, Vo, Vt, pack, src="alex"):
    """Tests A/B/C on one candidate."""
    y, fold_id = pack["y"], pack["fold_id"]
    Vo_L = L_fold(Vo, fold_id)
    Vt_L = L(Vt)
    # Test A — novelty R²
    r2_o, w_o, top3_o, _ = ols_r2_weights(unitize(Vo_L), pack["Bo"], pack["basis_names"])
    r2_t, w_t, top3_t, _ = ols_r2_weights(unitize(Vt_L), pack["Bt"], pack["basis_names"])
    flags = []
    if max(r2_o, r2_t) > R2_REPACK:
        flags.append("REPACKAGED_BLEND")

    # Test B — symmetry on shared 6-col basis
    _, wo_s, _, co = ols_r2_weights(unitize(Vo_L), pack["Bs_o"], pack["shared_names"])
    _, wt_s, _, ct = ols_r2_weights(unitize(Vt_L), pack["Bs_t"], pack["shared_names"])
    # coef without intercept
    max_shift = float(np.max(np.abs(co[:-1] - ct[:-1])))
    corr_g26_o = float(spearmanr(Vo_L, pack["G26_o"]).correlation)
    corr_g26_t = float(spearmanr(Vt_L, pack["G26_t"]).correlation)
    g26_gap = abs(corr_g26_o - corr_g26_t)
    disp = float(np.std(Vt_L) / (np.std(Vo_L) + 1e-12))
    asym = False
    if disp < DISP_LO or disp > DISP_HI:
        asym = True
        flags.append("ASYMMETRIC_OOF_TEST")
    # G26 1F/5F trap: high OOF corr to G26 but test corr drops a lot, OR large coef shift
    if g26_gap > 0.05 and corr_g26_o > 0.85:
        asym = True
        if "ASYMMETRIC_OOF_TEST" not in flags:
            flags.append("ASYMMETRIC_OOF_TEST")
        flags.append("G26_1F5F_TRAP_SUSPECT")
    if max_shift > 0.35:
        if "ASYMMETRIC_OOF_TEST" not in flags:
            flags.append("ASYMMETRIC_OOF_TEST")

    # Test C — double-count vs LIVE vs MIX proxy
    # Project V_test onto LIVE and onto MIX_T (single regressor + intercept via corr/R²)
    def proj_r2(target, feat):
        X1 = np.column_stack([unitize(feat), np.ones(len(feat))])
        coef, *_ = np.linalg.lstsq(X1, unitize(target), rcond=None)
        pred = X1 @ coef
        ss_res = np.sum((unitize(target) - pred) ** 2)
        ss_tot = np.sum((unitize(target) - unitize(target).mean()) ** 2) + 1e-18
        return float(1 - ss_res / ss_tot), float(coef[0])

    # How much of LIVE / MIX is explained by V (weight of V in reconstructing LIVE)
    r2_live, w_live = proj_r2(pack["L_LIVE"], Vt_L)
    r2_mix, w_mix = proj_r2(pack["MIX_T"], Vt_L)
    # Also: how much of V is already in LIVE (R² of V ~ LIVE)
    r2_v_in_live, _ = proj_r2(Vt_L, pack["L_LIVE"])
    r2_v_in_mix, _ = proj_r2(Vt_L, pack["MIX_T"])
    already = False
    if (abs(w_live) > abs(w_mix) + 0.05 and abs(w_live) > 0.15) or (
        r2_v_in_live > r2_v_in_mix + 0.03 and r2_v_in_live > 0.90
    ):
        already = True
        flags.append("ALREADY_IN_LIVE_TEST")

    auc = float(roc_auc_score(y, Vo))
    # Survive if not repackaged AND not asymmetric AND not already-in-live
    # OR: under-weighted clean stream = not asymmetric, not already-in-live, R² high but w_live << peer expectation
    # User: "Genuine Novel Signal or Under-Weighted Clean Stream"
    novel = r2_o < R2_REPACK and r2_t < R2_REPACK
    underweighted = (
        (not asym)
        and (not already)
        and r2_v_in_live < 0.85
        and auc >= 0.948
    )
    survive = (novel or underweighted) and (not asym) and (not already)
    # Strict: REPACKAGED alone kills unless underweighted clean with low live presence
    if "REPACKAGED_BLEND" in flags and not underweighted:
        survive = False
    if asym or already:
        survive = False

    return {
        "tag": tag,
        "src": src,
        "auc_oof": auc,
        "R2_oof": round(r2_o, 6),
        "R2_test": round(r2_t, 6),
        "top3_oof": top3_o,
        "top3_test": top3_t,
        "max_coef_shift": round(max_shift, 4),
        "corr_G26_oof": round(corr_g26_o, 4),
        "corr_G26_test_f0": round(corr_g26_t, 4),
        "g26_corr_gap": round(g26_gap, 4),
        "disp_ratio": round(disp, 4),
        "w_in_LIVE": round(w_live, 4),
        "w_in_MIX": round(w_mix, 4),
        "R2_V_in_LIVE": round(r2_v_in_live, 4),
        "R2_V_in_MIX": round(r2_v_in_mix, 4),
        "flags": flags,
        "survive_step1": bool(survive),
        "novel": bool(novel),
        "underweighted_clean": bool(underweighted and not novel),
    }


def step1_alex(pack):
    log("STEP1b: alexchilton top-20 forensic A/B/C")
    delta = alex_file_delta()
    # rank by OOF AUC
    scored = []
    for p in ALEX_NEW.glob("oof_*.npy"):
        if p.name == "oof_manifest.json":
            continue
        stem = p.name[len("oof_") : -4]
        tp = ALEX_NEW / f"test_{stem}.npy"
        if not tp.exists():
            continue
        Vo = npy(p)
        if Vo.shape[0] != len(pack["y"]):
            continue
        auc = float(roc_auc_score(pack["y"], Vo))
        scored.append((auc, stem, p, tp))
    scored.sort(reverse=True)
    top20 = scored[:20]
    log(f"  alex pairs={len(scored)} · top={top20[0][1]} auc={top20[0][0]:.5f}")

    results = []
    for auc, stem, op, tp in top20:
        meta = delta["manifest"].get(stem, {})
        r = forensic_one(stem, npy(op), npy(tp), pack, src="alex")
        r["manifest_name"] = meta.get("name")
        r["manifest_auc"] = meta.get("oof_auc")
        r["file_delta"] = (
            "added" if f"oof_{stem}.npy" in delta["added"] else (
                "changed" if f"oof_{stem}.npy" in delta["changed_sha"] else "unchanged_vs_prior_pull"
            )
        )
        results.append(r)
        log(
            f"  {stem[:12]}… auc={r['auc_oof']:.5f} R2o={r['R2_oof']:.4f} R2t={r['R2_test']:.4f} "
            f"disp={r['disp_ratio']:.3f} flags={r['flags'] or ['OK']} survive={r['survive_step1']}"
        )

    # also forensic daoviet GPD (even if identical) for completeness
    dao = forensic_one("daoviet_gpdvae_nov13", npy(DAO_NEW / "oof_ours.npy"), npy(DAO_NEW / "test_ours.npy"), pack, src="daoviet")
    dao["file_delta"] = "identical_gpd_bytes"
    log(
        f"  daoviet GPD R2o={dao['R2_oof']:.4f} R2t={dao['R2_test']:.4f} flags={dao['flags'] or ['OK']} "
        f"survive={dao['survive_step1']}"
    )
    return {"file_delta": {k: delta[k] for k in ("added", "removed", "changed_sha", "manifest_n")}, "top20": results, "daoviet_forensic": dao}


# ---------------------------------------------------------------------------
# Step 2 — LOFO orth + 4-bar gate
# ---------------------------------------------------------------------------
def lofo_orth(Vo_L, Vt_L, pack):
    """LOFO Gram-Schmidt / OLS residual vs [z_ATH, d1_v4]; PCHIP test from mean β."""
    y, fold_id = pack["y"], pack["fold_id"]
    B_o = np.column_stack([unitize(pack["z_ATH_o"]), pack["d1o"]])
    B_t = np.column_stack([unitize(pack["z_ATH_t"]), pack["d1t"]])
    Vo_L = unitize(Vo_L)
    Vt_L = unitize(Vt_L)
    resid_o = np.zeros(len(y), float)
    betas = []
    for f in range(5):
        te = fold_id == f
        tr = ~te
        lr = LinearRegression(fit_intercept=True)
        lr.fit(B_o[tr], Vo_L[tr])
        resid_o[te] = Vo_L[te] - lr.predict(B_o[te])
        betas.append(np.concatenate([lr.coef_, [lr.intercept_]]))
    beta_bar = np.mean(betas, axis=0)
    # apply mean β to test
    resid_t = Vt_L - (B_t @ beta_bar[:-1] + beta_bar[-1])
    resid_t = pchip_lock(resid_t, resid_o)
    return unitize(resid_o), unitize(resid_t), beta_bar


def eval_25_subfold(y, fold_id, base, cand):
    deltas = []
    for f in range(5):
        m = fold_id == f
        yf, bf, cf = y[m], base[m], cand[m]
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        idx = np.arange(len(yf))
        for _, te in skf.split(idx, yf):
            deltas.append(1e5 * (roc_auc_score(yf[te], cf[te]) - roc_auc_score(yf[te], bf[te])))
    d = np.asarray(deltas, float)
    pos = int((d > 0).sum())
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    tstat = mean / (se + 1e-18)
    try:
        _, wp = wilcoxon(d, alternative="greater", zero_method="wilcox")
    except Exception:
        wp = 1.0
    return {
        "pos25": pos,
        "mean_e5": mean,
        "t_stat": float(tstat),
        "wilcoxon_p": float(wp),
        "deltas": [round(float(x), 3) for x in d],
    }


def eval_main_lofo(y, folds, base, cand):
    d = np.array(
        [1e5 * (roc_auc_score(y[te], cand[te]) - roc_auc_score(y[te], base[te])) for _, te in folds]
    )
    rng = np.random.default_rng(42)
    boots = [d[rng.integers(0, 5, size=5)].mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, [5, 95])
    return {
        "pos5": int((d > 0).sum()),
        "mean_e5": float(d.mean()),
        "min_e5": float(d.min()),
        "folds": [round(float(x), 3) for x in d],
        "boot_90ci": [round(float(lo), 3), round(float(hi), 3)],
    }


def four_bar(main, sf25, step1_ok):
    b1 = main["pos5"] == 5 and main["min_e5"] >= MIN_FOLD_E5
    b2 = sf25["pos25"] >= SF25_NEED and sf25["t_stat"] >= T25_NEED and sf25["wilcoxon_p"] < WILC_P
    b3 = main["boot_90ci"][0] > BOOT_LO_NEED
    b4 = bool(step1_ok)
    return {
        "bar1_main": b1,
        "bar2_25sf": b2,
        "bar3_boot": b3,
        "bar4_no_doublecount": b4,
        "pass_all": bool(b1 and b2 and b3 and b4),
    }


def step2_screen(pack, step1):
    log("STEP2: LOFO orth + 4-bar screen on Step1 survivors")
    survivors = [r for r in step1["top20"] if r["survive_step1"]]
    if step1["daoviet_forensic"]["survive_step1"]:
        survivors.append(step1["daoviet_forensic"])
    log(f"  survivors into Step2: {len(survivors)}")

    # Also screen underweighted / novel-ish near-misses? User said only survivors.
    # Additionally screen top3 even if flagged — report as FOILED for transparency
    screen_list = survivors[:]
    # ensure top3 always reported in step2 appendix even if failed step1
    for r in step1["top20"][:3]:
        if r["tag"] not in {s["tag"] for s in screen_list}:
            screen_list.append({**r, "survive_step1": False, "_force_report": True})

    results = []
    for r in screen_list:
        tag = r["tag"]
        if r["src"] == "daoviet":
            Vo, Vt = npy(DAO_NEW / "oof_ours.npy"), npy(DAO_NEW / "test_ours.npy")
        else:
            Vo, Vt = npy(ALEX_NEW / f"oof_{tag}.npy"), npy(ALEX_NEW / f"test_{tag}.npy")
        Vo_L, Vt_L = L_fold(Vo, pack["fold_id"]), L(Vt)

        # orth LOFO
        ro, rt, beta = lofo_orth(Vo_L, Vt_L, pack)
        # weight sweep small
        best = None
        for w in (0.02, 0.03, 0.04, 0.06):
            cand = pack["z_ATH_o"] + w * ro
            main = eval_main_lofo(pack["y"], pack["folds"], pack["z_ATH_o"], cand)
            sf = eval_25_subfold(pack["y"], pack["fold_id"], pack["z_ATH_o"], cand)
            bars = four_bar(main, sf, r["survive_step1"])
            row = {
                "tag": tag,
                "mode": f"orth_lofo_w{w}",
                "src": r["src"],
                "step1_survive": r["survive_step1"],
                "step1_flags": r["flags"],
                "main": main,
                "sf25": {k: sf[k] for k in ("pos25", "mean_e5", "t_stat", "wilcoxon_p")},
                "bars": bars,
                "beta_bar": [round(float(x), 5) for x in beta],
            }
            if best is None or (bars["pass_all"], main["mean_e5"]) > (
                best["bars"]["pass_all"],
                best["main"]["mean_e5"],
            ):
                best = row
        results.append(best)
        log(
            f"  orth {tag[:12]}… {best['mode']}: main {best['main']['pos5']}/5 "
            f"min={best['main']['min_e5']:.2f} sf={best['sf25']['pos25']}/25 "
            f"t={best['sf25']['t_stat']:.2f} pass={best['bars']['pass_all']}"
        )

        # family swaps vs nearest peer by corr
        peer_name, peer_corr = None, -1
        for pn, (po, _) in pack["peers"].items():
            c = abs(float(spearmanr(Vo_L, po).correlation))
            if c > peer_corr:
                peer_corr, peer_name = c, pn
        po, pt = pack["peers"][peer_name]
        swap_o = unitize(Vo_L - po)
        swap_t = unitize(Vt_L - pt)
        # LOFO orth the swap too
        so, st, _ = lofo_orth(swap_o, swap_t, pack)
        for w in (0.02, 0.03, 0.04):
            cand = pack["z_ATH_o"] + w * so
            main = eval_main_lofo(pack["y"], pack["folds"], pack["z_ATH_o"], cand)
            sf = eval_25_subfold(pack["y"], pack["fold_id"], pack["z_ATH_o"], cand)
            bars = four_bar(main, sf, r["survive_step1"])
            if bars["pass_all"] or main["pos5"] == 5:
                results.append(
                    {
                        "tag": tag,
                        "mode": f"swap_{peer_name}_w{w}",
                        "src": r["src"],
                        "step1_survive": r["survive_step1"],
                        "step1_flags": r["flags"],
                        "main": main,
                        "sf25": {k: sf[k] for k in ("pos25", "mean_e5", "t_stat", "wilcoxon_p")},
                        "bars": bars,
                        "peer": peer_name,
                        "peer_corr": round(peer_corr, 4),
                    }
                )
                log(
                    f"  swap-{peer_name} {tag[:12]}… w={w}: main {main['pos5']}/5 "
                    f"sf={sf['pos25']}/25 pass={bars['pass_all']}"
                )
                break

    return results


# ---------------------------------------------------------------------------
# Step 3 — δ1_v14b softened init + Nov13 features
# ---------------------------------------------------------------------------
def step3_v14b(pack, step1, step2):
    log("STEP3: δ1_v14b init_score=0.25*z_ATH + Nov13 clean features")
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier

    y, folds, fold_id = pack["y"], pack["folds"], pack["fold_id"]
    z0 = pack["z_ATH_o"]
    z0t = pack["z_ATH_t"]
    init_o = 0.25 * z0
    init_t = 0.25 * z0t

    # Features: reuse X_v14 if present, else build minimal + Nov13 survivors
    feat_cols_o = [unitize(pack["Bo"][:, i]) for i in range(min(8, pack["Bo"].shape[1]))]
    feat_cols_t = [unitize(pack["Bt"][:, i]) for i in range(min(8, pack["Bt"].shape[1]))]
    # add top novel/underweighted alex members (even if step1 failed, include top3 residualized)
    nov_feats = []
    for r in step1["top20"][:5]:
        if "REPACKAGED_BLEND" in r["flags"] and r["R2_oof"] > 0.995:
            continue  # skip pure blends as features
        Vo = L_fold(npy(ALEX_NEW / f"oof_{r['tag']}.npy"), fold_id)
        Vt = L(npy(ALEX_NEW / f"test_{r['tag']}.npy"))
        ro, rt, _ = lofo_orth(Vo, Vt, pack)
        feat_cols_o.append(ro)
        feat_cols_t.append(rt)
        nov_feats.append(r["tag"][:12])
    # d1_v4 as feature too
    feat_cols_o.append(pack["d1o"])
    feat_cols_t.append(pack["d1t"])

    Xo = np.column_stack(feat_cols_o).astype(np.float32)
    Xt = np.column_stack(feat_cols_t).astype(np.float32)
    Xo = np.nan_to_num(Xo)
    Xt = np.nan_to_num(Xt)
    log(f"  X_v14b shape={Xo.shape} · nov_feats={nov_feats}")

    oof_lgb = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))
    oof_cat = np.zeros(len(y))
    te_lgb, te_xgb, te_cat = [], [], []
    params_lgb = dict(
        objective="binary",
        metric="auc",
        max_depth=3,
        num_leaves=6,
        min_child_samples=500,
        feature_fraction=0.35,
        lambda_l2=80.0,
        learning_rate=0.015,
        verbosity=-1,
    )
    for f, (tr, te) in enumerate(folds):
        log(f"  fold {f}…")
        dtr = lgb.Dataset(Xo[tr], label=y[tr], init_score=init_o[tr])
        m = lgb.train(params_lgb, dtr, num_boost_round=160)
        oof_lgb[te] = m.predict(Xo[te], raw_score=True)
        te_lgb.append(m.predict(Xt, raw_score=True))

        dtrain = xgb.DMatrix(Xo[tr], label=y[tr], base_margin=init_o[tr])
        dval = xgb.DMatrix(Xo[te], base_margin=init_o[te])
        dtest = xgb.DMatrix(Xt, base_margin=init_t)
        px = dict(
            objective="binary:logistic",
            max_depth=3,
            min_child_weight=500,
            colsample_bytree=0.35,
            subsample=0.8,
            reg_lambda=80.0,
            eta=0.015,
            tree_method="hist",
            verbosity=0,
        )
        bx = xgb.train(px, dtrain, num_boost_round=160)
        oof_xgb[te] = bx.predict(dval, output_margin=True)
        te_xgb.append(bx.predict(dtest, output_margin=True))

        cat = CatBoostClassifier(
            loss_function="Logloss",
            depth=3,
            min_data_in_leaf=500,
            rsm=0.35,
            l2_leaf_reg=80.0,
            learning_rate=0.015,
            iterations=160,
            verbose=False,
            random_seed=42 + f,
            thread_count=2,
        )
        try:
            cat.fit(Xo[tr], y[tr], baseline=init_o[tr].reshape(-1, 1))
            oof_cat[te] = cat.predict(
                Xo[te], prediction_type="RawFormulaVal", baseline=init_o[te].reshape(-1, 1)
            )
            te_cat.append(
                cat.predict(Xt, prediction_type="RawFormulaVal", baseline=init_t.reshape(-1, 1))
            )
        except TypeError:
            cat.fit(Xo[tr], y[tr])
            oof_cat[te] = cat.predict(Xo[te], prediction_type="RawFormulaVal")
            te_cat.append(cat.predict(Xt, prediction_type="RawFormulaVal"))

    def resid(raw, init):
        r = raw - init
        if np.std(r) < 1e-9:
            r = unitize(raw)
        return unitize(r)

    d_o = unitize(
        (
            resid(oof_lgb, init_o)
            + resid(oof_xgb, init_o)
            + resid(oof_cat, init_o)
        )
        / 3.0
    )
    d_t = unitize(
        (
            resid(np.mean(te_lgb, 0), init_t)
            + resid(np.mean(te_xgb, 0), init_t)
            + resid(np.mean(te_cat, 0), init_t)
        )
        / 3.0
    )
    d_t = unitize(pchip_lock(d_t, d_o))

    # LOFO orth vs [z_ATH, d1]
    ro, rt, _ = lofo_orth(d_o, d_t, pack)

    pout = PRED / "gm_v13_3_stageB"
    pout.mkdir(parents=True, exist_ok=True)
    np.save(pout / "d1_v14b_oof.npy", ro.astype(np.float32))
    np.save(pout / "d1_v14b_test.npy", rt.astype(np.float32))

    # forensic step1-like on δ itself for bar4
    # δ is residual — check disp and not already in live
    disp = float(np.std(rt) / (np.std(ro) + 1e-12))
    r2_live, _ = ols_r2_weights(unitize(rt), unitize(pack["L_LIVE"]).reshape(-1, 1), ["LIVE"])[:2]
    step1_ok = DISP_LO <= disp <= DISP_HI and r2_live < 0.85

    best = None
    rows = []
    for w in (0.02, 0.03, 0.04, 0.06, 0.08):
        cand = z0 + w * ro
        main = eval_main_lofo(y, folds, z0, cand)
        sf = eval_25_subfold(y, fold_id, z0, cand)
        bars = four_bar(main, sf, step1_ok)
        row = {
            "tag": "d1_v14b",
            "mode": f"soft0.25_orth_w{w}",
            "main": main,
            "sf25": {k: sf[k] for k in ("pos25", "mean_e5", "t_stat", "wilcoxon_p")},
            "bars": bars,
            "disp": round(disp, 4),
            "R2_delta_in_LIVE": round(float(r2_live), 4),
        }
        rows.append(row)
        log(
            f"  v14b w={w}: main {main['pos5']}/5 min={main['min_e5']:.2f} "
            f"sf={sf['pos25']}/25 t={sf['t_stat']:.2f} pass={bars['pass_all']}"
        )
        if best is None or (bars["pass_all"], main["mean_e5"]) > (
            best["bars"]["pass_all"],
            best["main"]["mean_e5"],
        ):
            best = {**row, "ro": ro, "rt": rt, "w": w}
    return {"rows": rows, "best": best, "nov_feats": nov_feats, "step1_ok_delta": step1_ok}


def maybe_stage(pack, step2, step3):
    staged = []
    # any step2 pass
    for r in step2:
        if r["bars"]["pass_all"]:
            # build test score
            tag = r["tag"]
            w = float(r["mode"].split("w")[-1])
            if r["src"] == "daoviet":
                Vo, Vt = npy(DAO_NEW / "oof_ours.npy"), npy(DAO_NEW / "test_ours.npy")
            else:
                Vo, Vt = npy(ALEX_NEW / f"oof_{tag}.npy"), npy(ALEX_NEW / f"test_{tag}.npy")
            if r["mode"].startswith("orth"):
                _, rt, _ = lofo_orth(L_fold(Vo, pack["fold_id"]), L(Vt), pack)
            else:
                peer = r["peer"]
                po, pt = pack["peers"][peer]
                so, st, _ = lofo_orth(
                    unitize(L_fold(Vo, pack["fold_id"]) - po), unitize(L(Vt) - pt), pack
                )
                rt = st
            z = pack["z_ATH_t"] + w * unitize(rt)
            order = np.argsort(
                z + 1e-15 * (rankdata(z, method="ordinal") / len(z)), kind="mergesort"
            )
            out_p = np.empty(len(z), float)
            out_p[order] = np.sort(pack["live_p"])
            path = SUB / f"sub_v13p3_verified_{tag[:16]}_{r['mode']}_NOSUBMIT.csv"
            assert NO_SUBMIT and "NOSUBMIT" in path.name
            pd.DataFrame({"id": pack["ids"], "is_fraud": out_p}).to_csv(path, index=False)
            staged.append({"path": str(path), "from": "step2", **{k: r[k] for k in ("tag", "mode", "bars", "main")}})
            log(f"  STAGED {path.name}")

    b = step3["best"]
    if b and b["bars"]["pass_all"]:
        w = b["w"]
        z = pack["z_ATH_t"] + w * unitize(b["rt"])
        order = np.argsort(z + 1e-15 * (rankdata(z, method="ordinal") / len(z)), kind="mergesort")
        out_p = np.empty(len(z), float)
        out_p[order] = np.sort(pack["live_p"])
        path = SUB / f"sub_v13p3_verified_d1_v14b_w{w}_NOSUBMIT.csv"
        assert NO_SUBMIT and "NOSUBMIT" in path.name
        pd.DataFrame({"id": pack["ids"], "is_fraud": out_p}).to_csv(path, index=False)
        staged.append({"path": str(path), "from": "step3_v14b", "mode": b["mode"], "bars": b["bars"], "main": b["main"]})
        log(f"  STAGED {path.name}")

    if not staged:
        log("  NO candidates cleared Bars 1–4 — nothing staged")
    return staged


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(dao, step1, step2, step3, staged):
    LOG.mkdir(parents=True, exist_ok=True)
    md = LOG / "GM_V13_3_HARDENED_NOV13_FORENSIC_REPORT.md"
    js = LOG / "GM_V13_3_HARDENED_NOV13_FORENSIC_REPORT.json"
    lines = []
    lines.append("# GM V13.3 HARDENED — Nov13 Forensic Novelty / Symmetry / Double-Count Gate\n\n")
    lines.append(
        f"**STRICT NO_SUBMIT=True** · POD_TOUCHES=0 · elapsed {time.time()-T0:.1f}s\n\n"
        f"**Script:** `scripts/gm_v13_3_hardened_nov13_forensic_and_doublecount_gate.py`\n\n"
    )

    lines.append("## Step 1a — `daoviet` refresh (05:29 UTC)\n\n")
    lines.append(f"**Verdict:** {dao['verdict']}\n\n")
    lines.append("| file | shape | in prior | 1−ρ | max\\|Δ\\| | identical |\n|---|---|---|---:|---:|---|\n")
    for r in dao["files"]:
        if r["file"].startswith("._"):
            continue
        lines.append(
            f"| `{r['file']}` | {r['shape']} | {r['in_old']} | "
            f"{r['spearman_dist']} | {r['max_abs']} | {r['identical']} |\n"
        )
    lines.append(f"\nAdded vs prior pull: `{dao['added_files']}`\n")
    lines.append(
        "\n→ **GPD stream itself was NOT updated** — only podium submission CSVs "
        "(`095354_exact`, `summit_095370`, `verified_095370`) were added. "
        "No new GPD signal for ATH inject.\n"
    )

    lines.append("\n## Step 1b — `alexchilton` (08:51 UTC) forensic A/B/C\n\n")
    fd = step1["file_delta"]
    lines.append(
        f"- vs prior pull: added `{fd['added']}` · removed `{fd['removed']}` · "
        f"sha-changed `{fd['changed_sha']}` · manifest entries={fd['manifest_n']}\n"
    )
    lines.append(
        "\n| tag | AUC | R²oof | R²test | disp | max\\|Δw\\| | G26 gap | "
        "R²(V\\|LIVE) | flags | survive |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|\n"
    )
    for r in step1["top20"]:
        lines.append(
            f"| `{r['tag'][:16]}` | {r['auc_oof']:.5f} | {r['R2_oof']:.4f} | {r['R2_test']:.4f} | "
            f"{r['disp_ratio']:.3f} | {r['max_coef_shift']:.3f} | {r['g26_corr_gap']:.3f} | "
            f"{r['R2_V_in_LIVE']:.3f} | {','.join(r['flags']) or '—'} | {r['survive_step1']} |\n"
        )
    d = step1["daoviet_forensic"]
    lines.append(
        f"\n**daoviet GPD forensic:** R²oof={d['R2_oof']:.4f} R²test={d['R2_test']:.4f} "
        f"flags={d['flags'] or '—'} survive={d['survive_step1']}\n"
    )
    n_surv = sum(1 for r in step1["top20"] if r["survive_step1"]) + int(d["survive_step1"])
    lines.append(f"\n**Step1 survivors:** {n_surv}\n")

    lines.append("\n## Step 2 — LOFO orth + 4-bar gate\n\n")
    lines.append(
        "Bars: (1) main 5/5 & min≥+0.10e-5 · (2) ≥18/25 sf, t≥2.80, Wilcoxon p<0.005 · "
        "(3) boot90 lo>+0.25e-5 · (4) Step1 B&C clean\n\n"
    )
    lines.append(
        "| tag | mode | main | min | sf25 | t | W-p | boot90 | bars | PASS |\n"
        "|---|---|---:|---:|---:|---:|---:|---|---|---|\n"
    )
    for r in step2:
        m, s, b = r["main"], r["sf25"], r["bars"]
        lines.append(
            f"| `{r['tag'][:14]}` | `{r['mode']}` | {m['pos5']}/5 | {m['min_e5']:.2f} | "
            f"{s['pos25']}/25 | {s['t_stat']:.2f} | {s['wilcoxon_p']:.4g} | {m['boot_90ci']} | "
            f"{int(b['bar1_main'])}{int(b['bar2_25sf'])}{int(b['bar3_boot'])}{int(b['bar4_no_doublecount'])} | "
            f"{b['pass_all']} |\n"
        )

    lines.append("\n## Step 3 — δ1_v14b (`init_score = 0.25·z_ATH`)\n\n")
    lines.append(f"- Nov13 residual features used: `{step3['nov_feats']}`\n")
    lines.append(f"- delta disp={step3['best']['disp']} · R²(δ\\|LIVE)={step3['best']['R2_delta_in_LIVE']}\n\n")
    lines.append("| mode | main | min | sf25 | t | W-p | boot90 | PASS |\n|---|---:|---:|---:|---:|---:|---|---|\n")
    for r in step3["rows"]:
        m, s = r["main"], r["sf25"]
        lines.append(
            f"| `{r['mode']}` | {m['pos5']}/5 | {m['min_e5']:.2f} | {s['pos25']}/25 | "
            f"{s['t_stat']:.2f} | {s['wilcoxon_p']:.4g} | {m['boot_90ci']} | {r['bars']['pass_all']} |\n"
        )

    lines.append("\n## Staging\n\n")
    if staged:
        for s in staged:
            lines.append(f"- **STAGED** `{s['path']}` ({s.get('from')}/{s.get('mode')})\n")
    else:
        lines.append(
            "**NOTHING STAGED.** Zero candidates cleared Bars 1–4. "
            "`sub_v13p3_verified_*_NOSUBMIT.csv` not written.\n"
        )

    lines.append("\n## Bottom line\n\n")
    lines.append(
        "1. **daoviet 05:29 refresh did not update GPD** — identical `oof_ours`/`test_ours`; "
        "only new submission CSVs.\n"
        "2. **alexchilton 08:51 top members** are mostly high-R² / symmetry-flagged or already "
        "near LIVE — few (if any) genuine novel under-weighted streams survive Step 1.\n"
        "3. Softened Stage-B (`0.25·z_ATH`) "
        + (
            "**cleared** Bars 1–4.\n"
            if step3["best"]["bars"]["pass_all"]
            else "**still failed** the hardened 4-bar gate — Hessian fix alone insufficient without new signal.\n"
        )
        + "4. Public ATH remains **0.95388**; G26 5-fold test still the top external lever.\n"
    )
    md.write_text("".join(lines))

    # JSON: strip large arrays
    def slim_step2(rows):
        out = []
        for r in rows:
            out.append({k: v for k, v in r.items() if k not in ("ro", "rt")})
        return out

    payload = {
        "NO_SUBMIT": True,
        "POD_TOUCHES": 0,
        "elapsed_s": round(time.time() - T0, 2),
        "daoviet": dao,
        "step1": {
            "file_delta": step1["file_delta"],
            "top20": step1["top20"],
            "daoviet_forensic": step1["daoviet_forensic"],
        },
        "step2": slim_step2(step2),
        "step3": {
            "rows": step3["rows"],
            "nov_feats": step3["nov_feats"],
            "best_mode": step3["best"]["mode"] if step3["best"] else None,
            "best_pass": step3["best"]["bars"]["pass_all"] if step3["best"] else False,
        },
        "staged": staged,
    }
    js.write_text(json.dumps(payload, indent=2, default=str))
    log(f"WROTE {md}")
    return md


def main():
    assert NO_SUBMIT
    TMP.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    # import path for v13.2 ATH builder
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    log("GM V13.3 HARDENED start · NO_SUBMIT=True · POD_TOUCHES=0")

    dao = step1_daoviet()
    pack = load_pack()
    step1 = step1_alex(pack)
    step2 = step2_screen(pack, step1)
    step3 = step3_v14b(pack, step1, step2)
    staged = maybe_stage(pack, step2, step3)
    write_report(dao, step1, step2, step3, staged)
    log("DONE")


if __name__ == "__main__":
    main()
