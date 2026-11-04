"""Part9B (Fixed): Build P_super_champ (0.952142 Nested CV) and clean full-signal fusion vs p7gs (0.95231).
Bugs fixed from initial Part9B:
1. Never rank-average mean-zero OLS residuals (AUC ~0.50) outside the logistic stacker.
   Orthogonalized residuals stay strictly inside newton-cholesky (C=0.005).
2. Post-stack blend uses only full-signal calibrated models:
   - newton-cholesky stacker on Part8B 74-member bank (C=0.005, OOF 0.952130)
   - Caruana hill-climb on Part8B 74-member bank (12% mix -> OOF 0.952140)
   - gb_super = 0.50 * rk(gb_bag5) + 0.50 * rk(gb_gpt2_joint) (standalone OOF 0.951824, 8% mix -> OOF 0.9521425)
3. Clean multi-anchor fusion with p7, gs (v11), and pb (Nilanjan probit stack), verified all full-signal.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

from part6_repro_and_expand import ART, folds_y, rlogit  # noqa: E402
from part9_joint_boost_and_basis import build_8b_champ  # noqa: E402

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:5.1f}s]", *a, flush=True)


def rk(v):
    v = np.asarray(v, float).ravel()
    return rankdata(v) / len(v)


def rho(a, b):
    return float(spearmanr(a, b).correlation)


def fast_auc(s, yy, bins=1 << 20):
    lo, hi = float(s.min()), float(s.max())
    b = ((s - lo) * ((bins - 1) / (hi - lo + 1e-12))).astype(np.int64)
    pos = np.bincount(b, weights=yy, minlength=bins)
    neg = np.bincount(b, minlength=bins) - pos
    return float((pos * (np.cumsum(neg) - neg)).sum() + 0.5 * (pos * neg).sum()) / (
        pos.sum() * neg.sum() + 1e-30
    )


def hill_climb(O, yy, iters=80):
    w = np.zeros(O.shape[1])
    w[int(np.argmax([fast_auc(O[:, i], yy) for i in range(O.shape[1])]))] = 1.0
    s = O @ w
    cur = fast_auc(s, yy)
    for _ in range(iters):
        sc, i = max((fast_auc(s + O[:, i], yy), i) for i in range(O.shape[1]))
        if sc <= cur + 1e-7:
            break
        w[i] += 1
        s = s + O[:, i]
        cur = sc
    return w / w.sum()


def build_p_super_champ(y, folds):
    names8, O8, T8 = build_8b_champ(y, folds)
    gb5_o = np.load(ART / "gb_bag5_oof.npy").ravel().astype(float)
    gb5_t = np.load(ART / "gb_bag5_full.npy").ravel().astype(float)
    gj_o = np.load("preds/gb_gpt2_joint_oof.npy").ravel().astype(float)
    gj_t = np.load("preds/gb_gpt2_joint_test.npy").ravel().astype(float)

    sup_o = 0.50 * rk(gb5_o) + 0.50 * rk(gj_o)
    sup_t = 0.50 * rk(gb5_t) + 0.50 * rk(gj_t)
    sup_auc = float(roc_auc_score(y, sup_o))
    log(f"gb_super (0.50 gb_bag5 + 0.50 gb_gpt2_joint) standalone OOF={sup_auc:.6f}")

    C_opt = 0.005
    hw_opt = 0.12
    sw_opt = 0.08

    st_oof = np.zeros(len(y))
    hc_oof = np.zeros(len(y))
    for a, b in folds:
        m = LogisticRegression(C=C_opt, tol=1e-8, max_iter=200, solver="newton-cholesky")
        m.fit(rlogit(O8[a]), y[a])
        st_oof[b] = rk(m.decision_function(rlogit(O8[b])))
        w = hill_climb(O8[a], y[a])
        hc_oof[b] = rk(O8[b] @ w)

    auc_newton = float(roc_auc_score(y, st_oof))
    mix_oof = rk((1.0 - hw_opt) * st_oof + hw_opt * hc_oof)
    auc_mix = float(roc_auc_score(y, mix_oof))
    P_super_oof = rk((1.0 - sw_opt) * mix_oof + sw_opt * rk(sup_o))
    auc_super = float(roc_auc_score(y, P_super_oof))
    fold_aucs = [float(roc_auc_score(y[b], P_super_oof[b])) for _, b in folds]

    log(f"Nested CV progression: pure_newton(C={C_opt})={auc_newton:.6f} -> +hill({hw_opt})={auc_mix:.6f} -> +gb_super({sw_opt})={auc_super:.6f}")

    m_full = LogisticRegression(C=C_opt, tol=1e-8, max_iter=200, solver="newton-cholesky")
    m_full.fit(rlogit(O8), y)
    st_te = rk(m_full.decision_function(rlogit(T8)))
    w_full = hill_climb(O8, y)
    hc_te = rk(T8 @ w_full)
    mix_te = rk((1.0 - hw_opt) * st_te + hw_opt * hc_te)
    P_super_te = rk((1.0 - sw_opt) * mix_te + sw_opt * rk(sup_t))

    Path("preds").mkdir(exist_ok=True)
    np.save("preds/P_super_champ_oof.npy", P_super_oof.astype(np.float32))
    np.save("preds/P_super_champ_test.npy", P_super_te.astype(np.float32))

    return P_super_oof, P_super_te, {
        "gb_super_solo_oof": sup_auc,
        "newton_c005_oof": auc_newton,
        "newton_hill_mix_oof": auc_mix,
        "P_super_champ_oof": auc_super,
        "folds": fold_aucs,
        "params": {"C": C_opt, "hill_w": hw_opt, "gb_super_w": sw_opt},
    }


def main():
    log("Part9B Fixed — building clean full-signal fusion")
    y = pd.read_csv("data/train.csv")["is_fraud"].to_numpy()
    folds = folds_y(y)
    sub_ids = pd.read_csv("data/sample_submission.csv")["id"].to_numpy()

    if Path("preds/P_super_champ_test.npy").exists() and Path("preds/P_super_champ_oof.npy").exists():
        P_oof = np.load("preds/P_super_champ_oof.npy").ravel().astype(float)
        P_te = np.load("preds/P_super_champ_test.npy").ravel().astype(float)
        cv_info = {
            "P_super_champ_oof": float(roc_auc_score(y, P_oof)),
            "folds": [float(roc_auc_score(y[b], P_oof[b])) for _, b in folds],
        }
        log(f"Loaded cached P_super_champ OOF={cv_info['P_super_champ_oof']:.6f}")
    else:
        P_oof, P_te, cv_info = build_p_super_champ(y, folds)

    p7 = pd.read_csv("external_bank/goodpjw_part7_out/submission.csv").set_index("id").loc[sub_ids, "is_fraud"].to_numpy(float)
    part6 = pd.read_csv("external_bank/malyshev_v11/nov04_part6.csv").set_index("id").loc[sub_ids, "is_fraud"].to_numpy(float)
    alex = pd.read_csv("external_bank/malyshev_v11/nov04_alex.csv").set_index("id").loc[sub_ids, "is_fraud"].to_numpy(float)
    gs = rk(0.6 * rk(part6) + 0.4 * rk(alex))
    p7gs = rk(0.5 * rk(p7) + 0.5 * rk(gs))
    pb = pd.read_csv("external_bank/nil_probit_0952/submission.csv").set_index("id").loc[sub_ids, "is_fraud"].to_numpy(float)
    slot5 = pd.read_csv("submissions/sub_slot5_p7off_v11_fusion.csv").set_index("id").loc[sub_ids, "is_fraud"].to_numpy(float)
    slot6 = pd.read_csv("submissions/sub_slot6_alt_065_020_015_fusion.csv").set_index("id").loc[sub_ids, "is_fraud"].to_numpy(float)

    # Clean full-signal fusion:
    # 0.35 * rk(P_super_champ) + 0.15 * rk(p7) + 0.44 * rk(gs) + 0.06 * rk(pb)
    # Algebraically: 50% Part7/8B/9B family (35% P_super_champ + 15% p7) + 44% gs + 6% pb
    sub_clean = rk(0.35 * rk(P_te) + 0.15 * rk(p7) + 0.44 * rk(gs) + 0.06 * rk(pb))

    # Sanity guard: verify every component is positively correlated (> 0.994) with p7gs!
    for cname, cvec in [("P_super_champ", P_te), ("p7", p7), ("gs", gs), ("pb", pb)]:
        r = rho(cvec, p7gs)
        assert r > 0.994, f"ABORT: component {cname} has low correlation {r:.6f} with p7gs!"

    out_csv = Path("submissions/sub_part9c_clean_super_p7gs_pb.csv")
    pd.DataFrame({"id": sub_ids, "is_fraud": sub_clean}).to_csv(out_csv, index=False)

    diag = {
        "rho_vs_p7gs": rho(sub_clean, p7gs),
        "d_p7gs_e4": (1.0 - rho(sub_clean, p7gs)) * 1e4,
        "rho_vs_slot5": rho(sub_clean, slot5),
        "d_slot5_e4": (1.0 - rho(sub_clean, slot5)) * 1e4,
        "rho_vs_slot6": rho(sub_clean, slot6),
        "d_slot6_e4": (1.0 - rho(sub_clean, slot6)) * 1e4,
    }
    log(
        f"wrote {out_csv}: 1-ρ(p7gs)={diag['d_p7gs_e4']:.2f}e-4 | "
        f"1-ρ(slot5)={diag['d_slot5_e4']:.2f}e-4 | 1-ρ(slot6)={diag['d_slot6_e4']:.2f}e-4"
    )

    summary_path = Path("logs/overnight/part9b_beat_p7gs.json")
    meta = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    meta["part9c_fixed"] = {
        "file": str(out_csv),
        "weights": {"P_super_champ": 0.35, "p7": 0.15, "gs": 0.44, "pb": 0.06},
        "cv_info": cv_info,
        "diagnostics": diag,
    }
    summary_path.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
