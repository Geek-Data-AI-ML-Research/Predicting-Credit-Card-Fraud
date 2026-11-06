#!/usr/bin/env python3
"""Deterministic builder and verifier for the frozen P_super_v4p (0.9522827) and P_super_v4p_hc (0.9523071) banks."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("INPUT_DIR", str(ROOT))

from part6_repro_and_expand import ART, folds_y, rlogit  # noqa: E402
from part10_upgrade_psuper import nest_hill, rho, rk  # noqa: E402
from high_ev_paradigm_loop import build_v3_bank  # noqa: E402


def build_and_cache_v4p_bank(force: bool = False) -> dict:
    cache_path = ROOT / "preds" / "v4p_bank_cache.npz"
    y = pd.read_csv(ROOT / "data" / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy()
    sub_ids = pd.read_csv(ROOT / "data" / "sample_submission.csv", usecols=["id"])["id"].to_numpy()
    folds = folds_y(y)
    a0, b0 = folds[0]
    n, nt = len(y), len(sub_ids)

    if cache_path.exists() and not force:
        z = np.load(cache_path, allow_pickle=False)
        return {
            "y": y,
            "sub_ids": sub_ids,
            "folds": folds,
            "names_v4p": [str(x) for x in z["names_v4p"]],
            "O_v4p": z["O_v4p"],
            "T_v4p": z["T_v4p"],
            "Eo": z["Eo"],
            "Et": z["Et"],
            "Xo": z["Xo"],
            "Xt": z["Xt"],
            "v4p_oof": z["v4p_oof"],
            "v4p_test": z["v4p_test"],
            "v4p_hc_oof": z["v4p_hc_oof"],
            "v4p_hc_test": z["v4p_hc_test"],
            "z_champ_oof": z["z_champ_oof"],
            "z_champ_test": z["z_champ_test"],
        }

    names_v3, O_v3, T_v3, wire_info, glm_o, glm_t = build_v3_bank(y, folds)
    gb5_o = np.load(ART / "gb_bag5_oof.npy").ravel().astype(float)
    gb5_t_bag = np.load(ART / "gb_testbag_full.npy").ravel().astype(float)
    bag_o = np.load(ROOT / "preds" / "gb_gpt2_bag5_oof.npy").ravel().astype(float)
    bag_t = np.load(ROOT / "preds" / "gb_gpt2_bag5_full.npy").ravel().astype(float)

    l1_pairs = [
        ("masaya_stack_main", ROOT / "external_bank/masaya_stack/oof.npy", ROOT / "external_bank/masaya_stack/test.npy"),
        ("dhan_18_blend_lm50", ROOT / "external_bank/dhananjay_part4_out/oof/18_blend_lm50_oof.npy", ROOT / "external_bank/dhananjay_part4_out/oof/18_blend_lm50_test.npy"),
        ("dhan_18_blend_pooled2", ROOT / "external_bank/dhananjay_part4_out/oof/18_blend_pooled2_oof.npy", ROOT / "external_bank/dhananjay_part4_out/oof/18_blend_pooled2_test.npy"),
    ]
    z_cnn = np.load(ROOT / "external_bank/fresh_ds/queyrusi_cnn/predictions.npz")
    z_gen = np.load(ROOT / "external_bank/fresh_ds/queyrusi_cnn/gen_predictions.npz")

    O_v4p = np.column_stack(
        [O_v3]
        + [rk(np.load(o).ravel()) for _, o, _ in l1_pairs]
        + [rk(z_gen["oof"].ravel()), rk(z_cnn["oof"].ravel())]
    )
    T_v4p = np.column_stack(
        [T_v3]
        + [rk(np.load(t).ravel()) for _, _, t in l1_pairs]
        + [rk(z_gen["test"].ravel()), rk(z_cnn["test"].ravel())]
    )
    names_v4p = list(names_v3) + [nm for nm, _, _ in l1_pairs] + ["quey_gen", "quey_cnn"]

    import gboost as gb

    kcnt_tr, kcnt_te = gb.KCNT[:n], gb.KCNT[n:]
    n_float = min(6, kcnt_tr.shape[1])

    def stdize(tr, te):
        mu, sd = float(tr.mean()), float(tr.std()) + 1e-12
        return (tr - mu) / sd, (te - mu) / sd

    r_rare_tr, r_rare_te = stdize((kcnt_tr[:, :n_float] <= 3).mean(1), (kcnt_te[:, :n_float] <= 3).mean(1))
    ni = {nm: i for i, nm in enumerate(names_v4p)}
    mod = ["gb_bag5", "gpt2", "glm_quad", "ebm_i25_bag4", "gbn_bag5_w512"]
    mats_tr = np.column_stack([rlogit(O_v4p[:, ni[k]]) for k in mod])
    mats_te = np.column_stack([rlogit(T_v4p[:, ni[k]]) for k in mod])
    r_disp_tr, r_disp_te = stdize(mats_tr.std(1), mats_te.std(1))
    v4_o = np.load(ROOT / "preds/P_super_v4_oof.npy").ravel()
    v4_t = np.load(ROOT / "preds/P_super_v4_test.npy").ravel()
    r_tail_tr, r_tail_te = stdize(np.abs(rlogit(rk(v4_o))), np.abs(rlogit(rk(v4_t))))

    regs_tr = {"r_rare": r_rare_tr, "r_disp": r_disp_tr, "r_tail": r_tail_tr}
    regs_te = {"r_rare": r_rare_te, "r_disp": r_disp_te, "r_tail": r_tail_te}
    Eo = np.column_stack([regs_tr[r] * rlogit(O_v4p[:, ni[f]]) for r in regs_tr for f in mod])
    Et = np.column_stack([regs_te[r] * rlogit(T_v4p[:, ni[f]]) for r in regs_te for f in mod])

    Xo = np.hstack([rlogit(O_v4p), Eo]).astype(np.float64)
    Xt = np.hstack([rlogit(T_v4p), Et]).astype(np.float64)

    oof_v4p = np.zeros(n, dtype=np.float64)
    for a, b in folds:
        m = LogisticRegression(C=0.008, tol=1e-8, max_iter=300, solver="newton-cholesky").fit(Xo[a], y[a])
        oof_v4p[b] = rk(m.decision_function(Xo[b]))
    m_full = LogisticRegression(C=0.008, tol=1e-8, max_iter=300, solver="newton-cholesky").fit(Xo, y)
    te_v4p = rk(m_full.decision_function(Xt))

    ref_v4p_o = np.load(ROOT / "preds/P_super_v4p_oof.npy").ravel()
    auc_diff = abs(float(roc_auc_score(y, oof_v4p)) - float(roc_auc_score(y, ref_v4p_o)))
    assert auc_diff < 1e-6, f"V4p reproduction AUC diff {auc_diff} >= 1e-6"

    hc_o, hc_t = nest_hill(O_v4p, y, folds)
    sup_o = 0.50 * rk(gb5_o) + 0.50 * rk(bag_o)
    sup_t = 0.50 * rk(gb5_t_bag) + 0.50 * rk(bag_t)

    v4p_hc_o = rk(0.84 * rk(oof_v4p) + 0.12 * rk(hc_o) + 0.04 * rk(sup_o)).astype(np.float32)
    v4p_hc_t = rk(0.84 * rk(te_v4p) + 0.12 * rk(hc_t) + 0.04 * rk(sup_t)).astype(np.float32)
    z_champ_oof = rlogit(v4p_hc_o).astype(np.float32)
    z_champ_test = rlogit(v4p_hc_t).astype(np.float32)

    tmp_path = cache_path.with_suffix(".tmp.npz")
    np.savez(
        tmp_path,
        names_v4p=np.array(names_v4p),
        O_v4p=O_v4p.astype(np.float32),
        T_v4p=T_v4p.astype(np.float32),
        Eo=Eo.astype(np.float32),
        Et=Et.astype(np.float32),
        Xo=Xo,
        Xt=Xt,
        v4p_oof=oof_v4p.astype(np.float32),
        v4p_test=te_v4p.astype(np.float32),
        v4p_hc_oof=v4p_hc_o,
        v4p_hc_test=v4p_hc_t,
        z_champ_oof=z_champ_oof,
        z_champ_test=z_champ_test,
    )
    tmp_path.replace(cache_path)
    return {
        "y": y,
        "sub_ids": sub_ids,
        "folds": folds,
        "names_v4p": names_v4p,
        "O_v4p": O_v4p.astype(np.float32),
        "T_v4p": T_v4p.astype(np.float32),
        "Eo": Eo.astype(np.float32),
        "Et": Et.astype(np.float32),
        "Xo": Xo,
        "Xt": Xt,
        "v4p_oof": oof_v4p.astype(np.float32),
        "v4p_test": te_v4p.astype(np.float32),
        "v4p_hc_oof": v4p_hc_o,
        "v4p_hc_test": v4p_hc_t,
        "z_champ_oof": z_champ_oof,
        "z_champ_test": z_champ_test,
    }


if __name__ == "__main__":
    t0 = time.time()
    bank = build_and_cache_v4p_bank(force=False)
    y = bank["y"]
    a0, b0 = bank["folds"][0]
    print(
        f"[{time.time()-t0:.1f}s] Cached V4p bank: cols={bank['O_v4p'].shape[1]} "
        f"V4p_OOF={roc_auc_score(y, bank['v4p_oof']):.7f} (Fold0={roc_auc_score(y[b0], bank['v4p_oof'][b0]):.7f}) -> "
        f"V4p_hc_OOF={roc_auc_score(y, bank['v4p_hc_oof']):.7f} (Fold0={roc_auc_score(y[b0], bank['v4p_hc_oof'][b0]):.7f})"
    )
