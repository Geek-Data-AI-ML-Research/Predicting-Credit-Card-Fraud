#!/usr/bin/env python3
"""Stage 6.2: Self-trained RealMLP + TabM (pytabkit CUDA) on te_smooth50 view.

NO kaggle submit. Promote vs te_xgb_smooth50 (0.949447) via Stage 2.2 gate.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from te_smooth50_features import build_codes, build_fold_matrix  # noqa: E402
from validation_stage2_handbook_cv import promotion_gate  # noqa: E402
from validation_stage3_baselines import SEED, TARGET, locked_folds  # noqa: E402

DATA = os.path.join(ROOT, "data")
LOGS = os.path.join(ROOT, "logs")
PRED = os.path.join(ROOT, "preds", "stage6")
CTRL_OOF = 0.949447


def load_control(y, folds):
    s4 = os.path.join(LOGS, "stage4_levers_result.json")
    if os.path.exists(s4):
        doc = json.load(open(s4))
        for a in doc.get("new_arms", doc.get("arms", [])):
            if a.get("name") == "te_xgb_smooth50":
                return a["oof_roc_auc"], [f["valid_roc_auc"] for f in a["folds"]]
    p = os.path.join(ROOT, "preds", "stage4", "te_xgb_smooth50_oof.npy")
    if os.path.exists(p):
        o = np.load(p)
        return float(roc_auc_score(y, o)), [
            float(roc_auc_score(y[va], o[va])) for _, va in folds
        ]
    return CTRL_OOF, [CTRL_OOF] * 5


def run_arm(name, make_model, train, test, y, folds, pack, ctrl_oof, ctrl_folds):
    dig_tr, dig_te, dig_num, te_names, codes_tr, codes_te = pack
    oof = np.zeros(len(y), np.float64)
    test_sum = np.zeros(len(test), np.float64)
    fold_recs = []
    t0 = time.time()
    print(f"\n=== Arm {name} ===", flush=True)
    for f, (tr, va) in enumerate(folds):
        X_tr, X_va, X_te = build_fold_matrix(
            train,
            test,
            dig_tr,
            dig_te,
            dig_num,
            codes_tr,
            codes_te,
            te_names,
            y,
            tr,
            va,
            smooth=50.0,
        )
        # align columns
        cols = X_tr.columns
        X_va = X_va.reindex(columns=cols, fill_value=0)
        X_te = X_te.reindex(columns=cols, fill_value=0)
        m = make_model(f)
        m.fit(X_tr, y[tr])
        if hasattr(m, "predict_proba"):
            tr_p = m.predict_proba(X_tr)[:, 1]
            va_p = m.predict_proba(X_va)[:, 1]
            te_p = m.predict_proba(X_te)[:, 1]
        else:
            tr_p = m.predict(X_tr)
            va_p = m.predict(X_va)
            te_p = m.predict(X_te)
            if tr_p.ndim > 1:
                tr_p, va_p, te_p = tr_p[:, 1], va_p[:, 1], te_p[:, 1]
        oof[va] = va_p
        test_sum += te_p
        fa = float(roc_auc_score(y[va], va_p))
        gap = float(roc_auc_score(y[tr], tr_p) - fa)
        fold_recs.append({"fold": f, "valid_roc_auc": fa, "train_valid_gap": gap})
        print(f"  fold{f}: auc={fa:.6f} gap={gap:.6f} ({time.time()-t0:.0f}s)", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    oof_auc = float(roc_auc_score(y, oof))
    folds_v = [r["valid_roc_auc"] for r in fold_recs]
    gate = promotion_gate(ctrl_folds, folds_v, ctrl_oof, oof_auc)
    te = test_sum / len(folds)
    os.makedirs(PRED, exist_ok=True)
    np.save(os.path.join(PRED, f"oof_{name}.npy"), oof.astype(np.float32))
    np.save(os.path.join(PRED, f"test_{name}_full.npy"), te.astype(np.float32))
    # also alias names requested
    np.save(os.path.join(PRED, f"{name}_oof.npy"), oof.astype(np.float32))
    np.save(os.path.join(PRED, f"{name}_full.npy"), te.astype(np.float32))
    return {
        "name": name,
        "oof_roc_auc": oof_auc,
        "folds": fold_recs,
        "gate_vs_te_xgb_smooth50": gate,
        "promote": gate["promote"],
    }


def main():
    assert torch.cuda.is_available(), "CUDA required for stage 6.2"
    print("CUDA:", torch.cuda.get_device_name(0), flush=True)
    from pytabkit import RealMLP_TD_Classifier, TabM_D_Classifier

    t0 = time.time()
    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    test = pd.read_csv(os.path.join(DATA, "test.csv"))
    y = train[TARGET].astype(int).values
    folds = locked_folds(y)
    ctrl_oof, ctrl_folds = load_control(y, folds)
    print(f"control te_xgb_smooth50 OOF={ctrl_oof:.6f}", flush=True)
    print("building codes...", flush=True)
    pack = build_codes(train, test)

    def make_realmlp(f):
        return RealMLP_TD_Classifier(
            device="cuda",
            n_cv=1,
            val_metric_name="1-auc_ovr",
            n_epochs=64,
            batch_size=1024,
            random_state=SEED + f,
            verbosity=0,
            tmp_folder=os.path.join("/tmp/s6e11_tmp", f"realmlp_f{f}"),
        )

    def make_tabm(f):
        return TabM_D_Classifier(
            device="cuda",
            n_cv=1,
            val_metric_name="1-auc_ovr",
            n_epochs=64,
            batch_size=1024,
            random_state=SEED + f,
            verbosity=0,
            tmp_folder=os.path.join("/tmp/s6e11_tmp", f"tabm_f{f}"),
        )

    arms = []
    arms.append(
        run_arm(
            "self_realmlp_smooth50",
            make_realmlp,
            train,
            test,
            y,
            folds,
            pack,
            ctrl_oof,
            ctrl_folds,
        )
    )
    arms.append(
        run_arm(
            "self_tabm_smooth50",
            make_tabm,
            train,
            test,
            y,
            folds,
            pack,
            ctrl_oof,
            ctrl_folds,
        )
    )

    doc = {
        "stage": "6.2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - t0, 1),
        "device": "cuda",
        "control": {"name": "te_xgb_smooth50", "oof": ctrl_oof, "folds": ctrl_folds},
        "arms": arms,
        "promoted": [a["name"] for a in arms if a["promote"]],
        "policy": "NO_KAGGLE_SUBMIT",
    }
    os.makedirs(LOGS, exist_ok=True)
    path = os.path.join(LOGS, "stage6_2_gpu_neural_result.json")
    json.dump(doc, open(path, "w"), indent=2)
    print(json.dumps({"promoted": doc["promoted"], "oofs": [(a["name"], a["oof_roc_auc"]) for a in arms]}, indent=2))
    print(f"wrote {path} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
