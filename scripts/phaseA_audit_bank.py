#!/usr/bin/env python3
"""Phase A — audit / align public OOF+test banks for overnight Part-6 run.

Hard constraints:
  - SKF(5, shuffle=True, random_state=42)
  - True sizes: N_TRAIN=699_960, N_TEST=299_984 (plan text 559968/300000 is wrong)
  - Quarantine OOFs that fail Fold2-high / Fold4-low signature
  - Dedup only Spearman rho > 0.9997 (do NOT drop milder clones)
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(os.environ.get("S6E11_ROOT", Path(__file__).resolve().parents[1]))
DATA = ROOT / "data"
BANK = ROOT / "external_bank"
OUT = ROOT / "logs" / "overnight"
PRED_A = ROOT / "preds" / "phaseA"
N_TRAIN = 699_960
N_TEST = 299_984
SEED = 42
RHO_DEDUP = 0.9997
TARGET = "is_fraud"


def locked_folds(y):
    return list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))


def fold_aucs(y, oof, folds):
    return [float(roc_auc_score(y[va], oof[va])) for _, va in folds]


def signature_ok(fa: list[float]) -> tuple[bool, str]:
    """SKF(rs=42) fingerprint: Fold2 highest (~0.955+), Fold4 lowest (~0.945)."""
    if len(fa) != 5:
        return False, "need_5_folds"
    f2, f4 = fa[2], fa[4]
    if f2 < max(fa) - 1e-12:
        return False, f"fold2_not_max f2={f2:.5f} max={max(fa):.5f}"
    if f4 > min(fa) + 1e-12:
        return False, f"fold4_not_min f4={f4:.5f} min={min(fa):.5f}"
    # Soft relative order on the named folds (Fold3 may sit between 2 and 1)
    if not (f2 > fa[1] and fa[0] > f4):
        return False, f"order_fail F2={f2:.5f} F1={fa[1]:.5f} F0={fa[0]:.5f} F4={f4:.5f}"
    if f2 < 0.952 or f4 > 0.948:
        return False, f"magnitude_off F2={f2:.5f} F4={f4:.5f}"
    return True, "ok"


def load_vec(path: Path, n: int) -> np.ndarray | None:
    try:
        if path.suffix == ".npy":
            a = np.load(path, allow_pickle=False)
        elif path.suffix == ".npz":
            z = np.load(path, allow_pickle=True)
            if isinstance(z, np.lib.npyio.NpzFile):
                for k in ("oof", "pred", "p", "y_pred", "is_fraud"):
                    if k in z.files and np.asarray(z[k]).size == n:
                        a = np.asarray(z[k]).reshape(-1)
                        break
                else:
                    # first 1d array of length n
                    a = None
                    for k in z.files:
                        v = np.asarray(z[k]).reshape(-1)
                        if v.size == n:
                            a = v
                            break
                    if a is None:
                        return None
            else:
                a = np.asarray(z).reshape(-1)
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
            cols = [c for c in df.columns if c.lower() not in ("id", "index")]
            if not cols:
                return None
            a = df[cols[0]].to_numpy(dtype=np.float64)
        else:
            return None
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        if a.size != n:
            return None
        if not np.isfinite(a).all():
            return None
        return a
    except Exception:
        return None


def discover_pairs(bank: Path):
    """Yield candidate (name, oof_path, test_path|None, has_refit_hint)."""
    oof_files = []
    for p in bank.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith("._"):
            continue
        low = p.name.lower()
        if any(x in low for x in ("oof", "train_pred", "pair_oof")):
            oof_files.append(p)
        elif low.endswith("_oof.npy") or low.endswith(".oof.npy"):
            oof_files.append(p)

    seen = set()
    for op in oof_files:
        if str(op) in seen:
            continue
        seen.add(str(op))
        stem = op.name
        # guess test sibling
        cands = []
        for pat, repl in [
            ("_oof", "_test"),
            ("_oof", "_full"),
            ("oof_", "test_"),
            ("oof_", "full_"),
            ("pair_oof", "pair_test"),
            ("_oof.npy", "_test.npy"),
            ("_oof.npy", "_full.npy"),
        ]:
            if pat in stem:
                cands.append(op.with_name(stem.replace(pat, repl, 1)))
        # same dir search
        base = re.sub(r"[_]?oof[_.]?", "", stem, flags=re.I)
        for sib in op.parent.iterdir():
            if not sib.is_file() or sib == op:
                continue
            sl = sib.name.lower()
            if any(t in sl for t in ("test", "full", "sub", "refit")) and base[:8] in sib.name.lower():
                cands.append(sib)
        test_p = None
        has_refit = False
        for c in cands:
            if c.exists():
                test_p = c
                if "full" in c.name.lower() or "refit" in c.name.lower():
                    has_refit = True
                break
        rel = str(op.relative_to(bank))
        yield rel, op, test_p, has_refit


def _progress(msg: str) -> None:
    print(f"[phaseA] {msg}", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PRED_A.mkdir(parents=True, exist_ok=True)
    CKPT = ROOT / "checkpoints" / "overnight"
    CKPT.mkdir(parents=True, exist_ok=True)

    _progress(f"loading train from {DATA / 'train.csv'}")
    train = pd.read_csv(DATA / "train.csv")
    y = train[TARGET].astype(int).to_numpy()
    assert len(y) == N_TRAIN, f"train size {len(y)} != {N_TRAIN}"
    folds = locked_folds(y)
    _progress(f"n_train={len(y)} scanning BANK={BANK}")

    # reference ctrl if present
    ctrl_path = ROOT / "preds" / "public_artifacts" / "lgb_diglead_l3_bag_oof.npy"
    if not ctrl_path.exists():
        ctrl_path = ROOT / "preds" / "stage9" / "lgb_diglead_l3_bag_oof.npy"
    ctrl = load_vec(ctrl_path, N_TRAIN) if ctrl_path.exists() else None
    ctrl_folds = fold_aucs(y, ctrl, folds) if ctrl is not None else None

    members = []
    quarantined = []
    n_seen = 0
    for rel, op, tp, has_refit in discover_pairs(BANK):
        n_seen += 1
        if n_seen % 10 == 1:
            _progress(f"discover {n_seen}: {rel}")
        oof = load_vec(op, N_TRAIN)
        if oof is None:
            # try as test-sized by mistake
            continue
        te = load_vec(tp, N_TEST) if tp is not None else None
        fa = fold_aucs(y, oof, folds)
        ok, reason = signature_ok(fa)
        rec = {
            "name": rel.replace("/", "__").replace(".", "_"),
            "path_oof": str(op),
            "path_test": str(tp) if tp else None,
            "oof_auc": float(roc_auc_score(y, oof)),
            "fold_aucs": fa,
            "has_test": te is not None,
            "has_refit_hint": bool(has_refit or (tp and "full" in tp.name.lower())),
            "signature_ok": ok,
            "signature_reason": reason,
        }
        if not ok:
            quarantined.append(rec)
            continue
        if te is None:
            rec["signature_reason"] = "ok_but_missing_test"
            # still keep OOF for nested CV / quarantine from test combiner later
        members.append((rec, oof, te))

    # also ingest flat public_artifacts / alex without discover miss
    extra_roots = [
        ROOT / "preds" / "public_artifacts",
        ROOT / "preds" / "public_alex",
        ROOT / "preds" / "public_tabpfn",
        ROOT / "preds" / "unattended",
        ROOT / "preds" / "stage9",
        ROOT / "preds" / "stage7",
    ]
    for er in extra_roots:
        if not er.exists():
            continue
        # symlink already under bank; also scan directly
        for op in er.rglob("*oof*.npy"):
            if op.name.startswith("._"):
                continue
            rel = f"local::{op.relative_to(ROOT)}"
            if any(m[0]["path_oof"] == str(op) for m in members):
                continue
            oof = load_vec(op, N_TRAIN)
            if oof is None:
                continue
            # sibling test/full
            te = None
            has_refit = False
            for suf in ("_test.npy", "_full.npy", "_te.npy"):
                cand = op.with_name(op.name.replace("_oof.npy", suf).replace("oof_", "test_"))
                if cand.exists():
                    te = load_vec(cand, N_TEST)
                    if te is not None:
                        has_refit = "full" in cand.name.lower()
                        break
            # tabpfn csv pairs
            if te is None and op.suffix == ".npy":
                pass
            fa = fold_aucs(y, oof, folds)
            ok, reason = signature_ok(fa)
            rec = {
                "name": rel.replace("/", "__"),
                "path_oof": str(op),
                "path_test": None,
                "oof_auc": float(roc_auc_score(y, oof)),
                "fold_aucs": fa,
                "has_test": te is not None,
                "has_refit_hint": has_refit,
                "signature_ok": ok,
                "signature_reason": reason,
            }
            if te is not None:
                # find path again for record
                for suf in ("_test.npy", "_full.npy"):
                    cand = op.with_name(op.name.replace("_oof.npy", suf))
                    if cand.exists() and load_vec(cand, N_TEST) is not None:
                        rec["path_test"] = str(cand)
                        break
            if not ok:
                quarantined.append(rec)
            else:
                members.append((rec, oof, te))

    # tabpfn csv pairs
    for csvp in (ROOT / "preds" / "public_tabpfn").glob("pair_oof_*.csv"):
        oof = load_vec(csvp, N_TRAIN)
        if oof is None:
            continue
        te_p = csvp.with_name(csvp.name.replace("pair_oof_", "pair_test_"))
        te = load_vec(te_p, N_TEST) if te_p.exists() else None
        fa = fold_aucs(y, oof, folds)
        ok, reason = signature_ok(fa)
        rec = {
            "name": f"tabpfn::{csvp.stem}",
            "path_oof": str(csvp),
            "path_test": str(te_p) if te_p.exists() else None,
            "oof_auc": float(roc_auc_score(y, oof)),
            "fold_aucs": fa,
            "has_test": te is not None,
            "has_refit_hint": False,
            "signature_ok": ok,
            "signature_reason": reason,
        }
        if ok:
            members.append((rec, oof, te))
        else:
            quarantined.append(rec)

    # Dedup rho > 0.9997 only (subsample — full 700k×O(n²) Spearman hangs for hours)
    _progress(f"dedup starting n_members={len(members)} (50k-row spearman subsample)")
    rng = np.random.RandomState(SEED)
    dedup_idx = np.sort(rng.choice(N_TRAIN, size=min(50_000, N_TRAIN), replace=False))
    keep_idx = []
    dropped_dup = []
    for i, (rec, oof, te) in enumerate(members):
        if i % 20 == 0:
            _progress(f"dedup {i}/{len(members)} kept={len(keep_idx)}")
        dup_of = None
        r = None
        oi = oof[dedup_idx]
        for j in keep_idx:
            r = spearmanr(oi, members[j][1][dedup_idx]).correlation
            if r is not None and r > RHO_DEDUP:
                dup_of = members[j][0]["name"]
                break
        if dup_of:
            dropped_dup.append({"name": rec["name"], "dup_of": dup_of, "rho": float(r)})
        else:
            keep_idx.append(i)
    _progress(f"dedup done kept={len(keep_idx)} dropped={len(dropped_dup)}")

    kept = [members[i] for i in keep_idx]
    # pack arrays
    names = [r["name"] for r, _, _ in kept]
    O = np.column_stack([o for _, o, _ in kept]).astype(np.float32)
    T = np.full((N_TEST, len(kept)), np.nan, dtype=np.float32)
    refit_mask = []
    for j, (rec, _, te) in enumerate(kept):
        refit_mask.append(bool(rec["has_refit_hint"] and te is not None))
        if te is not None:
            T[:, j] = te.astype(np.float32)

    np.savez_compressed(
        PRED_A / "bank_aligned.npz",
        names=np.array(names, dtype=object),
        O=O,
        T=T,
        refit_mask=np.array(refit_mask, dtype=bool),
        y=y.astype(np.int8),
    )

    report = {
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "cv": "StratifiedKFold(5, shuffle=True, random_state=42)",
        "ctrl_folds": ctrl_folds,
        "n_kept": len(kept),
        "n_quarantined": len(quarantined),
        "n_dedup_dropped": len(dropped_dup),
        "rho_dedup": RHO_DEDUP,
        "members": [r for r, _, _ in kept],
        "quarantined": quarantined,
        "dedup_dropped": dropped_dup,
        "leaderboard": sorted(
            [{"name": r["name"], "oof_auc": r["oof_auc"], "folds": r["fold_aucs"], "refit": r["has_refit_hint"]} for r, _, _ in kept],
            key=lambda x: -x["oof_auc"],
        ),
    }
    (OUT / "phaseA_audit.json").write_text(json.dumps(report, indent=2))
    (CKPT / "phaseA_audit.json").write_text(json.dumps(report, indent=2))
    # also dump lightweight resume manifest
    (CKPT / "phaseA_done.json").write_text(
        json.dumps({"kept": len(kept), "quarantined": len(quarantined), "dedup": len(dropped_dup), "bank_npz": str(PRED_A / "bank_aligned.npz")}, indent=2)
    )
    _progress(f"DONE kept={len(kept)} quarantined={len(quarantined)} dedup={len(dropped_dup)}")
    print(json.dumps({"kept": len(kept), "quarantined": len(quarantined), "dedup": len(dropped_dup), "top5": report["leaderboard"][:5]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
