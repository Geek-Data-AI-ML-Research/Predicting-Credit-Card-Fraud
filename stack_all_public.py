#!/usr/bin/env python3
"""Pool every public OOF/test artifact under this repo and stack them."""
from __future__ import annotations

import glob
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "submissions")
os.makedirs(OUT, exist_ok=True)

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


tr = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=["id", "is_fraud"])
y = tr["is_fraud"].astype(int).values
NTR = len(y)
sub = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
NTE = len(sub)
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(NTR), y))
log(f"train {NTR} test {NTE} fraud {y.mean():.4%}")

paths = {}
FULL = {}
GENERIC = {"oof", "preds", "artifacts", "output", "outputs", "working", "artifacts_extra"}


def owner_of(dn):
    b = os.path.basename(dn)
    return os.path.basename(os.path.dirname(dn)) if b in GENERIC else b


# ---------- .npy banks ----------
for f in sorted(glob.glob(f"{ROOT}/**/*.npy", recursive=True)):
    if "/._" in f or os.path.basename(f).startswith("._"):
        continue
    bn, dn = os.path.basename(f), os.path.dirname(f)
    if bn.startswith("oof_"):
        base, t = bn[4:-4], os.path.join(dn, "test_" + bn[4:])
    elif bn.endswith("_oof.npy"):
        base = bn[:-8]
        t = os.path.join(dn, base + "_test.npy")
        if not os.path.exists(t):
            t = os.path.join(dn, base + "_full.npy")
    else:
        continue
    n = owner_of(dn)[:16] + ":" + base
    if os.path.exists(t) and n not in paths:
        o_, t_ = np.load(f), np.load(t)
        if len(o_) == NTR and len(t_) == NTE:
            paths[n] = (o_.astype(np.float64), t_.astype(np.float64))
            fl = os.path.join(dn, "full_" + base + ".npy")
            if os.path.exists(fl):
                FULL[n] = np.load(fl).astype(np.float64)
            if t.endswith("_full.npy"):
                FULL[n] = t_.astype(np.float64)
        else:
            log("skip (shape)", n, o_.shape, t_.shape)

# ---------- .npz banks ----------
for f in sorted(glob.glob(f"{ROOT}/**/*.npz", recursive=True)):
    if "/._" in f or os.path.basename(f).startswith("._"):
        continue
    try:
        z = np.load(f, allow_pickle=False)
        keys = list(z.keys())
    except Exception as e:
        log("npz fail", f, e)
        continue
    log("npz", os.path.basename(f), {k: z[k].shape for k in keys})
    for k in keys:
        arr = z[k]
        if "oof" not in k.lower() or arr.ndim != 1 or len(arr) != NTR:
            continue
        test_arr = None
        for alt in (
            k.replace("oof", "test"),
            k.replace("oof", "full"),
            k.replace("oof", "tst"),
            k.replace("OOF", "TEST"),
            k.replace("oof", "pred_test"),
            "test",
            "full",
            "test_pred",
            "pred_test",
        ):
            if alt in z and getattr(z[alt], "ndim", 0) == 1 and len(z[alt]) == NTE:
                test_arr = z[alt]
                break
        # some packs store oof/test as 2d
        if test_arr is None:
            continue
        n = owner_of(os.path.dirname(f))[:16] + ":" + os.path.basename(f)[:-4] + ":" + k
        if n not in paths:
            paths[n] = (arr.astype(np.float64), test_arr.astype(np.float64))
            FULL[n] = test_arr.astype(np.float64)

# ---------- fold_predictions folders (cross-fitted) ----------
for man in glob.glob(f"{ROOT}/**/manifest.json", recursive=True):
    if "/._" in man:
        continue
    try:
        meta = json.load(open(man))
    except Exception:
        continue
    # try to assemble from fold files if documented
    log("manifest", man, list(meta)[:8] if isinstance(meta, dict) else type(meta))

log(f"collected {len(paths)} members")

# score & drop near-duplicates (corr > 0.9995 keep better AUC)
rows = []
for n, (o, t) in paths.items():
    auc = roc_auc_score(y, o)
    rows.append((n, auc, float(np.corrcoef(o, y)[0, 1])))
rows.sort(key=lambda x: -x[1])
for n, auc, c in rows[:30]:
    log(f"  {auc:.6f}  {n}")

keep = []
kept_oofs = []
for n, auc, _ in rows:
    o = paths[n][0]
    if auc < 0.94:
        continue
    dup = False
    for ko in kept_oofs:
        if np.corrcoef(o, ko)[0, 1] > 0.9995:
            dup = True
            break
    if dup:
        continue
    keep.append(n)
    kept_oofs.append(o)
log(f"kept {len(keep)} after dedupe")

# Build design: logits + ranks
Lo = np.column_stack([logit(paths[n][0]) for n in keep])
Lr = np.column_stack([rankdata(paths[n][0]) / NTR for n in keep])
X = np.hstack([Lo, Lr])
Lt = np.column_stack([logit(FULL.get(n, paths[n][1])) for n in keep])
Rt = np.column_stack([rankdata(FULL.get(n, paths[n][1])) / NTE for n in keep])
Xt = np.hstack([Lt, Rt])

# CV logistic stack
st = np.zeros(NTR)
for a, b in StratifiedKFold(5, shuffle=True, random_state=7).split(X, y):
    m = LogisticRegression(C=0.05, max_iter=5000, solver="lbfgs")
    st[b] = m.fit(X[a], y[a]).decision_function(X[b])
cv = roc_auc_score(y, st)
log(f"logit+rank stack CV {cv:.6f}")

# also pure logit stack
st2 = np.zeros(NTR)
for a, b in StratifiedKFold(5, shuffle=True, random_state=7).split(Lo, y):
    m = LogisticRegression(C=0.01, max_iter=5000, solver="lbfgs")
    st2[b] = m.fit(Lo[a], y[a]).decision_function(Lo[b])
cv2 = roc_auc_score(y, st2)
log(f"logit-only stack CV {cv2:.6f}")

if cv >= cv2:
    m = LogisticRegression(C=0.05, max_iter=5000, solver="lbfgs").fit(X, y)
    pred = expit(m.decision_function(Xt))
    tag = f"mega_logit_rank_{cv:.5f}"
    best_cv = cv
else:
    m = LogisticRegression(C=0.01, max_iter=5000, solver="lbfgs").fit(Lo, y)
    pred = expit(m.decision_function(Lt))
    tag = f"mega_logit_{cv2:.5f}"
    best_cv = cv2

# power-mean blend of top-k by OOF AUC as diversity backup
topk = [n for n, auc, _ in rows if auc >= rows[0][1] - 0.001][:12]
blend = np.mean([paths[n][1] for n in topk], axis=0)
# fuse stack with blend via second-level CV on OOF
oof_blend = np.mean([paths[n][0] for n in topk], axis=0)
fuse_oof = np.zeros(NTR)
Z = np.column_stack([st if cv >= cv2 else st2, logit(oof_blend)])
for a, b in StratifiedKFold(5, shuffle=True, random_state=11).split(Z, y):
    fuse_oof[b] = LogisticRegression(C=1.0, max_iter=2000).fit(Z[a], y[a]).decision_function(Z[b])
cv3 = roc_auc_score(y, fuse_oof)
log(f"fuse stack+topkblend CV {cv3:.6f}")
if cv3 > best_cv:
    Zm = np.column_stack([logit(pred), logit(blend)])  # approximate: use test stack pred logit
    # better: refit on full OOF then apply
    m3 = LogisticRegression(C=1.0, max_iter=2000).fit(Z, y)
    Zt = np.column_stack([logit(pred), logit(blend)])
    pred = expit(m3.decision_function(Zt))
    tag = f"mega_fuse_{cv3:.5f}"
    best_cv = cv3

sub[TARGET := "is_fraud"] = pred
path = os.path.join(OUT, f"{tag}.csv")
sub.to_csv(path, index=False)
sub.to_csv(os.path.join(OUT, "latest.csv"), index=False)
log("wrote", path, "mean", float(pred.mean()), "cv", best_cv)
print(path)
