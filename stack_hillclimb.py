#!/usr/bin/env python3
"""Rank + hill-climb blend of all public artifacts (pattern from LB 0.95178 notebook)."""
from __future__ import annotations

import glob
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
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
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


tr = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=["id", "is_fraud"])
y = tr["is_fraud"].astype(int).values
NTR = len(y)
sub = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
NTE = len(sub)
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(NTR), y))
log(f"train {NTR} test {NTE}")

paths = {}
FULL = {}
GENERIC = {"oof", "preds", "artifacts", "output", "outputs", "working", "artifacts_extra", "fold_predictions"}


def owner_of(dn):
    b = os.path.basename(dn)
    return os.path.basename(os.path.dirname(dn)) if b in GENERIC else b


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
    n = owner_of(dn)[:14] + ":" + base
    if os.path.exists(t) and n not in paths:
        o_, t_ = np.load(f), np.load(t)
        if len(o_) == NTR and len(t_) == NTE:
            paths[n] = (o_.astype(np.float64), t_.astype(np.float64))
            if t.endswith("_full.npy"):
                FULL[n] = t_.astype(np.float64)


# directory-style oof.npy / test.npy pairs (e.g. distillation study records)
for f in sorted(glob.glob(f"{ROOT}/**/oof.npy", recursive=True)):
    if "/._" in f:
        continue
    dn = os.path.dirname(f)
    tpath = os.path.join(dn, "test.npy")
    if not os.path.exists(tpath):
        continue
    n = "dir:" + os.path.relpath(dn, ROOT).replace(os.sep, "/")[:60]
    if n in paths:
        continue
    o_, t_ = np.load(f), np.load(tpath)
    if len(o_) == NTR and len(t_) == NTE:
        paths[n] = (o_.astype(np.float64), t_.astype(np.float64))

for f in sorted(glob.glob(f"{ROOT}/**/*.npz", recursive=True)):
    if "/._" in f or os.path.basename(f).startswith("._"):
        continue
    try:
        z = np.load(f, allow_pickle=False)
    except Exception:
        continue
    keys = list(z.keys())
    # standard oof/test pair
    for k in keys:
        if "oof" not in k.lower() or z[k].ndim != 1 or len(z[k]) != NTR:
            continue
        for alt in (
            k.replace("oof", "test"),
            k.replace("oof", "full"),
            k.replace("oof", "tst"),
            "test",
            "full",
            "stack_test",
        ):
            if alt in z and getattr(z[alt], "ndim", 0) == 1 and len(z[alt]) == NTE:
                n = owner_of(os.path.dirname(f))[:14] + ":" + os.path.basename(f)[:-4] + ":" + k
                paths[n] = (z[k].astype(np.float64), z[alt].astype(np.float64))
                break
    # fold_test mean as test prediction, seed_oof mean as oof
    if "seed_oof" in z and "fold_test" in z:
        o = z["seed_oof"].mean(1) if z["seed_oof"].ndim == 2 else z["seed_oof"]
        t = z["fold_test"].mean(0) if z["fold_test"].ndim == 2 else z["fold_test"]
        if len(o) == NTR and len(t) == NTE:
            n = owner_of(os.path.dirname(f))[:14] + ":" + os.path.basename(f)[:-4] + ":seedmean"
            paths[n] = (o.astype(np.float64), t.astype(np.float64))
    if "oof" in z and "fold_test" in z and z["oof"].ndim == 1 and len(z["oof"]) == NTR:
        t = z["fold_test"].mean(0) if z["fold_test"].ndim == 2 else z["fold_test"]
        if len(t) == NTE:
            n = owner_of(os.path.dirname(f))[:14] + ":" + os.path.basename(f)[:-4] + ":foldmean"
            paths[n] = (z["oof"].astype(np.float64), t.astype(np.float64))

# assemble cross-fitted fold packs: each model has fold0..fold4 with valid_index/valid/test
fold_root = os.path.join(ROOT, "artifacts_extra", "s6e11-cross-fitted-artifacts", "fold_predictions")
if os.path.isdir(fold_root):
    models = {}
    for f in sorted(glob.glob(fold_root + "/*_fold*.npz")):
        base = os.path.basename(f).rsplit("_fold", 1)[0]
        models.setdefault(base, []).append(f)
    for base, files in models.items():
        oof = np.zeros(NTR, np.float64)
        test_sum = np.zeros(NTE, np.float64)
        ok = True
        for f in files:
            z = np.load(f)
            if not {"valid_index", "valid", "test"} <= set(z.keys()):
                ok = False
                break
            oof[z["valid_index"]] = z["valid"]
            test_sum += z["test"]
        if ok and len(files) == 5:
            n = "crossfit:" + base
            paths[n] = (oof, test_sum / 5.0)
            log("assembled", n, "auc", roc_auc_score(y, oof))

log(f"{len(paths)} members collected")
names = list(paths)
O = np.column_stack([rankdata(paths[n][0]) / NTR for n in names])
Tm = np.column_stack([rankdata(paths[n][1]) / NTE for n in names])
auc = {n: float(roc_auc_score(y, O[:, i])) for i, n in enumerate(names)}
for n, a in sorted(auc.items(), key=lambda kv: -kv[1])[:25]:
    log(f"  {a:.6f}  {n}")

thr = max(auc.values()) - 0.0025
keep = [i for i, n in enumerate(names) if auc[n] >= thr]
names = [names[i] for i in keep]
O = O[:, keep]
Tm = Tm[:, keep]
log(f"{len(names)} kept (AUC >= {thr:.5f})")


def hill_climb(O_, yy, iters=40):
    w = np.zeros(O_.shape[1])
    w[int(np.argmax([roc_auc_score(yy, O_[:, i]) for i in range(O_.shape[1])]))] = 1
    cur = roc_auc_score(yy, O_ @ w)
    for _ in range(iters):
        sc, i = max(
            (roc_auc_score(yy, O_ @ (w + np.eye(O_.shape[1])[i])), i)
            for i in range(O_.shape[1])
        )
        if sc <= cur + 1e-7:
            break
        w[i] += 1
        cur = sc
    return w / w.sum()


cv = {k: np.zeros(NTR) for k in ["mean", "hill", "stack", "stack_c01", "mix"]}
for a, b in FOLDS:
    cv["mean"][b] = O[b].mean(1)
    cv["hill"][b] = O[b] @ hill_climb(O[a], y[a])
    cv["stack"][b] = (
        LogisticRegression(C=1.0, max_iter=3000)
        .fit(logit(O[a]), y[a])
        .decision_function(logit(O[b]))
    )
    cv["stack_c01"][b] = (
        LogisticRegression(C=0.01, max_iter=3000)
        .fit(logit(O[a]), y[a])
        .decision_function(logit(O[b]))
    )
    cv["mix"][b] = rankdata(cv["hill"][b]) / len(b) + rankdata(cv["stack"][b]) / len(b)

scores = {k: float(roc_auc_score(y, v)) for k, v in cv.items()}
log("nested-CV:", {k: round(v, 6) for k, v in scores.items()})
best = max(scores, key=scores.get)

w = hill_climb(O, y)
st = LogisticRegression(C=1.0, max_iter=3000).fit(logit(O), y)
st01 = LogisticRegression(C=0.01, max_iter=3000).fit(logit(O), y)
final = {
    "mean": Tm.mean(1),
    "hill": Tm @ w,
    "stack": st.decision_function(logit(Tm)),
    "stack_c01": st01.decision_function(logit(Tm)),
}
final["mix"] = rankdata(final["hill"]) / NTE + rankdata(final["stack"]) / NTE

# prefer full-refit test preds when available
Tm2 = Tm.copy()
for j, n in enumerate(names):
    if n in FULL:
        Tm2[:, j] = rankdata(FULL[n]) / NTE
f2 = {
    "mean": Tm2.mean(1),
    "hill": Tm2 @ w,
    "stack": st.decision_function(logit(Tm2)),
    "stack_c01": st01.decision_function(logit(Tm2)),
}
f2["mix"] = rankdata(f2["hill"]) / NTE + rankdata(f2["stack"]) / NTE

sub["is_fraud"] = rankdata(f2[best]) / NTE
path = os.path.join(OUT, f"hillclimb_{best}_{scores[best]:.5f}.csv")
sub.to_csv(path, index=False)
sub.to_csv(os.path.join(OUT, "latest.csv"), index=False)
json.dump(
    {"members": auc, "kept": names, "blend": scores, "best": best, "weights": dict(zip(names, w.tolist()))},
    open(os.path.join(OUT, "hillclimb_summary.json"), "w"),
    indent=1,
)
log(f"BEST={best} CV={scores[best]:.6f} wrote {path}")
log("hill weights:", {n: round(float(v), 3) for n, v in zip(names, w) if v > 0})
print(path)
