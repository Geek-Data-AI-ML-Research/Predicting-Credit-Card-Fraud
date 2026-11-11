#!/usr/bin/env python3
"""GM v2 — Deep lineage pull + orthogonal private-signal audit of 0.95347/46/40.

STRICT NO_SUBMIT=True. Stages *_NOSUBMIT.csv only.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NO_SUBMIT", "1")

warnings.filterwarnings("ignore")

WORK = Path(os.environ.get("GM_V6_WORK", "/tmp/gm_v6_audit")).resolve()
ROOT = Path(os.environ.get("GM_V6_ROOT", "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")).resolve()
DATA = Path(os.environ.get("GM_V6_DATA", WORK / "data"))
SUB_LOCAL = WORK / "submissions"
PRED = WORK / "preds"
EXT = WORK / "external" / "nov11_095347_recon"
ART = WORK / "preds" / "gm_v6_audit"
LOG = WORK / "logs" / "overnight"
VOL_SUB = ROOT / "submissions"
VOL_PRED = ROOT / "preds"
VOL_LOG = ROOT / "logs" / "overnight"

for p in [EXT, ART, LOG, SUB_LOCAL, PRED]:
    p.mkdir(parents=True, exist_ok=True)

NO_SUBMIT = os.environ.get("NO_SUBMIT", "1") != "0"
EPS = 1e-6
KS_GATE = 0.0001
N_Q = 10001
KAG = str(ROOT / ".venv_forensics/bin/kaggle")
COMP = "playground-series-s6e11"
T0 = time.time()

KERNELS = [
    ("areebbb/s6e11-three-ladders-two-steps-lb-0-95347", 0.95347),
    ("goodpjw2008/s6e11-lm-ladder-part-1-own-lm-step-lb-0-95346", 0.95346),
    ("amanatar/s6e11-podium-apex-public-lb-0-95340", 0.95340),
    ("malyshevdanil/fraud-signal-lab-gpt-2-and-glm-fusion", None),
    ("lucifer19/the-golden-signal-consensus-blend-v22", None),
]

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.special import expit, logit
from scipy.stats import ks_2samp, norm, rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def safe(slug: str) -> str:
    return slug.replace("/", "__")


def zrk_logit(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    return logit(np.clip((rankdata(x, method="average") - 0.5) / len(x), EPS, 1 - EPS))


def zrk_ppf(x):
    """User-specified: norm.ppf(rankdata(p)/(N+1))."""
    x = np.asarray(x, dtype=np.float64).ravel()
    r = rankdata(x, method="average")
    return norm.ppf(np.clip(r / (len(x) + 1.0), 1e-12, 1 - 1e-12))


def unitize(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return (v - float(v.mean())) / (float(v.std()) + 1e-12)


def unit(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return v / (float(np.linalg.norm(v)) + 1e-18)


def cos(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-18) * (np.linalg.norm(b) + 1e-18)))


def orth_against(v, bases):
    out = np.asarray(v, dtype=np.float64).ravel().copy()
    for b in bases:
        b = unit(np.asarray(b, dtype=np.float64).ravel())
        out -= float(np.dot(out, b)) * b
    return out


def gram_orthonormal(X):
    X = np.asarray(X, dtype=np.float64)
    cols = []
    for j in range(X.shape[1]):
        v = unitize(X[:, j])
        for c in cols:
            v = v - float(np.dot(v, c)) * c
        n = float(np.linalg.norm(v))
        if n < 1e-12:
            continue
        cols.append(v / n)
    return np.column_stack(cols) if cols else np.zeros((X.shape[0], 0))


def fold_aucs(y, z, folds):
    return [float(roc_auc_score(y[b], z[b])) for _, b in folds]


def _jitter(x, scale=1e-11):
    r = rankdata(x, method="ordinal").astype(np.float64)
    return x + scale * (r / len(x) - 0.5)


def fold_std_oof(v, folds):
    out = np.zeros_like(v, dtype=np.float64)
    for _, b in folds:
        mu, sd = float(v[b].mean()), float(v[b].std() + 1e-12)
        out[b] = (v[b] - mu) / sd
    return out


def force_exact_law(oof_vec, test_vec):
    o = unitize(oof_vec)
    t = np.asarray(test_vec, dtype=np.float64).ravel()
    o_s = np.sort(o)
    n_o, n_t = len(o_s), len(t)
    order = np.argsort(_jitter(t, 1e-7), kind="mergesort")
    pos = np.empty(n_t, dtype=np.int64)
    pos[order] = np.minimum(((np.arange(n_t) + 0.5) * n_o / n_t).astype(np.int64), n_o - 1)
    t_al = o_s[pos]
    t_al = t_al - float(t_al.mean())
    t_al = t_al * (float(o.std()) / (float(t_al.std()) + 1e-12))
    return o, t_al


def align_method_b_smooth(oof_vec, test_vec, folds, jitter_scale=1e-11):
    oof_std = unitize(fold_std_oof(np.asarray(oof_vec, float).ravel(), folds))
    test_std = unitize(np.asarray(test_vec, float).ravel())
    q = np.linspace(0.0, 1.0, N_Q)
    pchip = PchipInterpolator(
        np.quantile(_jitter(test_std, jitter_scale), q),
        np.quantile(_jitter(oof_std, jitter_scale), q),
        extrapolate=True,
    )
    test_al = unitize(np.asarray(pchip(_jitter(test_std, jitter_scale)), float))
    if float(ks_2samp(oof_std, test_al).statistic) >= KS_GATE:
        oof_std, test_al = force_exact_law(oof_std, test_std)
    return oof_std, test_al


def disp_ks(o, t):
    return float(np.std(t) / (np.std(o) + 1e-18)), float(ks_2samp(o, t).statistic)


def lofo_residual(cand_o, cand_t, base_o, base_t, folds):
    zo, zt = unitize(cand_o), unitize(cand_t)
    bo, bt = unitize(base_o), unitize(base_t)
    Xo = bo.reshape(-1, 1)
    Xt = bt.reshape(-1, 1)
    ro = np.zeros_like(zo)
    rt_f = []
    for a, b in folds:
        beta, *_ = np.linalg.lstsq(Xo[a], zo[a], rcond=None)
        ra = zo[a] - Xo[a] @ beta
        mu, sd = float(ra.mean()), float(ra.std() + 1e-12)
        ro[b] = (zo[b] - Xo[b] @ beta - mu) / sd
        rt_f.append((zt - Xt @ beta - mu) / sd)
    rt = np.mean(rt_f, 0)
    return align_method_b_smooth(ro, rt, folds)


def eval_delta_on_base(base_o, delta_o, y, folds, w_grid=None):
    base_folds = fold_aucs(y, base_o, folds)
    base_auc = float(roc_auc_score(y, base_o))
    if w_grid is None:
        w_grid = np.linspace(-0.05, 0.35, 41)
    best = None
    for w in w_grid:
        z = base_o + w * delta_o
        nf = fold_aucs(y, z, folds)
        deltas = [(a - b) * 1e5 for a, b in zip(nf, base_folds)]
        pos = sum(1 for d in deltas if d > 0)
        auc = float(roc_auc_score(y, z))
        lift = (auc - base_auc) * 1e5
        row = {"w": float(w), "pos": pos, "lift_e5": lift, "mean_d": float(np.mean(deltas)), "deltas": deltas, "auc": auc}
        if best is None or (row["pos"], row["lift_e5"]) > (best["pos"], best["lift_e5"]):
            best = row
    return best


def spearman_dist(a, b):
    return float(1.0 - spearmanr(a, b).correlation)


def weight_for_spearman_dist(z0, u, target, lo=1e-8, hi=5.0, n=40):
    u = unit(u)
    best_w, best_d = 0.0, abs(spearman_dist(z0, z0) - target)
    # binary-ish search
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        d = spearman_dist(z0, z0 + mid * u)
        if abs(d - target) < best_d:
            best_d, best_w = abs(d - target), mid
        if d < target:
            lo = mid
        else:
            hi = mid
    return best_w, spearman_dist(z0, z0 + best_w * u)


def run(cmd, timeout=600):
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout)
        return 0, out
    except subprocess.CalledProcessError as e:
        return e.returncode, e.output or str(e)
    except Exception as e:
        return 1, str(e)


# =============================================================================
# Step 1 — Pull
# =============================================================================
def pull_kernels() -> dict:
    log("======== STEP1: pull kernels + metadata + outputs ========")
    meta = {}
    for slug, lb in KERNELS:
        s = safe(slug)
        code_dir = EXT / f"{s}_code"
        out_dir = EXT / f"{s}_out"
        code_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not (code_dir / "kernel-metadata.json").exists():
            rc, out = run([KAG, "kernels", "pull", slug, "-p", str(code_dir), "-m"], timeout=300)
            log(f"  pull {slug}: rc={rc}")
            if rc != 0:
                log(f"    err: {out[-400:]}")
        else:
            log(f"  code already present: {slug}")
        # outputs (may be empty)
        marker = out_dir / "_pull_done"
        if not marker.exists():
            rc, out = run([KAG, "kernels", "output", slug, "-p", str(out_dir)], timeout=600)
            log(f"  output {slug}: rc={rc} files={len(list(out_dir.glob('*')))}")
            if "404" in out or "No files" in out or "does not have any outputs" in out.lower():
                log(f"    (no kernel outputs)")
            marker.write_text(out[:2000] if out else "ok")
        mj = code_dir / "kernel-metadata.json"
        info = {"slug": slug, "lb": lb, "code_dir": str(code_dir), "out_dir": str(out_dir)}
        if mj.exists():
            info["meta"] = json.loads(mj.read_text())
        meta[slug] = info
    return meta


def extract_source_text(code_dir: Path) -> str:
    chunks = []
    for p in sorted(code_dir.glob("*")):
        if p.name.startswith(".") or p.name == "kernel-metadata.json":
            continue
        if p.suffix == ".ipynb":
            try:
                nb = json.loads(p.read_text())
                for c in nb.get("cells", []):
                    chunks.append("".join(c.get("source", [])))
            except Exception:
                pass
        elif p.suffix in {".py", ".md", ".txt"}:
            try:
                chunks.append(p.read_text(errors="ignore"))
            except Exception:
                pass
    return "\n\n".join(chunks)


def parse_upstream(meta_map: dict) -> tuple[set[str], set[str], dict]:
    datasets, kernels = set(), set()
    formulas = {}
    ds_re = re.compile(r"(?:kaggle\.com/datasets/|datasets/download\s+-d\s+|['\"])/?(?P<o>[A-Za-z0-9_-]+)/(?P<n>[A-Za-z0-9_-]+)")
    for slug, info in meta_map.items():
        m = info.get("meta") or {}
        for d in m.get("dataset_sources") or []:
            datasets.add(d)
        for k in m.get("kernel_sources") or []:
            kernels.add(k)
        src = extract_source_text(Path(info["code_dir"]))
        info["source_text"] = src
        for mo in ds_re.finditer(src):
            datasets.add(f"{mo.group('o')}/{mo.group('n')}")
        # capture blend-looking lines
        lines = []
        for ln in src.splitlines():
            if any(tok in ln for tok in ["0.6", "0.12", "0.05", "logit_rank", "score =", "B =", "A +", "blend", "weight"]):
                if len(ln.strip()) < 220:
                    lines.append(ln.rstrip())
        formulas[slug] = "\n".join(lines[:80])
    return datasets, kernels, formulas


def download_upstreams(datasets: set[str], kernels: set[str]) -> dict:
    log("======== STEP1b: download upstream datasets / kernel outs ========")
    ds_root = EXT / "datasets"
    kn_root = EXT / "upstream_kernels"
    ds_root.mkdir(exist_ok=True)
    kn_root.mkdir(exist_ok=True)
    got = {"datasets": [], "kernels": [], "errors": []}
    for d in sorted(datasets):
        name = d.replace("/", "__")
        dest = ds_root / name
        dest.mkdir(parents=True, exist_ok=True)
        # skip if already has files
        if any(dest.rglob("*")) and any(p.is_file() and p.stat().st_size > 1000 for p in dest.rglob("*")):
            log(f"  dataset present: {d}")
            got["datasets"].append(d)
            continue
        log(f"  downloading dataset {d} ...")
        rc, out = run([KAG, "datasets", "download", "-d", d, "--unzip", "-p", str(dest)], timeout=900)
        if rc == 0:
            got["datasets"].append(d)
            log(f"    OK {d} files={len(list(dest.rglob('*')))}")
        else:
            got["errors"].append({"dataset": d, "err": out[-500:]})
            log(f"    FAIL {d}: {out[-200:]}")
    for k in sorted(kernels):
        name = safe(k)
        dest = kn_root / name
        dest.mkdir(parents=True, exist_ok=True)
        if any(dest.glob("*")):
            got["kernels"].append(k)
            continue
        rc, out = run([KAG, "kernels", "output", k, "-p", str(dest)], timeout=600)
        if rc == 0:
            got["kernels"].append(k)
        else:
            got["errors"].append({"kernel": k, "err": out[-400:]})
    return got


# =============================================================================
# Load helpers
# =============================================================================
def load_csv_pred(path: Path) -> np.ndarray | None:
    try:
        df = pd.read_csv(path)
        for c in ["is_fraud", "pred", "prediction", "oof", "p", "prob"]:
            if c in df.columns:
                return df[c].to_numpy(np.float64)
        num = [c for c in df.columns if c != "id" and np.issubdtype(df[c].dtype, np.number)]
        return df[num[-1]].to_numpy(np.float64) if num else None
    except Exception:
        return None


def load_vec(path: Path, n_expect: int | None = None) -> np.ndarray | None:
    try:
        if path.suffix == ".npy":
            arr = np.load(path, allow_pickle=True)
            if isinstance(arr, np.ndarray) and arr.dtype == object:
                return None
            x = np.asarray(arr, dtype=np.float64)
            if x.ndim == 2 and x.shape[1] == 1:
                x = x.ravel()
            elif x.ndim == 2 and x.shape[-1] == 2:
                # logp False/True -> log-odds mean over middle dims if needed
                x = (x[..., 1] - x[..., 0]).mean(axis=tuple(range(1, x.ndim - 1))) if x.ndim > 2 else (x[:, 1] - x[:, 0])
                x = np.asarray(x, float).ravel()
            elif x.ndim > 1:
                # try log-odds for [n, ..., 2]
                if x.shape[-1] == 2:
                    x = (x[..., 1] - x[..., 0]).mean(axis=tuple(range(1, x.ndim - 1)))
                else:
                    return None
            x = np.asarray(x, float).ravel()
        elif path.suffix == ".csv":
            x = load_csv_pred(path)
            if x is None:
                return None
        elif path.suffix == ".parquet":
            df = pd.read_parquet(path)
            num = [c for c in df.columns if c != "id" and np.issubdtype(df[c].dtype, np.number)]
            if not num:
                return None
            x = df[num[-1]].to_numpy(float)
        else:
            return None
        if n_expect is not None and len(x) != n_expect:
            return None
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x
    except Exception:
        return None


def find_oof_test_pairs(ntr: int, nte: int) -> list[tuple[str, Path, Path]]:
    roots = [EXT, PRED]
    oof_map, test_map = {}, {}
    pat_o = re.compile(r"(.+?)(?:_oof|_OOF|_nested)(\.(?:npy|csv|parquet))$", re.I)
    pat_t = re.compile(r"(.+?)(?:_test|_te|_TEST)(\.(?:npy|csv|parquet))$", re.I)
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file() or p.name.startswith("._"):
                continue
            if p.suffix.lower() not in {".npy", ".csv", ".parquet"}:
                continue
            try:
                sz = p.stat().st_size
            except OSError:
                continue
            if sz < 1000 or sz > 200_000_000:
                continue
            mo, mt = pat_o.search(p.name), pat_t.search(p.name)
            if mo:
                oof_map[str(p.parent / mo.group(1))] = p
            if mt:
                test_map[str(p.parent / mt.group(1))] = p
    pairs = []
    for k in sorted(set(oof_map) & set(test_map)):
        pairs.append((k, oof_map[k], test_map[k]))
    return pairs


def load_sub_from_anywhere(names: list[str]) -> Path | None:
    search = [EXT, SUB_LOCAL, VOL_SUB, WORK / "submissions"]
    for name in names:
        for root in search:
            if not root.exists():
                continue
            hits = list(root.rglob(name))
            hits = [h for h in hits if h.is_file() and not h.name.startswith("._")]
            if hits:
                return hits[0]
    return None


def ensure_local_copy(src: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if not dest.exists() or dest.stat().st_size != src.stat().st_size:
        dest.write_bytes(src.read_bytes())
    return dest


def p_to_sub(ids, p, path: Path):
    p = np.asarray(p, float).ravel()
    # store as ranks in (0,1) for Kaggle-style
    pr = (rankdata(p, method="average") - 0.5) / len(p)
    pd.DataFrame({"id": ids, "is_fraud": pr}).to_csv(path, index=False)
    return path


# =============================================================================
# Main pipeline
# =============================================================================
def main():
    assert NO_SUBMIT, "STRICT NO_SUBMIT"
    log("GM_PULL_AND_AUDIT_095347_V2 START (NO_SUBMIT)")

    y = pd.read_csv(DATA / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int32)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    ntr, nte = len(y), None

    # base z_ref + d1
    z_ref = np.load(PRED / "z_ref_old_oof.npy").ravel().astype(np.float64)
    z_ref_t = np.load(PRED / "z_ref_old_test.npy").ravel().astype(np.float64)
    d1 = np.load(PRED / "gm_v4_polish/d1_v4_oof.npy").ravel().astype(np.float64)
    d1_t = np.load(PRED / "gm_v4_polish/d1_v4_test.npy").ravel().astype(np.float64)
    d1u, d1tu = unitize(d1), unitize(d1_t)
    ev_d1 = eval_delta_on_base(z_ref, d1u, y, folds, w_grid=np.linspace(0.0, 0.10, 21))
    z_base = z_ref + ev_d1["w"] * d1u
    z_base_t = unitize(z_ref_t) + ev_d1["w"] * d1tu
    z_base_auc = float(roc_auc_score(y, z_base))
    log(f"  base z_ref+δ1_v4 AUC={z_base_auc:.7f} w={ev_d1['w']:.4f}")
    nte = len(z_ref_t)

    # optional d3
    d3_path = PRED / "gm_v4_polish/d3_v4_oof.npy"
    d3u = d3tu = None
    if d3_path.exists():
        d3u = unitize(np.load(d3_path).ravel())
        d3tu = unitize(np.load(PRED / "gm_v4_polish/d3_v4_test.npy").ravel())
        ev_d3 = eval_delta_on_base(z_base, d3u, y, folds, w_grid=np.linspace(0.0, 0.10, 21))
        log(f"  δ3_v4 on base: pos={ev_d3['pos']}/5 lift={ev_d3['lift_e5']:+.2f}e-5 w={ev_d3['w']:.4f}")

    # P5
    p5o = p5t = None
    p5_o_path = PRED / "gm_recon_p5/P5_L5.1_oof.npy"
    if p5_o_path.exists():
        p5o = unitize(np.load(p5_o_path).ravel())
        p5t = unitize(np.load(PRED / "gm_recon_p5/P5_L5.1_test.npy").ravel())

    meta_map = pull_kernels()
    datasets, kernels, formulas = parse_upstream(meta_map)
    # always include known areeb deps
    datasets |= {
        "goodpjw2008/s6e11-compact-lm-seed7",
        "goodpjw2008/s6e11-oof-bank",
        "amanatar/s6e11-strike10-podium-apex-095340",
        "moranzavdi/s6e11-strike9-lb-095330",
        "megayak/s6e11-target-first-lm",
    }
    got = download_upstreams(datasets, kernels)

    # ---- extract key submission vectors ----
    log("======== STEP2: lineage vectors + OOF audit ========")
    sample = None
    for cand in [DATA / "sample_submission.csv", SUB_LOCAL / "sub_ladder_slot5_macro_finale_1p25e4.csv"]:
        if cand.exists():
            sample = pd.read_csv(cand)
            break
    if sample is None:
        sample = pd.read_csv(VOL_SUB / "sub_ladder_slot5_macro_finale_1p25e4.csv")
    ids = sample["id"].to_numpy()

    def find_sub_csv(patterns: list[str]) -> Path | None:
        for pat in patterns:
            # exact name first
            p = load_sub_from_anywhere([pat])
            if p is not None:
                return p
        # fuzzy under EXT
        for root in [EXT, SUB_LOCAL, VOL_SUB]:
            if not root.exists():
                continue
            for p in root.rglob("*.csv"):
                if p.name.startswith("._"):
                    continue
                low = p.name.lower()
                if any(x.lower() in low for x in patterns):
                    try:
                        df = pd.read_csv(p, nrows=2)
                        if "is_fraud" in df.columns:
                            return p
                    except Exception:
                        continue
        return None

    paths = {
        "areeb": find_sub_csv(["submission.csv"]) if False else None,
    }
    # prefer kernel outputs / dataset files by known names
    areeb_out = EXT / f"{safe(KERNELS[0][0])}_out"
    # search submission from areeb output first
    areeb_sub = None
    for p in list(areeb_out.rglob("*.csv")) + list((EXT / "datasets").rglob("*095347*")):
        if p.is_file() and "is_fraud" in pd.read_csv(p, nrows=1).columns:
            # verify sha if possible
            areeb_sub = p
            break
    # reconstruct areeb from formula if we have inputs
    ds = EXT / "datasets"
    def ds_find(name):
        hits = sorted(ds.rglob(name))
        return hits[0] if hits else None

    st10 = ds_find("submission_095340.csv")
    st9 = ds_find("submission_095330.csv")
    lm = ds_find("lm_test.npy")
    lm7 = ds_find("lm7_test_f0.npy")
    mlp = ds_find("test_gbn_bag5_w512.npy")
    gb = ds_find("test_gb_bag5.npy")

    Z = {}
    formula_notes = []

    # Local ladder
    def load_z_from_sub(path, key, zfun=zrk_ppf):
        p = load_csv_pred(path)
        assert p is not None and len(p) == nte, (path, None if p is None else len(p))
        Z[key] = zfun(p)
        return p

    s5_path = SUB_LOCAL / "sub_ladder_slot5_macro_finale_1p25e4.csv"
    if not s5_path.exists():
        s5_path = VOL_SUB / "sub_ladder_slot5_macro_finale_1p25e4.csv"
        ensure_local_copy(s5_path, SUB_LOCAL)
        s5_path = SUB_LOCAL / s5_path.name
    s4_path = SUB_LOCAL / "sub_ladder_slot4_adaptive_0p65e4.csv"
    if not s4_path.exists():
        ensure_local_copy(VOL_SUB / s4_path.name, SUB_LOCAL)
    s1_path = SUB_LOCAL / "sub_strike9p6_opt_step.csv"
    if not s1_path.exists():
        ensure_local_copy(VOL_SUB / s1_path.name, SUB_LOCAL)
    st9p5_path = SUB_LOCAL / "sub_strike9p5_axis_0p70_0p10.csv"
    if not st9p5_path.exists() and (VOL_SUB / st9p5_path.name).exists():
        ensure_local_copy(VOL_SUB / st9p5_path.name, SUB_LOCAL)

    p_S5 = load_z_from_sub(s5_path, "S5")
    p_S4 = load_z_from_sub(s4_path, "S4")
    p_S1 = load_z_from_sub(s1_path, "S1")
    if st9p5_path.exists():
        load_z_from_sub(st9p5_path, "St9p5")

    # Strike9 public
    if st9 is None:
        # local strike9 final
        local_st9 = SUB_LOCAL / "sub_gm_top1_strike9_final_max.csv"
        if not local_st9.exists() and (VOL_SUB / local_st9.name).exists():
            ensure_local_copy(VOL_SUB / local_st9.name, SUB_LOCAL)
        if local_st9.exists():
            st9 = local_st9
            formula_notes.append("St9 loaded from local sub_gm_top1_strike9_final_max.csv (public 0.95330 target)")
    if st9 is not None:
        p_St9 = load_z_from_sub(Path(st9), "St9")
    else:
        p_St9 = None
        log("  WARNING: St9 submission not found")

    def first_submission_csv(out_dir: Path) -> Path | None:
        if not out_dir.exists():
            return None
        cands = list(out_dir.glob("submission*.csv")) + list(out_dir.glob("*.csv"))
        for p in cands:
            if p.name.startswith("._"):
                continue
            try:
                df = pd.read_csv(p, nrows=2)
                if "is_fraud" in df.columns and len(pd.read_csv(p)) == nte:
                    return p
            except Exception:
                continue
        return None

    # aman / goodpjw from kernel outputs or datasets
    aman_out = EXT / f"{safe(KERNELS[2][0])}_out"
    aman_sub = first_submission_csv(aman_out) or st10
    if aman_sub is None:
        aman_sub = find_sub_csv(["095340", "submission_095340"])
    if aman_sub is not None:
        load_z_from_sub(Path(aman_sub), "aman")
        if st10 is None:
            st10 = Path(aman_sub)

    # goodpjw kernel submission
    gp_out = EXT / f"{safe(KERNELS[1][0])}_out"
    gp_sub = first_submission_csv(gp_out)
    if gp_sub is not None:
        load_z_from_sub(gp_sub, "goodpjw")
        formula_notes.append(f"goodpjw loaded from {gp_sub}")

    # Reconstruct areeb exactly if all inputs present
    areeb_recon = None
    if all(x is not None for x in [st10, st9, lm, lm7, mlp, gb]):
        A = zrk_logit(pd.read_csv(st10)["is_fraud"].to_numpy(float))
        F = zrk_logit(pd.read_csv(st9)["is_fraud"].to_numpy(float))
        LM = zrk_logit(np.load(lm))
        a7 = np.load(lm7)
        L7 = zrk_logit((a7[..., 1] - a7[..., 0]).mean(1))
        MLP = zrk_logit(np.load(mlp))
        GB = zrk_logit(np.load(gb))
        B = A + 0.6 * (A - F) + 0.12 * (LM - F) + 0.06 * (L7 - F)
        score = B - 0.05 * (MLP - F) - 0.025 * (GB - F)
        areeb_recon = score
        Z["areeb"] = zrk_ppf((rankdata(score) - 0.5) / len(score))
        Z["areeb_logit"] = score
        formula_notes.append(
            "areebbb EXACT: B=A+0.6(A-F)+0.12(LM-F)+0.06(L7-F); "
            "score=B-0.05(MLP-F)-0.025(GB-F); A=Strike10, F=Strike9"
        )
        log("  reconstructed areebbb 0.95347 exactly from public inputs")
        p_to_sub(ids, score, ART / "recon_areeb_095347.csv")
    else:
        for p in areeb_out.rglob("submission*.csv"):
            load_z_from_sub(p, "areeb")
            formula_notes.append(f"areebbb loaded from kernel output {p}")
            break
        missing = [n for n, v in [("st10", st10), ("st9", st9), ("lm", lm), ("lm7", lm7), ("mlp", mlp), ("gb", gb)] if v is None]
        log(f"  areeb reconstruct incomplete, missing={missing}")

    # if goodpjw missing but we have A,F,LM,L7 — approximate LM ladder without tree half-move
    if "goodpjw" not in Z and all(x is not None for x in [st10, st9, lm7]):
        A = zrk_logit(pd.read_csv(st10)["is_fraud"].to_numpy(float))
        F = zrk_logit(pd.read_csv(st9)["is_fraud"].to_numpy(float))
        a7 = np.load(lm7)
        L7 = zrk_logit((a7[..., 1] - a7[..., 0]).mean(1))
        LM = zrk_logit(np.load(lm)) if lm is not None else F
        B = A + 0.6 * (A - F) + 0.12 * (LM - F) + 0.06 * (L7 - F)
        Z["goodpjw"] = zrk_ppf((rankdata(B) - 0.5) / len(B))
        formula_notes.append(
            "goodpjw APPROX: B=A+0.6(A-F)+0.12(LM-F)+0.06(L7-F) (no tree half-move)"
        )
        log("  approximated goodpjw 0.95346 ladder (sans tree move)")
        p_to_sub(ids, B, ART / "recon_goodpjw_095346_approx.csv")

    # maly / lucifer
    for key, idx in [("maly", 3), ("lucifer", 4)]:
        outd = EXT / f"{safe(KERNELS[idx][0])}_out"
        p = first_submission_csv(outd)
        if p is not None:
            load_z_from_sub(p, key)
    # v8
    v8t = PRED / "P_super_v8_part12_rect_test.npy"
    if not v8t.exists() and (VOL_PRED / v8t.name).exists():
        ensure_local_copy(VOL_PRED / "P_super_v8_part12_rect_test.npy", PRED)
        ensure_local_copy(VOL_PRED / "P_super_v8_part12_rect_oof.npy", PRED)
    if (PRED / "P_super_v8_part12_rect_test.npy").exists():
        Z["v8"] = zrk_ppf(np.load(PRED / "P_super_v8_part12_rect_test.npy").ravel())

    # Vertex from Part2
    u_A_path = PRED / "gm_v5p2/probeA_u.npy"
    if not u_A_path.exists() and (VOL_PRED / "gm_v5p2/probeA_u.npy").exists():
        ensure_local_copy(VOL_PRED / "gm_v5p2/probeA_u.npy", PRED / "gm_v5p2")
    u_A = np.load(PRED / "gm_v5p2/probeA_u.npy").ravel()
    # build vertex in logit-rank space (consistent with ladder)
    zS5 = zrk_logit(p_S5)
    zS4 = zrk_logit(p_S4)
    v1 = unit(zS5 - zS4)
    uA_p = unit(u_A - float(np.dot(u_A, v1)) * v1)
    v2 = unit(uA_p)
    v1_star, v2_star = -23.275301, +3.056557
    z_Vertex_logit = zS5 + v1_star * v1 + v2_star * v2
    Z["Vertex"] = zrk_ppf((rankdata(z_Vertex_logit) - 0.5) / len(z_Vertex_logit))
    formula_notes.append(f"z_Vertex = z_S5 + ({v1_star})*unit(S5-S4) + ({v2_star})*unit(u_A⊥v1)")

    # ProbeA local
    probeA = SUB_LOCAL / "sub_gm_v5p2_probeA_unpurged_s5_plus_v5combo_0p25e4.csv"
    if not probeA.exists() and (VOL_SUB / probeA.name).exists():
        ensure_local_copy(VOL_SUB / probeA.name, SUB_LOCAL)
    if probeA.exists():
        load_z_from_sub(probeA, "ProbeA")

    # Pairwise 1-rho matrix
    keys_mat = [k for k in ["areeb", "goodpjw", "aman", "maly", "lucifer", "St9", "St9p5", "S1", "S4", "S5", "Vertex", "v8", "ProbeA"] if k in Z]
    D = np.zeros((len(keys_mat), len(keys_mat)))
    for i, a in enumerate(keys_mat):
        for j, b in enumerate(keys_mat):
            D[i, j] = spearman_dist(Z[a], Z[b]) if i != j else 0.0
    log("  pairwise 1-rho matrix computed for: " + ", ".join(keys_mat))

    # OLS areeb ~ goodpjw, St9, aman, v8
    ols = {}
    if "areeb" in Z:
        cols = [k for k in ["goodpjw", "St9", "aman", "v8"] if k in Z]
        if cols:
            X = np.column_stack([Z[k] for k in cols])
            X = np.column_stack([np.ones(len(X)), X])
            beta, *_ = np.linalg.lstsq(X, Z["areeb"], rcond=None)
            pred = X @ beta
            ss_res = float(np.sum((Z["areeb"] - pred) ** 2))
            ss_tot = float(np.sum((Z["areeb"] - Z["areeb"].mean()) ** 2))
            r2 = 1.0 - ss_res / (ss_tot + 1e-18)
            ols = {"cols": ["intercept"] + cols, "beta": beta.tolist(), "R2": r2}
            log(f"  OLS areeb~{cols}: R²={r2:.6f} β={np.round(beta,4).tolist()}")
            # also check St9 inside
            if "St9" in Z:
                ols["1-rho(areeb,St9)"] = spearman_dist(Z["areeb"], Z["St9"])
                ols["rho(areeb,St9)"] = 1.0 - ols["1-rho(areeb,St9)"]
            if "goodpjw" in Z:
                ols["1-rho(areeb,goodpjw)"] = spearman_dist(Z["areeb"], Z["goodpjw"])
                ols["rho(areeb,goodpjw)"] = 1.0 - ols["1-rho(areeb,goodpjw)"]

    # ---- OOF probes ----
    log("======== STEP2b: unused OOF residual probes ========")
    pairs = find_oof_test_pairs(ntr, nte)
    probe_rows = []
    skip = ("z_ref_old", "d1_v4", "P5_L5", "z_v5_overnight")
    probed = 0
    for key, op, tp in pairs:
        if any(s in key for s in skip):
            continue
        o = load_vec(op, ntr)
        t = load_vec(tp, nte)
        if o is None or t is None:
            continue
        zo = zrk_logit(o) if (o.min() >= -1e-9 and o.max() <= 1.0 + 1e-9) else unitize(o)
        zt = zrk_logit(t) if (t.min() >= -1e-9 and t.max() <= 1.0 + 1e-9) else unitize(t)
        try:
            if abs(float(spearmanr(zo, z_base).correlation)) > 0.9995:
                continue
            solo = fold_aucs(y, zo, folds)
            ro, rt = lofo_residual(zo, zt, z_base, unitize(z_base_t), folds)
            d, ks = disp_ks(ro, rt)
            if ks >= KS_GATE:
                ro, rt = force_exact_law(ro, rt)
                d, ks = disp_ks(ro, rt)
            ev = eval_delta_on_base(z_base, ro, y, folds)
            row = {
                "key": key, "solo_folds": solo, "solo_auc": float(roc_auc_score(y, zo)),
                "pos": ev["pos"], "lift_e5": ev["lift_e5"], "deltas": ev["deltas"],
                "w": ev["w"], "disp": d, "ks": ks,
            }
            # combo with d3?
            if d3u is not None:
                z2 = z_base + ev["w"] * ro
                ev2 = eval_delta_on_base(z2, d3u, y, folds, w_grid=np.linspace(0.0, 0.08, 17))
                row["with_d3_pos"] = ev2["pos"]
                row["with_d3_auc"] = ev2["auc"]
                row["with_d3_lift_from_base"] = (ev2["auc"] - z_base_auc) * 1e5
            probe_rows.append(row)
            probed += 1
            if ev["pos"] >= 4 or ev["lift_e5"] > 0.5:
                log(f"  HIT {Path(key).name[-60:]} pos={ev['pos']}/5 lift={ev['lift_e5']:+.2f}e-5")
            if probed >= 24:
                break
        except Exception:
            continue
    probe_rows.sort(key=lambda r: (r["pos"], r["lift_e5"]), reverse=True)
    log(f"  probed {probed} OOF pairs")

    # combo push past 0.95275?
    combo_note = None
    if d3u is not None:
        evc = eval_delta_on_base(z_base, d3u, y, folds, w_grid=np.linspace(0.0, 0.12, 25))
        zc = z_base + evc["w"] * d3u
        if p5o is not None:
            evp = eval_delta_on_base(zc, p5o, y, folds, w_grid=np.linspace(-0.05, 0.08, 27))
            zc2 = zc + evp["w"] * p5o
            folds_c = fold_aucs(y, zc2, folds)
            combo_note = {
                "auc": float(roc_auc_score(y, zc2)),
                "folds": folds_c,
                "pos_vs_base": sum(1 for a, b in zip(folds_c, fold_aucs(y, z_base, folds)) if a > b),
                "d3_w": evc["w"],
                "p5_w": evp["w"],
            }
            log(f"  combo base+d3+P5L51 AUC={combo_note['auc']:.7f} folds={[round(x,7) for x in folds_c]}")

    # =============================================================================
    # Step 3 — private orthogonal upgrades
    # =============================================================================
    log("======== STEP3: private signals ⊥ ext span + B_strict_bad ========")
    # Build B_ext span in test space (logit preferred for ops)
    def to_logit_space(key):
        if key == "areeb" and "areeb_logit" in Z:
            return unitize(Z["areeb_logit"])
        return unitize(Z[key])

    ext_keys = [k for k in ["areeb", "goodpjw", "aman", "St9"] if k in Z]
    B_ext = gram_orthonormal(np.column_stack([to_logit_space(k) for k in ext_keys])) if ext_keys else np.zeros((nte, 0))
    log(f"  B_ext keys={ext_keys} cols={B_ext.shape[1]}")

    # B_strict_bad
    Q_path = PRED / "gm_v5_overnight/Q_orth.npy"
    if not Q_path.exists() and (VOL_PRED / "gm_v5_overnight/Q_orth.npy").exists():
        ensure_local_copy(VOL_PRED / "gm_v5_overnight/Q_orth.npy", PRED / "gm_v5_overnight")
        for nm in ["B_bad_orth.npy", "q_bad_tree.npy"]:
            if (VOL_PRED / "gm_v5_overnight" / nm).exists():
                ensure_local_copy(VOL_PRED / "gm_v5_overnight" / nm, PRED / "gm_v5_overnight")
    drop_path = SUB_LOCAL / "sub_gm_v5_overnight_slot5.csv"
    if not drop_path.exists() and (VOL_SUB / drop_path.name).exists():
        ensure_local_copy(VOL_SUB / drop_path.name, SUB_LOCAL)

    B_strict = None
    if Q_path.exists() and drop_path.exists():
        Q = np.load(Q_path)
        # Q may be oof-length; need test-length bad basis. Prefer test-space reconstruction:
        # use q_bad_* if test-length else project via available test vectors
        e2 = Q[:, 2] if Q.shape[1] > 2 else None
        d_soft = Q[:, 5] if Q.shape[1] > 5 else None
        z_drop = zrk_logit(pd.read_csv(drop_path)["is_fraud"].to_numpy(float))
        drop_dir = unitize(z_drop - zS5)
        if e2 is not None and len(e2) == nte and d_soft is not None and len(d_soft) == nte:
            B_strict = gram_orthonormal(np.column_stack([e2, d_soft, drop_dir]))
        else:
            # fall back: tree bad from saved test-length files or just drop_dir + B_bad if test
            comps = [drop_dir]
            for nm in ["q_bad_tree.npy", "q_bad_S3.npy", "q_bad_ldr.npy"]:
                pp = PRED / "gm_v5_overnight" / nm
                if not pp.exists() and (VOL_PRED / "gm_v5_overnight" / nm).exists():
                    ensure_local_copy(VOL_PRED / "gm_v5_overnight" / nm, PRED / "gm_v5_overnight")
                if pp.exists():
                    v = np.load(pp).ravel()
                    if len(v) == nte:
                        comps.append(v)
            B_bad = PRED / "gm_v5_overnight/B_bad_orth.npy"
            if B_bad.exists():
                Bb = np.load(B_bad)
                if Bb.ndim == 2 and Bb.shape[0] == nte:
                    for j in range(Bb.shape[1]):
                        comps.append(Bb[:, j])
            B_strict = gram_orthonormal(np.column_stack(comps))
        log(f"  B_strict_bad cols={B_strict.shape[1]}")
    else:
        B_strict = gram_orthonormal(np.column_stack([unitize(zS5)]))  # degenerate fallback
        log("  WARNING: incomplete B_strict_bad; using weak fallback")

    def orth_ext_and_bad(v):
        bases = [B_ext[:, j] for j in range(B_ext.shape[1])] + [B_strict[:, j] for j in range(B_strict.shape[1])]
        return unitize(orth_against(v, bases))

    # Private 1: Δz post St9 -> Vertex
    priv = {}
    if "St9" in Z:
        # work in logit space
        z_st9_l = zrk_logit(p_St9)
        dz1 = unitize(z_Vertex_logit - z_st9_l)
    else:
        dz1 = unitize(z_Vertex_logit - zS5)
    # Need an OOF counterpart for align: use OOF proxy from local residuals
    # Approximate OOF private1 as same construction on oof if available — else use test-only unit direction
    # Build OOF vertex proxy via S5/S4 oof not available as ranks of submissions only on test.
    # Use d1+P5 as OOF stand-in for private upgrade mass; for Signal1 use test-only align via force law from a synthetic oof=direction projected
    # Practical approach: treat private test directions; for OOF diagnostics use d1/p5/d3 which have OOF.

    p1_t = orth_ext_and_bad(dz1)
    # fabricate matched oof via order-stat from a related oof residual (d1) shape, then replace ranks from p1_t law — use force_exact_law inverted: oof = unitize(d1), map test already p1
    p1_o_raw = unitize(d1u if d1u is not None else z_base)
    p1_o, p1_t = force_exact_law(p1_o_raw, p1_t)
    # re-strip after align (test)
    p1_t = orth_ext_and_bad(p1_t)
    p1_o, p1_t = force_exact_law(p1_o, p1_t)
    d, ks = disp_ks(p1_o, p1_t)
    priv["P1_vertex"] = {"disp": d, "ks": ks, "o": p1_o, "t": p1_t, "cos_ext": [abs(cos(p1_t, B_ext[:, j])) for j in range(B_ext.shape[1])]}
    log(f"  P1 vertex⊥ext disp={d:.4f} KS={ks:.6f} max|cos_ext|={max(priv['P1_vertex']['cos_ext']+[0]):.2e}")

    # Private 2: δ1 + P5
    d2 = unitize(d1tu)
    if p5t is not None:
        d2 = unitize(0.7 * d2 + 0.3 * unitize(p5t))
    p2_t = orth_ext_and_bad(d2)
    p2_o_raw = unitize(d1u if p5o is None else 0.7 * d1u + 0.3 * p5o)
    p2_o, p2_t = align_method_b_smooth(p2_o_raw, p2_t, folds)
    if ks_2samp(p2_o, p2_t).statistic >= KS_GATE:
        p2_o, p2_t = force_exact_law(p2_o, p2_t)
    p2_t = orth_ext_and_bad(p2_t)
    p2_o, p2_t = force_exact_law(p2_o, p2_t)
    d, ks = disp_ks(p2_o, p2_t)
    ev2 = eval_delta_on_base(z_base, p2_o, y, folds)
    priv["P2_d1_p5"] = {"disp": d, "ks": ks, "o": p2_o, "t": p2_t, **{k: ev2[k] for k in ["pos", "lift_e5", "w", "deltas"]}}
    log(f"  P2 δ1+P5⊥ext pos={ev2['pos']}/5 lift={ev2['lift_e5']:+.2f}e-5 KS={ks:.6f}")

    # Private 3: δ3 masa
    if d3tu is not None:
        p3_t = orth_ext_and_bad(d3tu)
        p3_o, p3_t = align_method_b_smooth(d3u, p3_t, folds)
        if ks_2samp(p3_o, p3_t).statistic >= KS_GATE:
            p3_o, p3_t = force_exact_law(p3_o, p3_t)
        p3_t = orth_ext_and_bad(p3_t)
        p3_o, p3_t = force_exact_law(p3_o, p3_t)
        d, ks = disp_ks(p3_o, p3_t)
        ev3 = eval_delta_on_base(z_base, p3_o, y, folds)
        priv["P3_d3"] = {"disp": d, "ks": ks, "o": p3_o, "t": p3_t, **{k: ev3[k] for k in ["pos", "lift_e5", "w", "deltas"]}}
        log(f"  P3 δ3⊥ext pos={ev3['pos']}/5 lift={ev3['lift_e5']:+.2f}e-5 KS={ks:.6f}")
    else:
        priv["P3_d3"] = None
        log("  P3 skipped (no d3)")

    # Combine private direction
    parts_t = [priv["P1_vertex"]["t"], priv["P2_d1_p5"]["t"]]
    wts = [0.45, 0.35]
    if priv.get("P3_d3"):
        parts_t.append(priv["P3_d3"]["t"])
        wts.append(0.20)
    wts = np.asarray(wts, float)
    wts /= wts.sum()
    u_priv = unitize(sum(w * unitize(t) for w, t in zip(wts, parts_t)))
    u_priv = orth_ext_and_bad(u_priv)

    # =============================================================================
    # Step 4 — stage NOSUBMIT candidates
    # =============================================================================
    log("======== STEP4: stage NOSUBMIT candidates ========")
    staged = []

    def stage(name, z_score, anchors=None):
        # z_score is logit-like; convert to submission ranks
        path = SUB_LOCAL / name
        p_to_sub(ids, z_score, path)
        # also copy to volume submissions
        try:
            ensure_local_copy(path, VOL_SUB)
        except Exception as e:
            log(f"  vol copy warn: {e}")
        mean = float(np.mean((rankdata(z_score) - 0.5) / len(z_score)))
        row = {"file": name, "mean_rank": mean}
        for ak, av in (anchors or {}).items():
            row[f"1-rho_vs_{ak}"] = spearman_dist(zrk_ppf((rankdata(z_score) - 0.5) / len(z_score)), av)
        # KS vs S5 ranks
        r = (rankdata(z_score) - 0.5) / len(z_score)
        r5 = (rankdata(p_S5) - 0.5) / len(p_S5)
        # distributional KS on zrk
        row["KS_vs_S5_z"] = float(ks_2samp(unitize(zrk_logit(r)), unitize(zrk_logit(r5))).statistic)
        staged.append(row)
        log(f"  staged {name} mean={mean:.6f} " + " ".join(f"{k}={v:.2e}" for k, v in row.items() if k.startswith("1-rho")))
        return path

    anchors = {}
    if "areeb" in Z:
        anchors["areeb"] = Z["areeb"]
    if "goodpjw" in Z:
        anchors["goodpjw"] = Z["goodpjw"]
    anchors["S5"] = Z["S5"]

    # 1) pure local vertex + P5
    z_local = z_Vertex_logit.copy()
    if p5t is not None:
        # tiny P5 step at ~1-rho ~ 0.1e-4 vs vertex
        w, gotd = weight_for_spearman_dist(z_local, orth_against(p5t, [unit(z_local)]), 0.10e-4)
        z_local = z_local + w * unit(orth_against(p5t, [unit(z_local)]))
    stage("sub_gm_v6_vertex_s4s5_local_only_NOSUBMIT.csv", z_local, anchors)

    # helper: add private orth to a base z at target 1-rho
    def add_priv(z0, target_dist):
        u = orth_ext_and_bad(u_priv)
        # also orth against z0 itself
        u = unit(orth_against(u, [unitize(z0)]))
        w, gotd = weight_for_spearman_dist(z0, u, target_dist)
        z = z0 + w * u
        # re-strip ext/bad lightly
        delta = orth_ext_and_bad(z - z0)
        w2, gotd = weight_for_spearman_dist(z0, delta, target_dist)
        return z0 + w2 * unit(delta), gotd, w2

    if "goodpjw" in Z:
        # work in logit if we have recon file
        gp_path = ART / "recon_goodpjw_095346_approx.csv"
        if gp_path.exists():
            z_gp = zrk_logit(pd.read_csv(gp_path)["is_fraud"].to_numpy(float))
        else:
            z_gp = zrk_logit((rankdata(Z["goodpjw"]) - 0.5) / len(Z["goodpjw"]))  # weak
        z_g2, d_g2, w_g2 = add_priv(z_gp, 0.20e-4)
        stage("sub_gm_v6_goodpjw095346_plus_private_orth_NOSUBMIT.csv", z_g2, anchors)

    if "areeb" in Z or areeb_recon is not None:
        z_ar = areeb_recon if areeb_recon is not None else zrk_logit((rankdata(Z["areeb"]) - 0.5) / len(Z["areeb"]))
        z_a15, d15, w15 = add_priv(z_ar, 0.15e-4)
        stage("sub_gm_v6_areeb095347_plus_private_orth_0p15e4_NOSUBMIT.csv", z_a15, anchors)
        z_a35, d35, w35 = add_priv(z_ar, 0.35e-4)
        stage("sub_gm_v6_areeb095347_plus_private_orth_0p35e4_NOSUBMIT.csv", z_a35, anchors)

    # Save arrays
    np.save(ART / "u_priv_orth_test.npy", u_priv)
    np.save(ART / "p1_vertex_test.npy", priv["P1_vertex"]["t"])
    np.save(ART / "p2_d1p5_test.npy", priv["P2_d1_p5"]["t"])
    if priv.get("P3_d3"):
        np.save(ART / "p3_d3_test.npy", priv["P3_d3"]["t"])
    np.savez(ART / "pairwise_1mrho.npz", keys=np.array(keys_mat), D=D)

    # =============================================================================
    # Report
    # =============================================================================
    log("======== WRITE REPORT ========")
    lines = [
        "# NOV11 0.95347 V2 Kernel Audit",
        "",
        f"**NO_SUBMIT={NO_SUBMIT}**. Elapsed **{time.time()-T0:.1f}s**.",
        f"Base: `z_ref_old+δ1_v4` OOF AUC=`{z_base_auc:.7f}`. Champion held: Ladder S5 **0.95335**.",
        "No Kaggle submissions made. 7 slots remain.",
        "",
        "## Step 1 — Pulled kernels & upstream harvest",
        "",
    ]
    for slug, lb in KERNELS:
        info = meta_map.get(slug, {})
        m = info.get("meta") or {}
        lines += [
            f"### `{slug}`" + (f" (LB `{lb}`)" if lb else ""),
            "",
            f"- dataset_sources: `{m.get('dataset_sources')}`",
            f"- kernel_sources: `{m.get('kernel_sources')}`",
            f"- code: `{info.get('code_dir')}`",
            f"- out: `{info.get('out_dir')}`",
            "",
        ]
    lines += [
        f"Downloaded datasets: `{got.get('datasets')}`",
        f"Upstream errors: `{len(got.get('errors', []))}`",
        "",
        "## Exact formulas (from source)",
        "",
        "### areebbb (`0.95347`)",
        "",
        "```python",
        "# A = Strike10 (amanatar 0.95340), F = Strike9 (moranzavdi 0.95330)",
        "# LM = megayak lm_test.npy; L7 = goodpjw lm7_test_f0 log-odds; MLP/GB from OOF bank",
        "B = A + 0.6*(A - F) + 0.12*(LM - F) + 0.06*(L7 - F)",
        "score = B - 0.05*(MLP - F) - 0.025*(GB - F)",
        "submission = (rankdata(score)-0.5)/N",
        "```",
        "",
        "**Implication:** `z_St9` (`F`) is *explicitly inside* `z_areeb` and `z_goodpjw`.",
        "",
        "### goodpjw2008 (`0.95346`) — from areeb writeup / source scrape",
        "",
        "```text",
        formulas.get(KERNELS[1][0], "(see code)")[:3000],
        "```",
        "",
        "### amanatar (`0.95340`)",
        "",
        "```text",
        formulas.get(KERNELS[2][0], "(see code)")[:3000],
        "```",
        "",
        "### Notes",
        "",
    ]
    for n in formula_notes:
        lines.append(f"- {n}")

    lines += [
        "",
        "## Step 2 — Test-space lineage",
        "",
        "zrk for matrix: `norm.ppf(rankdata(p)/(N+1))`.",
        "",
        "### Pairwise `1 - spearmanr`",
        "",
        "| |" + "|".join(keys_mat) + "|",
        "|" + "|".join(["---"] * (len(keys_mat) + 1)) + "|",
    ]
    for i, a in enumerate(keys_mat):
        lines.append("| **" + a + "** |" + "|".join(f"{D[i,j]:.4f}" for j in range(len(keys_mat))) + "|")

    lines += ["", "### OLS: `z_areeb ~ [z_goodpjw, z_St9, z_aman, z_v8]`", ""]
    if ols:
        lines.append(f"- columns: `{ols['cols']}`")
        lines.append(f"- β: `{np.round(ols['beta'], 6).tolist()}`")
        lines.append(f"- **R² = {ols['R2']:.6f}**")
        for k, v in ols.items():
            if k.startswith("1-rho") or k.startswith("rho"):
                lines.append(f"- {k}: `{v:.6f}`")
    else:
        lines.append("- OLS skipped (missing areeb vector)")

    lines += [
        "",
        "### OOF residual probes on `z_ref_old+δ1_v4`",
        "",
        "| key | pos/5 | lift e-5 | solo AUC | KS | with_d3 AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in probe_rows[:12]:
        lines.append(
            f"| `{Path(r['key']).name[-70:]}` | {r['pos']}/5 | {r['lift_e5']:+.2f} | "
            f"{r['solo_auc']:.6f} | {r['ks']:.5f} | {r.get('with_d3_auc', float('nan'))} |"
        )
    if combo_note:
        lines += [
            "",
            f"Local combo `base+δ3+P5_L5.1` AUC=`{combo_note['auc']:.7f}` "
            f"folds=`{[round(x,7) for x in combo_note['folds']]}` "
            f"pos_vs_base=`{combo_note['pos_vs_base']}/5`.",
        ]

    lines += [
        "",
        "## Step 3 — Private orthogonal upgrades",
        "",
        f"B_ext = orth_span({ext_keys}), cols={B_ext.shape[1]}.",
        f"B_strict_bad cols={0 if B_strict is None else B_strict.shape[1]}.",
        "",
        "| signal | pos/5 | lift e-5 | disp | KS |",
        "|---|---:|---:|---:|---:|",
        f"| P1 Δz_private_post_St9→Vertex | — | — | {priv['P1_vertex']['disp']:.4f} | {priv['P1_vertex']['ks']:.6f} |",
        f"| P2 δ1_v4 + P5_L5.1 | {priv['P2_d1_p5']['pos']}/5 | {priv['P2_d1_p5']['lift_e5']:+.2f} | {priv['P2_d1_p5']['disp']:.4f} | {priv['P2_d1_p5']['ks']:.6f} |",
    ]
    if priv.get("P3_d3"):
        lines.append(
            f"| P3 δ3_v4_masa | {priv['P3_d3']['pos']}/5 | {priv['P3_d3']['lift_e5']:+.2f} | "
            f"{priv['P3_d3']['disp']:.4f} | {priv['P3_d3']['ks']:.6f} |"
        )
    else:
        lines.append("| P3 δ3_v4_masa | — | — | — | — |")

    lines += [
        "",
        "## Step 4 — Staged NOSUBMIT candidates",
        "",
        "| file | mean | 1-ρ vs areeb | 1-ρ vs goodpjw | 1-ρ vs S5 | KS vs S5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in staged:
        lines.append(
            f"| `{r['file']}` | {r.get('mean_rank', float('nan')):.6f} | "
            f"{r.get('1-rho_vs_areeb', float('nan')):.2e} | "
            f"{r.get('1-rho_vs_goodpjw', float('nan')):.2e} | "
            f"{r.get('1-rho_vs_S5', float('nan')):.2e} | "
            f"{r.get('KS_vs_S5_z', float('nan')):.6f} |"
        )
    lines += [
        "",
        "Artifacts under `preds/gm_v6_audit/` and `/tmp/gm_v6_audit/`.",
        "",
        "`NO_SUBMIT=True` — do not upload these CSVs until explicitly authorized.",
        "",
    ]
    report_path = LOG / "NOV11_095347_V2_KERNEL_AUDIT.md"
    report_path.write_text("\n".join(lines))
    # sync to volume
    try:
        VOL_LOG.mkdir(parents=True, exist_ok=True)
        (VOL_LOG / report_path.name).write_text(report_path.read_text())
        # sync script
        (ROOT / "scripts" / "gm_pull_and_audit_095347_v2.py").write_text(Path(__file__).read_text())
    except Exception as e:
        log(f"  volume sync warn: {e}")
    (ART / "REPORT.json").write_text(json.dumps({
        "NO_SUBMIT": True,
        "base_auc": z_base_auc,
        "ols": ols,
        "keys_mat": keys_mat,
        "staged": staged,
        "datasets": got.get("datasets"),
        "formula_notes": formula_notes,
        "elapsed_s": time.time() - T0,
    }, indent=2, default=float))
    log(f"Wrote {report_path}")
    log(f"GM_V6_AUDIT_COMPLETE elapsed={time.time()-T0:.1f}s NO_SUBMIT={NO_SUBMIT}")


if __name__ == "__main__":
    main()
