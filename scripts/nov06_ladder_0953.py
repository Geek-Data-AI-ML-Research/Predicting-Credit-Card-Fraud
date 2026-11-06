#!/usr/bin/env python3
"""Nov06 research + adaptive 4-slot ladder aiming for Public LB >= 0.953000."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
COMP = "playground-series-s6e11"
OUT_DIR = ROOT / "submissions" / "nov06_ladder"
LOG_PATH = ROOT / "logs" / "overnight" / "nov06_ladder_0953.json"
SAMPLE = ROOT / "data" / "sample_submission.csv"
EPS = 1e-7


def _kaggle_env():
    env = os.environ.copy()
    for k in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        env.pop(k, None)
    env["PATH"] = str(ROOT / ".venv_forensics" / "bin") + ":" + env.get("PATH", "")
    return env


def load_vec(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path).astype(np.float64).ravel()
    df = pd.read_csv(path)
    cols = [
        c
        for c in df.columns
        if c.lower() in ("is_fraud", "prediction", "pred", "target", "prob")
    ]
    col = cols[0] if cols else df.columns[1]
    return df[col].astype(np.float64).values


def rank01(x: np.ndarray) -> np.ndarray:
    return (rankdata(x.astype(np.float64)) - 1.0) / (len(x) - 1.0)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def orth_resid(v: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Rank-space residual of v orthogonal to base (both rank01)."""
    vr, br = rank01(v), rank01(base)
    vr = (vr - vr.mean()) / (vr.std() + EPS)
    br = (br - br.mean()) / (br.std() + EPS)
    coef = float(np.dot(vr, br) / (np.dot(br, br) + EPS))
    return vr - coef * br


def save_sub(pred: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(SAMPLE)
    assert len(pred) == len(sample)
    out = sample.copy()
    out["is_fraud"] = rank01(pred)
    out.to_csv(path, index=False)
    return path


def submit(path: Path, message: str) -> int | None:
    cmd = [
        "kaggle",
        "competitions",
        "submit",
        "-c",
        COMP,
        "-f",
        str(path),
        "-m",
        message[:480],
    ]
    print("SUBMIT:", path.name, "|", message[:120])
    r = subprocess.run(cmd, env=_kaggle_env(), capture_output=True, text=True)
    print(r.stdout)
    print(r.stderr)
    if r.returncode != 0:
        return None
    # parse newest matching submission ref
    time.sleep(3)
    return None


def list_subs(n: int = 15) -> list[dict]:
    r = subprocess.run(
        ["kaggle", "competitions", "submissions", "-c", COMP],
        env=_kaggle_env(),
        capture_output=True,
        text=True,
    )
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    # skip header lines until numeric ref
    rows = []
    for ln in lines:
        parts = ln.split()
        if not parts or not parts[0].isdigit():
            continue
        ref = int(parts[0])
        # publicScore near end
        score = None
        status = "UNKNOWN"
        for tok in parts:
            if tok.startswith("SubmissionStatus."):
                status = tok.split(".", 1)[1]
            try:
                f = float(tok)
                if 0.9 < f < 1.0:
                    score = f
            except ValueError:
                pass
        fname = parts[1] if len(parts) > 1 else ""
        rows.append({"ref": ref, "file": fname, "status": status, "public_lb": score, "raw": ln})
        if len(rows) >= n:
            break
    return rows


def poll_file(fname: str, timeout_s: int = 600) -> float | None:
    """Poll until submission for fname has a public score."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        rows = list_subs(20)
        for row in rows:
            if row["file"] == fname and row["public_lb"] is not None:
                print(f"POLLED {fname}: {row['public_lb']} (ref={row['ref']})")
                return float(row["public_lb"])
        print(f"waiting for {fname} ...")
        time.sleep(20)
    return None


def main():
    report: dict = {
        "date": "2026-11-06",
        "goal": 0.953000,
        "prior_best": 0.95242,
        "lb_top_at_research": 0.95274,
        "research": {},
        "ladder": [],
    }

    # ---- load bank ----
    bank = {
        "champ": load_vec(ROOT / "submissions/sub_live10_grand_logit_extrapolation.csv"),
        "live8": load_vec(ROOT / "submissions/sub_live8_logit_simplex_plus_ascent.csv"),
        "live2": load_vec(ROOT / "submissions/sub_live2_psuper_v4_top_fusion.csv"),
        "v4": load_vec(ROOT / "preds/P_super_v4_test.npy"),
        "v4m": load_vec(ROOT / "preds/P_super_v4_master_test.npy"),
        "v4p": load_vec(ROOT / "preds/P_super_v4p_test.npy"),
        "part9c": load_vec(ROOT / "submissions/sub_part9c_clean_super_p7gs_pb.csv"),
        "areeb": load_vec(
            ROOT / "external_bank/areebbb_95239_out/submission.csv"
        ),
        "areeb_mlp10": load_vec(
            ROOT
            / "external_bank/fresh_nov06/areebbb_s6e11-glm-boost-residual-mlp-pool-lb-0-95239_out/submission_member_p7gs_mlp10.csv"
        ),
        "aman": load_vec(ROOT / "external_bank/amanatar_sovereign/submission.csv"),
        "sohail": load_vec(
            ROOT
            / "external_bank/fresh/sohailkhanlml_messi-is-pessi-lb-0-95231_out/submission.csv"
        ),
        "quey_p8": load_vec(
            ROOT
            / "external_bank/fresh_nov06/queyrusi_s6e11-part-8-two-stacks-one-blend-lb-0-95235_out/submission.csv"
        ),
        "quey_cnn": load_vec(
            ROOT
            / "external_bank/fresh/queyrusi_two-stacks-digit-cnn-lb-0-95226_out/submission.csv"
        ),
        "part10": load_vec(
            ROOT
            / "external_bank/fresh_nov06/goodpjw2008_s6e11-part-10-test-bag-fusion-lb-0-95239_out/submission.csv"
        ),
        "part10_cv": load_vec(
            ROOT
            / "external_bank/fresh_nov06/goodpjw2008_s6e11-part-10-test-bag-fusion-lb-0-95239_out/submission_cv_best.csv"
        ),
        "nil": load_vec(ROOT / "external_bank/nil_probit_0952/submission.csv"),
        "sameer": load_vec(
            ROOT
            / "external_bank/fresh_nov06/sameershaikh27_s6e11-fraud-pooled-stack-of-public-oof-banks_out/submission.csv"
        ),
        "sameer_hill": load_vec(
            ROOT
            / "external_bank/fresh_nov06/sameershaikh27_s6e11-fraud-pooled-stack-of-public-oof-banks_out/submission_hill.csv"
        ),
        "masaya": load_vec(
            ROOT
            / "external_bank/fresh_ds_nov06/masayakawamata_s6e11-public-stack-v10-outputs/test.npy"
        ),
        "tabpack": load_vec(
            ROOT
            / "external_bank/fresh_ds_nov06/masayakawamata_s6e11-tabpack-cpu-starter-predictions/test_tabpack.csv"
        ),
        "nina_ds": load_vec(
            ROOT / "external_bank/fresh_ds_nov06/nina2025_ps-s6e11-03/0.95198.csv"
        ),
        "adamstra": load_vec(
            ROOT
            / "external_bank/fresh_nov06/adamstra_s6e11-top-10-tail-sharpened-probit-fusion_out/submission.csv"
        ),
        "maly_lab": load_vec(
            ROOT
            / "external_bank/fresh_nov06/malyshevdanil_fraud-signal-lab-gpt-2-and-glm-fusion_out/submission.csv"
        ),
    }

    champ = bank["champ"]
    N = len(champ)
    for k, v in bank.items():
        assert len(v) == N, (k, len(v), N)

    # research summary: spearman vs champ
    research_corr = {}
    for k, v in bank.items():
        research_corr[k] = float(np.corrcoef(rank01(champ), rank01(v))[0, 1])
    report["research"] = {
        "strategy": (
            "Public LB ceiling ~0.95274; our best 0.95242. Pure remix of "
            "V4/Part9/Amanatar plateaued. Nov06 ladder injects newer public peaks "
            "(Part10 0.95239, Queyrusi Part8 0.95235, Sohail 0.95231, Areeb 0.95239) "
            "plus orthogonal residuals (TabPack ρ≈0.93, Part10_cv/Masaya/Sameer/Nina)."
        ),
        "corr_vs_champ": research_corr,
        "slot_plan": {
            "1": "New multi-peak rank fusion: champ+part10+areeb+quey_p8+sohail",
            "2": "Orth residual inject: champ + TabPack/Part10_cv/Masaya/Sameer/Nina",
            "3": "Adaptive: ascend along best of Slot1/2 vs champ; else logit simplex of peaks",
            "4": "Hard push multi-vector extrapolation toward 0.953",
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    preds: dict[str, np.ndarray] = {}

    # ---- Slot 1: multi-peak fusion ----
    s1 = (
        0.32 * rank01(bank["champ"])
        + 0.20 * rank01(bank["part10"])
        + 0.18 * rank01(bank["areeb"])
        + 0.15 * rank01(bank["quey_p8"])
        + 0.15 * rank01(bank["sohail"])
    )
    paths["s1"] = save_sub(s1, OUT_DIR / "sub_n06_s1_multipeak_fusion.csv")
    preds["s1"] = s1

    # ---- Slot 2: orthogonal residual injection ----
    z = logit(rank01(bank["champ"]))
    z = z + 0.22 * orth_resid(bank["tabpack"], bank["champ"])
    z = z + 0.18 * orth_resid(bank["part10_cv"], bank["champ"])
    z = z + 0.14 * orth_resid(bank["masaya"], bank["champ"])
    z = z + 0.12 * orth_resid(bank["sameer_hill"], bank["champ"])
    z = z + 0.10 * orth_resid(bank["nina_ds"], bank["champ"])
    z = z + 0.08 * orth_resid(bank["nil"], bank["champ"])
    s2 = sigmoid(z)
    paths["s2"] = save_sub(s2, OUT_DIR / "sub_n06_s2_orth_residual_inject.csv")
    preds["s2"] = s2

    # Submit Slot1 + Slot2 first (independent), poll both
    submit(
        paths["s1"],
        "N06-S1: Multi-peak rank fusion 0.32champ+0.20Part10(0.95239)+0.18Areeb+0.15QueyP8(0.95235)+0.15Sohail(0.95231) aim>=0.953",
    )
    lb1 = poll_file(paths["s1"].name, timeout_s=720)
    report["ladder"].append(
        {
            "slot": 1,
            "file": str(paths["s1"].relative_to(ROOT)),
            "public_lb": lb1,
            "note": "multi-peak champ/part10/areeb/quey_p8/sohail",
        }
    )
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(report, indent=2))

    submit(
        paths["s2"],
        "N06-S2: Champ + orth residuals TabPack(ρ0.93)+Part10_cv+MasayaV10+SameerHill+Nina95198+Nil aim>=0.953",
    )
    lb2 = poll_file(paths["s2"].name, timeout_s=720)
    report["ladder"].append(
        {
            "slot": 2,
            "file": str(paths["s2"].relative_to(ROOT)),
            "public_lb": lb2,
            "note": "orth residual inject from low-corr public vectors",
        }
    )
    LOG_PATH.write_text(json.dumps(report, indent=2))

    best_lb = max(x for x in [0.95242, lb1, lb2] if x is not None)
    best_key = "champ"
    best_pred = bank["champ"]
    if lb1 is not None and lb1 >= best_lb - 1e-12:
        best_lb, best_key, best_pred = lb1, "s1", preds["s1"]
    if lb2 is not None and lb2 > best_lb:
        best_lb, best_key, best_pred = lb2, "s2", preds["s2"]

    # ---- Slot 3 adaptive ----
    # Prefer ascent along the better of S1/S2 vs champ; also try peak logit simplex
    if lb1 is not None and lb2 is not None and max(lb1, lb2) >= 0.95242:
        # positive gradient exists
        if lb1 >= lb2:
            axis = rank01(preds["s1"]) - rank01(bank["champ"])
            note = f"ascent along S1-champ (S1={lb1})"
            alpha = 0.35 if lb1 > 0.95242 else 0.25
        else:
            axis = rank01(preds["s2"]) - rank01(bank["champ"])
            note = f"ascent along S2-champ (S2={lb2})"
            alpha = 0.40 if lb2 > 0.95242 else 0.28
        s3 = rank01(best_pred) + alpha * axis
        # mild mix of areeb_mlp residual (slightly less corr member)
        s3 = s3 + 0.08 * orth_resid(bank["areeb_mlp10"], best_pred)
    else:
        # plateau or drop: logit simplex of newest peaks + champ
        note = "logit simplex champ/part10/areeb/quey/sohail/v4m (S1/S2 did not beat 0.95242)"
        members = [
            bank["champ"],
            bank["part10"],
            bank["areeb"],
            bank["quey_p8"],
            bank["sohail"],
            bank["v4m"],
        ]
        w = np.array([0.28, 0.18, 0.16, 0.14, 0.12, 0.12], dtype=float)
        Z = np.column_stack([logit(rank01(m)) for m in members])
        s3 = sigmoid(Z @ w)

    paths["s3"] = save_sub(s3, OUT_DIR / "sub_n06_s3_adaptive_ascent.csv")
    preds["s3"] = s3
    submit(
        paths["s3"],
        f"N06-S3: Adaptive {note} | best_so_far={best_lb} aim>=0.953",
    )
    lb3 = poll_file(paths["s3"].name, timeout_s=720)
    report["ladder"].append(
        {
            "slot": 3,
            "file": str(paths["s3"].relative_to(ROOT)),
            "public_lb": lb3,
            "note": note,
        }
    )
    LOG_PATH.write_text(json.dumps(report, indent=2))

    # update best
    if lb3 is not None and lb3 > best_lb:
        best_lb, best_key, best_pred = lb3, "s3", preds["s3"]

    # ---- Slot 4 hard push multi-vector extrapolation ----
    # Collect verified ranked preds with known LB
    verified = [("champ", bank["champ"], 0.95242)]
    if lb1 is not None:
        verified.append(("s1", preds["s1"], lb1))
    if lb2 is not None:
        verified.append(("s2", preds["s2"], lb2))
    if lb3 is not None:
        verified.append(("s3", preds["s3"], lb3))
    verified.sort(key=lambda t: t[2], reverse=True)

    peak_name, peak_pred, peak_lb = verified[0]
    # weak / mid references for ascent axes
    weak = bank["live2"]  # 0.95240 close but slightly different
    mid = bank["part9c"]  # 0.95228
    orth_base = bank["part10_cv"]

    z4 = logit(rank01(peak_pred))
    # multi-vector: toward peak from weaker verified / known
    if len(verified) >= 2:
        _, p2, lb_p2 = verified[1]
        z4 = z4 + 0.30 * (logit(rank01(peak_pred)) - logit(rank01(p2)))
    z4 = z4 + 0.18 * (logit(rank01(peak_pred)) - logit(rank01(weak)))
    z4 = z4 + 0.12 * (logit(rank01(peak_pred)) - logit(rank01(mid)))
    # orthogonal novelty push
    z4 = z4 + 0.16 * orth_resid(bank["tabpack"], peak_pred)
    z4 = z4 + 0.10 * orth_resid(orth_base, peak_pred)
    z4 = z4 + 0.08 * orth_resid(bank["masaya"], peak_pred)
    # if S2 helped, lean into it
    if lb2 is not None and lb1 is not None and lb2 > lb1:
        z4 = z4 + 0.20 * (logit(rank01(preds["s2"])) - logit(rank01(preds["s1"])))
    elif lb1 is not None and lb2 is not None and lb1 > lb2:
        z4 = z4 + 0.15 * (logit(rank01(preds["s1"])) - logit(rank01(preds["s2"])))

    s4 = sigmoid(z4)
    note4 = (
        f"multi-vector extrap from peak={peak_name}@{peak_lb}; "
        f"axes vs verified#2/live2/part9c + TabPack/Part10cv/Masaya"
    )
    paths["s4"] = save_sub(s4, OUT_DIR / "sub_n06_s4_hardpush_extrap.csv")
    preds["s4"] = s4
    submit(
        paths["s4"],
        f"N06-S4: Hard-push multi-vector extrap from {peak_name}={peak_lb} + orth TabPack/Part10cv aim>=0.953",
    )
    lb4 = poll_file(paths["s4"].name, timeout_s=720)
    report["ladder"].append(
        {
            "slot": 4,
            "file": str(paths["s4"].relative_to(ROOT)),
            "public_lb": lb4,
            "note": note4,
        }
    )

    lbs = [x["public_lb"] for x in report["ladder"] if x["public_lb"] is not None]
    report["best_public_lb"] = max(lbs) if lbs else None
    report["passed_0953"] = bool(report["best_public_lb"] and report["best_public_lb"] >= 0.953)
    report["slots_used"] = 4
    LOG_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("WROTE", LOG_PATH)


if __name__ == "__main__":
    main()
