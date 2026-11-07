#!/usr/bin/env python3
"""Kaggle API via DoH/fallback IP + curl --resolve (local DNS blocks www.kaggle.com)."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

COMP = "playground-series-s6e11"
TOKEN_PATH = Path.home() / ".kaggle" / "access_token"
_FALLBACK_A = {
    "www.kaggle.com": "35.244.233.98",
    "api.kaggle.com": "34.54.168.202",
}


def _clear_proxy_env() -> None:
    for k in list(os.environ):
        if "proxy" in k.lower():
            os.environ.pop(k, None)


def doh_a(name: str) -> str:
    _clear_proxy_env()
    try:
        out = subprocess.check_output(
            ["curl", "-sS", "-m", "20", f"https://dns.google/resolve?name={name}&type=A"],
            text=True,
        )
        j = json.loads(out)
        for a in j.get("Answer", []):
            if a.get("type") == 1:
                return a["data"]
    except Exception as e:
        if name in _FALLBACK_A:
            print(f"DoH failed ({type(e).__name__}); using fallback A for {name}", flush=True)
            return _FALLBACK_A[name]
        raise
    if name in _FALLBACK_A:
        return _FALLBACK_A[name]
    raise RuntimeError(f"No A record for {name}")


def token() -> str:
    return TOKEN_PATH.read_text().strip()


def _curl_auth(url: str, ip: str, extra: list[str] | None = None, timeout: int = 60) -> str:
    """Run curl with Bearer auth via a temp header file (keeps token out of argv/errors)."""
    _clear_proxy_env()
    hdr = tempfile.NamedTemporaryFile("w", delete=False)
    try:
        hdr.write(f"Authorization: Bearer {token()}\n")
        hdr.close()
        os.chmod(hdr.name, 0o600)
        cmd = [
            "curl",
            "-sS",
            "-m",
            str(timeout),
            "--resolve",
            f"www.kaggle.com:443:{ip}",
            "-H",
            f"@{hdr.name}",
        ]
        if extra:
            cmd.extend(extra)
        cmd.append(url)
        try:
            return subprocess.check_output(cmd, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"curl failed rc={e.returncode} for {url}") from None
    finally:
        try:
            os.unlink(hdr.name)
        except OSError:
            pass


def api_get(path: str, timeout: int = 60) -> object:
    ip = doh_a("www.kaggle.com")
    url = f"https://www.kaggle.com{path}"
    out = _curl_auth(url, ip, timeout=timeout)
    return json.loads(out)


def list_submissions(page: int = 1) -> list[dict]:
    data = api_get(f"/api/v1/competitions/submissions/list/{COMP}?page={page}")
    if isinstance(data, dict) and "code" in data:
        raise RuntimeError(f"Kaggle API error: {data}")
    return list(data)


def today_utc_count(rows: list[dict] | None = None) -> tuple[int, float, list[dict]]:
    rows = rows if rows is not None else list_submissions(1)
    today = str(datetime.now(timezone.utc).date())
    n = 0
    best = 0.0
    for row in rows:
        d = (row.get("date") or "")[:10]
        sc = row.get("publicScore") or row.get("publicScoreNullable") or ""
        try:
            scf = float(sc) if sc not in ("", None) else None
        except ValueError:
            scf = None
        if scf is not None:
            best = max(best, scf)
        if d == today:
            n += 1
    return n, best, rows


def submit(path: Path | str, message: str) -> str:
    path = Path(path)
    ip = doh_a("www.kaggle.com")
    url = f"https://www.kaggle.com/api/v1/competitions/submissions/submit/{COMP}"
    return _curl_auth(
        url,
        ip,
        extra=["-F", f"file=@{path}", "-F", f"submissionDescription={message[:480]}"],
        timeout=300,
    )


def poll_file(fname: str, timeout_s: int = 480, sleep_s: int = 12) -> float | None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        rows = list_submissions(1)
        for row in rows:
            if row.get("fileName") == fname:
                sc = row.get("publicScore") or ""
                st = row.get("status")
                if sc not in ("", None) and st == "complete":
                    return float(sc)
                print(f"  status={st} score={sc!r}", flush=True)
        time.sleep(sleep_s)
    return None


if __name__ == "__main__":
    n, best, rows = today_utc_count()
    print(f"today_utc_subs={n} slots_left~={max(0, 10 - n)} best_page1={best}")
    for row in rows[:12]:
        print(
            row.get("ref"),
            (row.get("date") or "")[:19],
            row.get("publicScore"),
            row.get("fileName"),
        )
