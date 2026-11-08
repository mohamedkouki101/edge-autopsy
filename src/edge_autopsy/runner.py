"""Experiment log (PLAN §7.4). Every run appends one JSON row to experiments.jsonl.

V5 extends this into run_experiment(cfg). Rows are never edited or deleted.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


# Provenance without git (Kouki, 2026-11-08: this build never runs git). A run is stamped with
# a sha256 over everything that can change its numbers: the code, the configs and the pinned
# dependencies. Run outputs, docs and caches are not part of it.
CODE_DIRS = ("src", "configs")
CODE_FILES = ("pyproject.toml", "uv.lock", "tasks.py")
_SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache"}


def _code_files(root: Path) -> list[Path]:
    files = [root / f for f in CODE_FILES if (root / f).is_file()]
    for d in CODE_DIRS:
        for f in (root / d).rglob("*"):
            cached = _SKIP_DIRS & set(f.relative_to(root).parts) or f.suffix == ".pyc"
            if f.is_file() and not cached:
                files.append(f)
    return sorted(files, key=lambda f: f.relative_to(root).as_posix())


def code_fingerprint(root: Path = ROOT) -> str:
    """sha256 over (relative path, content) of the code, configs and lock file.
    CRLF is normalised to LF so a Windows and a WSL checkout of the same code agree."""
    h = hashlib.sha256()
    for f in _code_files(root):
        rel = f.relative_to(root).as_posix().encode()
        data = f.read_bytes().replace(b"\r\n", b"\n")
        h.update(len(rel).to_bytes(8, "big") + rel + len(data).to_bytes(8, "big") + data)
    return h.hexdigest()


def make_run_id() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H%M%SZ") + "_" + secrets.token_hex(3)


def _json_safe(x):
    """NaN/inf are not valid JSON: write them as null."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_json_safe(v) for v in x]
    return x


def log_run(log_path: Path, record: dict, run_id: str | None = None) -> str:
    """Append `record` (+ run_id, code_hash, timestamp) to the JSONL log; return run_id."""
    if record.get("phase") == "holdout" and os.environ.get("UNSEAL") != "I_ACCEPT":
        raise PermissionError("phase=holdout rows need UNSEAL=I_ACCEPT (P9 only)")
    row = {
        "run_id": run_id or make_run_id(),
        "logged_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_hash": code_fingerprint(),
        **record,
    }
    with Path(log_path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(row), sort_keys=True, allow_nan=False) + "\n")
    return row["run_id"]
