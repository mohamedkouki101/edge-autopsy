"""Freeze the pre-registration (task V1; git-free version approved 2026-11-08).

`PREREGISTRATION.lock` (JSON) records, for each frozen version of PREREGISTRATION.md, its
sha256, its length and when it was frozen. Rules:
- `freeze` works once. Re-freezing an edited file would bless a moved goalpost.
- `amend` only accepts text appended after the frozen text that contains a `## Amendment`
  section; every earlier version stays in the lock. Nothing is ever rewritten.
- `verify` returns the current sha256 or raises `PreregError`. The hold-out guard (V3) calls it.
Line endings are normalised (CRLF -> LF), so a Windows and a WSL copy of the file agree.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

AMENDMENT_HEADER = "## Amendment"


class PreregError(RuntimeError):
    """The pre-registration is missing, not frozen, or changed outside an amendment."""


def _text(doc: Path) -> bytes:
    return Path(doc).read_bytes().replace(b"\r\n", b"\n")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fingerprint(doc: Path) -> str:
    return _sha(_text(doc))


def _now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _version(data: bytes, label: str) -> dict:
    return {"label": label, "sha256": _sha(data), "n_bytes": len(data), "frozen_at": _now()}


def _read_lock(lock: Path) -> dict:
    if not Path(lock).is_file():
        raise PreregError(f"pre-registration is not frozen: {lock} does not exist")
    return json.loads(Path(lock).read_text(encoding="utf-8"))


def _write_lock(lock: Path, rec: dict) -> None:
    Path(lock).write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")


def freeze(doc: Path, lock: Path, label: str = "v1") -> str:
    if Path(lock).exists():
        raise PreregError(f"already frozen ({lock}); use amend() to add a dated amendment")
    data = _text(doc)
    rec = {"file": Path(doc).name, "versions": [_version(data, label)]}
    _write_lock(lock, rec)
    return rec["versions"][0]["sha256"]


def verify(doc: Path, lock: Path) -> str:
    rec = _read_lock(lock)
    if not Path(doc).is_file():
        raise PreregError(f"pre-registration document missing: {doc}")
    current, frozen = fingerprint(doc), rec["versions"][-1]
    if current != frozen["sha256"]:
        raise PreregError(
            f"{Path(doc).name} changed since it was frozen ({frozen['label']}, "
            f"{frozen['frozen_at']}): sha256 {current[:12]} != {frozen['sha256'][:12]}. "
            f"Revert the edit, or append a '{AMENDMENT_HEADER}' section and call amend()."
        )
    return current


def amend(doc: Path, lock: Path, label: str) -> str:
    rec = _read_lock(lock)
    last = rec["versions"][-1]
    data = _text(doc)
    head, tail = data[: last["n_bytes"]], data[last["n_bytes"] :]
    if _sha(head) != last["sha256"]:
        raise PreregError("amendments must append to the frozen text; the frozen part was edited")
    if AMENDMENT_HEADER.encode() not in tail:
        raise PreregError(f"an amendment must add a '{AMENDMENT_HEADER} ...' section")
    rec["versions"].append(_version(data, label))
    _write_lock(lock, rec)
    return rec["versions"][-1]["sha256"]
