"""Tiny fake French-library zips with the real files' layout (header text, blocks, footer)."""

from __future__ import annotations

import io
import zipfile

IND_NAMES = [f"Ind{i:02d}" for i in range(1, 50)]


def _row(date: str, values: list[float]) -> str:
    return date + "," + ",".join(f"{v:7.2f}" for v in values)


def ind49_csv(vw_rows: dict[str, list[float]], ew_rows: dict[str, list[float]]) -> str:
    head = ",".join([""] + IND_NAMES)
    lines = [
        "This file was created using the 202609 CRSP database.",
        "It contains value- and equal-weighted returns for 49 industry portfolios.",
        "",
        "The portfolios are constructed at the end of June.",
        "",
        "Missing data are indicated by -99.99 or -999.",
        "",
        "",
        "  Average Value Weighted Returns -- Daily",
        head,
        *[_row(d, v) for d, v in vw_rows.items()],
        "",
        "  Average Equal Weighted Returns -- Daily",
        head,
        *[_row(d, v) for d, v in ew_rows.items()],
        "",
        "Copyright 2026 Eugene F. Fama and Kenneth R. French",
    ]
    return "\r\n".join(lines) + "\r\n"


def ff3_csv(rows: dict[str, list[float]]) -> str:
    lines = [
        "This file was created by using the 202609 CRSP database.",
        "The Tbill return is the simple daily rate ...",
        "",
        ",Mkt-RF,SMB,HML,RF",
        *[_row(d, v) for d, v in rows.items()],
        "",
        "Copyright 2026 Eugene F. Fama and Kenneth R. French",
    ]
    return "\r\n".join(lines) + "\r\n"


def mom_csv(rows: dict[str, float]) -> str:
    lines = [
        "This file was created by using the 202609 CRSP database.  It",
        "contains a momentum factor ...",
        "",
        "Missing data are indicated by -99.99 or -999.",
        "",
        ",Mom",
        *[_row(d, [v]) for d, v in rows.items()],
        "",
        "Copyright 2026 Eugene F. Fama and Kenneth R. French",
    ]
    return "\r\n".join(lines) + "\r\n"


def zip_bytes(csv_name: str, text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(csv_name, text.encode("latin-1"))
    return buf.getvalue()


def default_ind49_zip() -> bytes:
    vw = {"20260928": [0.10] * 49, "20260929": [-0.20] * 49, "20260930": [0.30] * 49}
    ew = {"20260928": [1.0] * 49, "20260929": [2.0] * 49, "20260930": [3.0] * 49}
    return zip_bytes("49_Industry_Portfolios_Daily.csv", ind49_csv(vw, ew))
