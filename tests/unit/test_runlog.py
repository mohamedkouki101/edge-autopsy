"""B5/P3: every run appends one JSON row with the PLAN §7.4 fields.

Provenance is a code fingerprint, not git (Kouki, 2026-11-08: the build never runs git)."""

import json

from edge_autopsy.runner import log_run

REQUIRED = {
    "run_id",
    "code_hash",
    "config_hash",
    "data_manifest_hash",
    "phase",
    "universe",
    "model",
    "params",
    "portfolio",
    "cost_bps",
    "lag",
    "borrow_bps",
    "seed",
    "n_inner_trials",
    "metrics",
    "results_path",
    "pre_registered",
    "notes",
}


def test_log_run_appends_rows(tmp_path):
    log = tmp_path / "experiments.jsonl"
    rec = dict(
        phase="dev",
        universe="ff49",
        model="momentum_12_1",
        params={},
        portfolio={},
        cost_bps=10,
        lag=1,
        borrow_bps=25,
        seed=1,
        n_inner_trials=0,
        metrics={"sharpe_net": 0.1},
        results_path="results/x/",
        pre_registered=False,
        notes="",
        config_hash="h",
        data_manifest_hash="d",
    )
    a = log_run(log, rec)
    b = log_run(log, rec)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 2 and a != b
    assert REQUIRED <= set(rows[0])
    assert not {"git_commit", "git_dirty"} & set(rows[0])
    assert len(rows[0]["code_hash"]) == 64
    assert rows[0]["metrics"] == rows[1]["metrics"]


def test_log_run_rejects_holdout_phase_without_unseal(tmp_path):
    import pytest

    with pytest.raises(PermissionError):
        log_run(tmp_path / "e.jsonl", {"phase": "holdout"})


def test_non_finite_metrics_written_as_null(tmp_path):
    log = tmp_path / "e.jsonl"
    log_run(log, {"phase": "dev", "metrics": {"a": float("nan"), "b": float("inf"), "c": 1.0}})
    text = log.read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["metrics"] == {"a": None, "b": None, "c": 1.0}


def _tree(root):
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "a.py").write_text("x = 1\n")
    (root / "configs").mkdir()
    (root / "configs" / "base.yaml").write_text("seed: 1\n")
    (root / "pyproject.toml").write_text("[project]\n")
    (root / "uv.lock").write_text("lock\n")
    (root / "tasks.py").write_text("t = 1\n")
    (root / "experiments.jsonl").write_text("")
    (root / "results").mkdir()


def test_code_hash_ignores_run_outputs_and_caches(tmp_path):
    from edge_autopsy.runner import code_fingerprint

    _tree(tmp_path)
    h0 = code_fingerprint(tmp_path)
    (tmp_path / "experiments.jsonl").write_text('{"a": 1}\n')  # appending a row
    (tmp_path / "results" / "r.parquet").write_bytes(b"x")
    (tmp_path / "src" / "pkg" / "__pycache__").mkdir()
    (tmp_path / "src" / "pkg" / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\0")
    (tmp_path / "README.md").write_text("docs are not code\n")
    assert code_fingerprint(tmp_path) == h0


def test_code_hash_changes_with_code_config_and_dependencies(tmp_path):
    from edge_autopsy.runner import code_fingerprint

    _tree(tmp_path)
    h0 = code_fingerprint(tmp_path)
    for rel in ("src/pkg/a.py", "configs/base.yaml", "uv.lock", "pyproject.toml", "tasks.py"):
        f = tmp_path / rel
        old = f.read_text()
        f.write_text(old + "# changed\n")
        assert code_fingerprint(tmp_path) != h0, rel
        f.write_text(old)
        assert code_fingerprint(tmp_path) == h0, rel  # deterministic: revert -> same hash
    (tmp_path / "src" / "pkg" / "a.py").rename(tmp_path / "src" / "pkg" / "b.py")
    assert code_fingerprint(tmp_path) != h0  # paths are part of the fingerprint
    (tmp_path / "src" / "pkg" / "b.py").rename(tmp_path / "src" / "pkg" / "a.py")
    (tmp_path / "src" / "pkg" / "new.py").write_text("")
    assert code_fingerprint(tmp_path) != h0  # an added (even empty) file counts


def test_code_hash_same_on_windows_and_linux_line_endings(tmp_path):
    from edge_autopsy.runner import code_fingerprint

    _tree(tmp_path)
    h0 = code_fingerprint(tmp_path)
    (tmp_path / "src" / "pkg" / "a.py").write_bytes(b"x = 1\r\n")
    assert code_fingerprint(tmp_path) == h0


def test_log_run_never_calls_git_or_any_subprocess(tmp_path, monkeypatch):
    import subprocess

    def boom(*a, **k):
        raise AssertionError(f"subprocess called: {a}")

    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    log_run(tmp_path / "e.jsonl", {"phase": "dev", "metrics": {}})
    assert json.loads((tmp_path / "e.jsonl").read_text())["code_hash"]
