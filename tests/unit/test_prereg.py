"""V1: the pre-registration is frozen by a sha256 in PREREGISTRATION.lock (git-free, approved
2026-11-08). Any edit after freezing must be detectable; amendments are appended, never hidden."""

import json

import pytest

from edge_autopsy.prereg import PreregError, amend, fingerprint, freeze, verify


@pytest.fixture
def doc(tmp_path):
    p = tmp_path / "PREREGISTRATION.md"
    p.write_text("# Pre-registration\n\nH1: IC t > 2.0\n", encoding="utf-8")
    return p


def test_freeze_then_verify_passes(doc, tmp_path):
    lock = tmp_path / "PREREGISTRATION.lock"
    h = freeze(doc, lock, label="v1")
    assert verify(doc, lock) == h == fingerprint(doc)
    rec = json.loads(lock.read_text())
    assert rec["file"] == "PREREGISTRATION.md"
    assert [v["label"] for v in rec["versions"]] == ["v1"]
    assert rec["versions"][0]["sha256"] == h and rec["versions"][0]["frozen_at"].endswith("Z")


def test_one_character_edit_fails_verify(doc, tmp_path):
    lock = tmp_path / "PREREGISTRATION.lock"
    freeze(doc, lock, label="v1")
    doc.write_text(doc.read_text().replace("2.0", "1.9"))  # moving a goalpost
    with pytest.raises(PreregError, match="changed since it was frozen"):
        verify(doc, lock)


def test_missing_lock_fails_verify(doc, tmp_path):
    with pytest.raises(PreregError, match="not frozen"):
        verify(doc, tmp_path / "PREREGISTRATION.lock")


def test_missing_document_fails_verify(doc, tmp_path):
    lock = tmp_path / "PREREGISTRATION.lock"
    freeze(doc, lock, label="v1")
    doc.unlink()
    with pytest.raises(PreregError, match="missing"):
        verify(doc, lock)


def test_tampered_lock_fails_verify(doc, tmp_path):
    lock = tmp_path / "PREREGISTRATION.lock"
    freeze(doc, lock, label="v1")
    rec = json.loads(lock.read_text())
    rec["versions"][0]["sha256"] = "0" * 64
    lock.write_text(json.dumps(rec))
    with pytest.raises(PreregError, match="changed since it was frozen"):
        verify(doc, lock)


def test_cannot_freeze_twice(doc, tmp_path):
    """Re-freezing would silently bless an edited file; only amend() may add a version."""
    lock = tmp_path / "PREREGISTRATION.lock"
    freeze(doc, lock, label="v1")
    with pytest.raises(PreregError, match="already frozen"):
        freeze(doc, lock, label="v1b")


def test_amend_appends_a_version_and_keeps_history(doc, tmp_path):
    lock = tmp_path / "PREREGISTRATION.lock"
    h1 = freeze(doc, lock, label="v1")
    doc.write_text(doc.read_text() + "\n## Amendment 1 (2026-11-09)\n\nReason: typo.\n")
    h2 = amend(doc, lock, label="amendment-1")
    assert h2 != h1 and verify(doc, lock) == h2
    versions = json.loads(lock.read_text())["versions"]
    assert [v["sha256"] for v in versions] == [h1, h2]


def test_amend_requires_an_amendment_section_and_append_only(doc, tmp_path):
    lock = tmp_path / "PREREGISTRATION.lock"
    freeze(doc, lock, label="v1")
    doc.write_text(doc.read_text() + "\nquiet extra line\n")
    with pytest.raises(PreregError, match="Amendment"):
        amend(doc, lock, label="sneaky")
    doc.write_text("# Pre-registration\n\nH1: IC t > 1.0\n\n## Amendment 1\n\nlower bar\n")
    with pytest.raises(PreregError, match="append"):
        amend(doc, lock, label="rewrite")  # the frozen text itself was edited


def test_fingerprint_ignores_windows_line_endings(doc):
    h = fingerprint(doc)
    doc.write_bytes(doc.read_bytes().replace(b"\n", b"\r\n"))
    assert fingerprint(doc) == h


def test_project_preregistration_is_frozen_and_hash_is_in_readme():
    """V1 acceptance: the real PREREGISTRATION.md verifies and README shows the same sha256."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    h = verify(root / "PREREGISTRATION.md", root / "PREREGISTRATION.lock")
    assert h in (root / "README.md").read_text(encoding="utf-8")


def test_tasks_prereg_command_verifies(capsys):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "tasks.py"
    spec = importlib.util.spec_from_file_location("tasks", path)
    tasks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tasks)
    assert tasks.main(["prereg"]) == 0
    assert "verified" in capsys.readouterr().out
