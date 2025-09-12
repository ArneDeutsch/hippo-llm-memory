# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
import json
import sys
from pathlib import Path
from subprocess import run

from hippo_eval.eval import audit as audit_datasets


def test_audit_datasets_passes() -> None:
    """All required datasets and configs are present with correct checksums."""
    for suite in audit_datasets.SUITES:
        run(
            [
                sys.executable,
                "-m",
                "hippo_eval.datasets.cli",
                "--suite",
                suite,
                "--size",
                "50",
                "--seed",
                "1337",
                "--out",
                f"data/{suite}",
            ],
            check=True,
        )
    ok, issues = audit_datasets.audit()
    assert ok, f"Audit reported issues: {issues}"
    assert issues == []

    manifest_path = Path("data") / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())

    assert manifest["sizes"] == audit_datasets.SIZES
    assert manifest["seeds"] == audit_datasets.SEEDS

    data_root = Path("data")
    for suite in audit_datasets.SUITES:
        entries = {e["file"]: e for e in manifest[suite]}
        checksums = json.loads((data_root / suite / "checksums.json").read_text())

        assert len(entries) == len(checksums)
        for fname, digest in checksums.items():
            key = f"{suite}/{fname}"
            assert key in entries, f"Missing {key} in manifest"
            entry = entries[key]
            assert entry["sha256"] == digest
            assert (data_root / entry["file"]).exists()
