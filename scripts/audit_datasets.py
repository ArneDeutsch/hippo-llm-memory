"""Audit presence and checksums of dataset files and evaluation configs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from hippo_mem.eval.datasets import sha256_file

SIZES = [50, 200, 1000]
SEEDS = [1337, 2025, 4242]

SUITES = ["episodic", "semantic", "spatial"]

EPISODIC_PATTERN = "episodic_{size}_{seed}.jsonl"
SEMANTIC_PATTERNS = [
    "semantic_h2_clean_{size}_{seed}.jsonl",
    "semantic_h3_contradict_{size}_{seed}.jsonl",
]
SPATIAL_PATTERN = "spatial_{size}_{seed}.jsonl"

MEMORY_CONFIGS = [
    Path("configs/eval/memory/hei_nw.yaml"),
    Path("configs/eval/memory/sgc_rss.yaml"),
    Path("configs/eval/memory/smpd.yaml"),
    Path("configs/eval/memory/all.yaml"),
]
BASELINE_CONFIGS = [
    Path("configs/eval/baselines/core.yaml"),
    Path("configs/eval/baselines/rag.yaml"),
    Path("configs/eval/baselines/longctx.yaml"),
]


def read_checksums(path: Path) -> Dict[str, str]:
    """Parse a checksums file into a mapping of filename to digest."""
    mapping: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        digest, filename = line.split(maxsplit=1)
        mapping[filename.strip()] = digest
    return mapping


def audit(data_dir: Path | None = None) -> Tuple[bool, List[str]]:
    """Verify datasets and configs exist with correct checksums and emit manifest.

    Parameters
    ----------
    data_dir:
        Directory containing dataset JSONL files and ``checksums.txt``.

    Returns
    -------
    tuple[bool, list[str]]
        ``True`` and an empty list if all checks pass; ``False`` and issue
        descriptions otherwise.
    """

    data_dir = data_dir or Path("data")
    checksum_path = data_dir / "checksums.txt"
    issues: List[str] = []
    if not checksum_path.exists():
        return False, [f"Missing checksums file: {checksum_path}"]

    checksum_map = read_checksums(checksum_path)

    for size in SIZES:
        for seed in SEEDS:
            fname = EPISODIC_PATTERN.format(size=size, seed=seed)
            verify_file(data_dir / fname, checksum_map, issues)
            for pattern in SEMANTIC_PATTERNS:
                fname = pattern.format(size=size, seed=seed)
                verify_file(data_dir / fname, checksum_map, issues)
            fname = SPATIAL_PATTERN.format(size=size, seed=seed)
            verify_file(data_dir / fname, checksum_map, issues)

    for cfg in MEMORY_CONFIGS + BASELINE_CONFIGS:
        if not cfg.exists():
            issues.append(f"Missing config: {cfg}")

    manifest = build_manifest(data_dir, issues)
    (data_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    return not issues, issues


def verify_file(path: Path, checksum_map: Dict[str, str], issues: List[str]) -> None:
    """Check that ``path`` exists and matches the recorded checksum."""
    if not path.exists():
        issues.append(f"Missing dataset: {path.name}")
        return
    digest = sha256_file(path)
    recorded = checksum_map.get(path.name)
    if recorded != digest:
        issues.append(f"Checksum mismatch: {path.name}")


def build_manifest(data_dir: Path, issues: List[str]) -> Dict[str, Any]:
    """Construct a manifest of dataset files with checksums and item counts."""
    manifest: Dict[str, Any] = {"sizes": SIZES, "seeds": SEEDS}
    for suite in SUITES:
        suite_dir = data_dir / suite
        checksum_file = suite_dir / "checksums.json"
        if not checksum_file.exists():
            issues.append(f"Missing checksums file: {checksum_file}")
            continue
        try:
            checksums = json.loads(checksum_file.read_text())
        except json.JSONDecodeError:
            issues.append(f"Invalid JSON: {checksum_file}")
            continue
        entries: List[Dict[str, Any]] = []
        for fname, recorded in sorted(checksums.items()):
            path = suite_dir / fname
            if not path.exists():
                issues.append(f"Missing dataset: {path}")
                continue
            digest = sha256_file(path)
            if digest != recorded:
                issues.append(f"Checksum mismatch: {path}")
            with path.open("r", encoding="utf-8") as fh:
                items = sum(1 for _ in fh)
            entries.append({"file": f"{suite}/{fname}", "sha256": digest, "items": items})
        manifest[suite] = entries
    return manifest


def main() -> int:
    """CLI entry point returning ``0`` on success, ``1`` otherwise."""
    ok, problems = audit()
    if not ok:
        for problem in problems:
            print(problem)
        return 1
    print("All datasets and configs verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
