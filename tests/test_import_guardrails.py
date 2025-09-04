"""Import boundary guardrail tests."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest


def test_no_hippo_eval_imports_in_hippo_mem() -> None:
    """Ensure core package does not depend on hippo_eval."""

    base = Path(__file__).resolve().parents[1] / "hippo_mem"
    for path in base.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("hippo_eval"):
                        raise AssertionError(f"{path} imports {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("hippo_eval"):
                    raise AssertionError(f"{path} imports {node.module}")


def test_legacy_paths_removed() -> None:
    """Old hippo_mem evaluation aliases have been dropped."""

    for mod in (
        "hippo_mem.eval",
        "hippo_mem.metrics",
        "hippo_mem.reporting",
        "hippo_mem.tasks",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(mod)
