"""Import boundary guardrail tests."""

from __future__ import annotations

import ast
import importlib
import sys
import warnings
from pathlib import Path


def test_no_hippo_eval_imports_in_hippo_mem() -> None:
    """Ensure core package does not depend on hippo_eval."""

    base = Path(__file__).resolve().parents[1] / "hippo_mem"
    skip = {"eval", "metrics", "reporting", "tasks"}
    for path in base.rglob("*.py"):
        if path.name == "__init__.py" and path.parent.name in skip:
            continue
        rel = path.relative_to(base)
        if rel.parts and rel.parts[0] in skip:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("hippo_eval"):
                        raise AssertionError(f"{path} imports {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("hippo_eval"):
                    raise AssertionError(f"{path} imports {node.module}")


def test_import_shim_emits_warning() -> None:
    """Importing hippo_mem.eval emits a deprecation warning."""

    sys.modules.pop("hippo_mem.eval", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        importlib.import_module("hippo_mem.eval")
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
