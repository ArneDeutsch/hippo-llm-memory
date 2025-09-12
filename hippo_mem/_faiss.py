# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""FAISS import helper with warning suppression."""

from __future__ import annotations

import warnings

try:  # pragma: no cover - import side effects only
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"(?i)builtin type .*swig.* has no __module__ attribute",
            category=DeprecationWarning,
        )
        import faiss  # type: ignore
    for _name in ["SwigPyObject", "SwigPyPacked", "swigvarlink"]:
        _typ = getattr(faiss, _name, None)
        if _typ is not None and getattr(_typ, "__module__", None) is None:
            _typ.__module__ = "faiss"
except Exception:  # pragma: no cover - fallback path
    faiss = None  # type: ignore

__all__ = ["faiss"]
