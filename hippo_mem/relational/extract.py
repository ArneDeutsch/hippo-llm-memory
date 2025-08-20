"""Thin wrapper exposing tuple extraction utilities.

Summary
-------
Re-exports :func:`hippo_mem.relational.tuples.extract_tuples` so callers
can depend on a stable module path. This avoids importing the heavier
``tuples`` module in client code.
"""

from __future__ import annotations

from .tuples import TupleType, extract_tuples

__all__ = ["extract_tuples", "TupleType"]
