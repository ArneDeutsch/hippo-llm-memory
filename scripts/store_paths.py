"""Re-exports store layout helpers for TEACH→TEST handoff."""

from hippo_mem.utils.stores import (
    StoreLayout,
    assert_store_exists,
    derive,
    is_memory_preset,
    validate_store,
)

__all__ = [
    "StoreLayout",
    "assert_store_exists",
    "derive",
    "is_memory_preset",
    "validate_store",
]
