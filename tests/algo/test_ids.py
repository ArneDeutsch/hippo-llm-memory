# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""Tests for run identifier validation."""

import pytest

from hippo_mem.utils import validate_run_id


@pytest.mark.parametrize("good", ["run1", "RID_123-foo"])
def test_valid_run_ids(good: str) -> None:
    assert validate_run_id(good) == good


def test_non_string_rejected() -> None:
    with pytest.raises(TypeError):
        validate_run_id(123)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["ab", "bad id", "a" * 65, "bad/id"])
def test_invalid_slug_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        validate_run_id(bad)
