"""Spatial metric canonicalisation tests."""

import pytest

from hippo_mem.metrics.spatial import em, ensure_prediction_format, f1

CASES = [
    ("(0,0) -> (0,1) -> (0,2)", "DD"),
    ("(1,1) -> (0,1) -> (0,0)", "LU"),
    ("[(2,2), (1,2), (1,3)]", "LD"),
    ("UURDDL", "UURDDL"),
    ("u u r d d l", "UURDDL"),
    ("[0,0] to [0,1]", "D"),
    ("0,0 -> 1,0 -> 1,1 -> 2,1", "RDR"),
    ("Up,Right,Down", "URD"),
    ("move: up up left", "UUL"),
    ("(0,0) → (1,0) → (1,1)", "RD"),
    ("(0,0) -> (0,1).", "D"),
]


@pytest.mark.parametrize("pred,gold", CASES)
def test_ensure_prediction_format(pred, gold):
    assert ensure_prediction_format(pred) == gold


@pytest.mark.parametrize("pred,gold", CASES)
def test_em_f1(pred, gold):
    assert em(pred, gold) == 1
    assert f1(pred, gold) == 1.0
