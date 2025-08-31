from __future__ import annotations

import math

import hypothesis.extra.numpy as hnp
import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from hippo_mem.episodic.gating import WriteGate


@given(
    st.floats(min_value=1e-6, max_value=1.0),
    st.floats(min_value=1e-6, max_value=1.0),
)
def test_score_monotonic_in_surprise(p_high: float, p_low: float) -> None:
    """Lower model probability yields a higher salience score."""

    assume(p_high > p_low + 1e-3)
    gate = WriteGate()
    query = np.zeros(1, dtype="float32")
    keys = np.zeros((0, 1), dtype="float32")
    s_high = gate.score(p_high, query, keys)
    s_low = gate.score(p_low, query, keys)
    assert s_low > s_high


@given(
    st.floats(min_value=0.0, max_value=math.pi / 2),
    st.floats(min_value=0.0, max_value=math.pi / 2),
)
def test_score_monotonic_in_novelty(angle1: float, angle2: float) -> None:
    """Greater novelty (larger angle) increases the score."""

    assume(angle2 > angle1 + 1e-3)
    gate = WriteGate()
    key = np.array([[1.0, 0.0]], dtype="float32")
    q1 = np.array([math.cos(angle1), math.sin(angle1)], dtype="float32")
    q2 = np.array([math.cos(angle2), math.sin(angle2)], dtype="float32")
    prob = 0.5
    s1 = gate.score(prob, q1, key)
    s2 = gate.score(prob, q2, key)
    assert s2 > s1


@given(
    st.floats(min_value=1e-6, max_value=1.0),
    hnp.arrays(np.float32, 4, elements=st.floats(-1.0, 1.0)),
)
def test_threshold_blocks_writes(prob: float, query: np.ndarray) -> None:
    """Scores below ``tau`` do not permit writes."""

    keys = np.zeros((0, 4), dtype="float32")
    base_gate = WriteGate()
    score = base_gate.score(prob, query, keys)
    assume(score < 1 - 1e-6)
    gate_block = WriteGate(tau=score + 1e-6)
    decision = gate_block(prob, query, keys)
    assert decision.action != "insert"
    gate_allow = WriteGate(tau=max(score - 1e-6, 0.0))
    decision2 = gate_allow(prob, query, keys)
    assert decision2.action == "insert"


def test_write_gate_rejects_bad_config() -> None:
    with pytest.raises(ValueError) as exc:
        WriteGate(tau=-0.1)
    assert "tau" in str(exc.value)
    with pytest.raises(ValueError) as exc:
        WriteGate(alpha=1.5)
    assert "alpha" in str(exc.value)
