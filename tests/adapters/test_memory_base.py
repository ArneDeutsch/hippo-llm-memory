import hypothesis.extra.numpy as hnp
import torch
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from hippo_mem.adapters.memory_base import MemoryAdapterBase


@st.composite
def hidden_states(draw):
    seq_len = draw(st.integers(min_value=2, max_value=6))
    dim = draw(st.integers(min_value=1, max_value=8))
    arr = draw(hnp.arrays(dtype="float32", shape=(seq_len, dim), elements=st.floats(-1, 1)))
    tensor = torch.from_numpy(arr)
    assume(tensor[-1].norm().item() > 0)
    if tensor.size(0) > 1:
        assume(tensor[:-1].mean(dim=0).norm().item() > 0)
    return tensor


@given(hidden_states())
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_build_key_span_and_unit_norm(hidden: torch.Tensor) -> None:
    """Keys are unit-normalised and span-aware."""

    key_default = MemoryAdapterBase.build_key(hidden)
    key_last = MemoryAdapterBase.build_key(hidden, span=(hidden.size(0) - 1, hidden.size(0)))
    assert torch.allclose(key_default, key_last)
    assert torch.allclose(key_default.norm(), torch.tensor(1.0), atol=1e-6)

    expected = torch.nn.functional.normalize(hidden[:-1].mean(dim=0), p=2, dim=-1)
    key_span = MemoryAdapterBase.build_key(hidden, span=(0, hidden.size(0) - 1))
    assert torch.allclose(key_span, expected)
    assert torch.allclose(key_span.norm(), torch.tensor(1.0), atol=1e-6)
