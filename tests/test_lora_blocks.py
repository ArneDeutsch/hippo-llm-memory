from __future__ import annotations

from types import SimpleNamespace

from hippo_mem.adapters.lora import (
    _block_gpt2,
    _block_llama,
    _find_first_block,
    _targets_llama,
    default_target_modules,
    inspect_first_block,
)


def test_targets_llama_returns_projection_names() -> None:
    """_targets_llama returns the canonical projection modules."""

    assert _targets_llama(object()) == [
        "q_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
    ]


def test_block_gpt2_missing_layers_returns_none() -> None:
    """_block_gpt2 handles models without ``h`` layers."""

    class Model:
        def __init__(self) -> None:
            class Sub:
                pass

            self.transformer = Sub()
            self.transformer.h = []

    assert _block_gpt2(Model()) is None


def test_block_llama_missing_layers_returns_none() -> None:
    """_block_llama handles models without ``layers``."""

    class Model:
        def __init__(self) -> None:
            self.model = SimpleNamespace(layers=[])

    assert _block_llama(Model()) is None


def test_find_first_block_fallback_when_absent() -> None:
    """_find_first_block returns ``None`` when no common attrs are present."""

    class Model:
        def __init__(self) -> None:
            self.config = SimpleNamespace()

    model = Model()
    assert _find_first_block(model) is None
    assert inspect_first_block(model) == []
    assert default_target_modules(model) == []
