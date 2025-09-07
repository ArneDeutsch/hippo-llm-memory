import pytest
from torch import nn

from hippo_mem.adapters.patch import find_transformer_blocks


class NoBlocks(nn.Module):
    pass


class EmptyBlocks(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])


def test_find_transformer_blocks_missing() -> None:
    model = NoBlocks()
    with pytest.raises(AttributeError):
        find_transformer_blocks(model)


def test_find_transformer_blocks_empty() -> None:
    model = EmptyBlocks()
    with pytest.raises(ValueError):
        find_transformer_blocks(model)
