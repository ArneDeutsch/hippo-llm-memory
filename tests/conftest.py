"""Pytest configuration for path setup and marker handling."""

import os
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("RUN_ID", "test")


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run tests marked as slow"
    )
    parser.addoption(
        "--runintegration",
        action="store_true",
        default=False,
        help="run integration tests",
    )


def pytest_collection_modifyitems(config, items):
    skip_slow = not config.getoption("--runslow")
    skip_integration = not config.getoption("--runintegration")

    for item in items:
        fspath = str(getattr(item, "fspath", ""))
        if "tests/cli/" in fspath.replace("\\", "/"):
            item.add_marker(pytest.mark.integration)

        if skip_slow and "slow" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="need --runslow to run slow tests"))
        if skip_integration and "integration" in item.keywords:
            item.add_marker(
                pytest.mark.skip(reason="need --runintegration to run integration tests")
            )


@pytest.fixture(scope="session", autouse=True)
def fast_llm_monkeypatch():
    class DummyTok:
        pad_token_id = 0
        eos_token_id = 0
        pad_token = "<pad>"
        eos_token = "<eos>"

        def __init__(self):
            pass

        def __call__(self, text, return_tensors=None, add_special_tokens=False):
            ids = torch.tensor([[1, 2, 3]])
            attn = torch.ones_like(ids)
            return {"input_ids": ids, "attention_mask": attn}

        def decode(self, ids, skip_special_tokens=True):
            return "ok"

    class DummyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(pad_token_id=0, eos_token_id=0)
            self.generation_config = types.SimpleNamespace(pad_token_id=0, eos_token_id=0)
            self.device = torch.device("cpu")

        def generate(self, input_ids=None, attention_mask=None, max_new_tokens=32, **kw):
            return input_ids

    if os.environ.get("FAST_TESTS", "1") != "1":
        return

    mp = pytest.MonkeyPatch()
    mp.setattr(
        "hippo_eval.eval.harness.AutoTokenizer",
        types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyTok()),
    )
    mp.setattr(
        "hippo_eval.eval.harness.AutoModelForCausalLM",
        types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyLM()),
    )
    yield
    mp.undo()
