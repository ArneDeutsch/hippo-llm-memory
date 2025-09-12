# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from hippo_eval.eval.harness import _evaluate, _init_modules
from hippo_eval.eval.modes import TestStrategy
from hippo_eval.eval.types import Task
from hippo_mem.common.telemetry import registry


def test_retrieval_flag_disables_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    tokenizer = AutoTokenizer.from_pretrained("models/tiny-gpt2")
    model = AutoModelForCausalLM.from_pretrained("models/tiny-gpt2")
    modules = _init_modules("hei_nw", {})
    tasks = [Task(prompt="hello", answer="world")]

    registry.reset()
    _evaluate(
        tasks,
        modules,
        tokenizer,
        model,
        8,
        use_chat_template=False,
        system_prompt=None,
        retrieval_enabled=False,
        long_context_enabled=False,
        strategy=TestStrategy(),
    )
    snaps = registry.all_snapshots()
    assert all(snap["requests"] == 0 for snap in snaps.values())
