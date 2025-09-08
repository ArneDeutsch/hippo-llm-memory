import torch
import torch.nn as nn

from hippo_eval.eval.harness import _evaluate
from hippo_eval.eval.modes import TeachStrategy
from hippo_eval.eval.types import Task
from hippo_mem.common.gates import GateCounters
from hippo_mem.common.telemetry import registry
from hippo_mem.episodic.gating import WriteGate
from hippo_mem.episodic.store import EpisodicStore


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, prompt, return_tensors):
        return {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }

    def decode(self, ids, skip_special_tokens=True):
        return "ok"


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cpu")

    def generate(self, **inputs):
        inp = inputs["input_ids"]
        gen = torch.tensor([[3, 4]])
        return torch.cat([inp, gen], dim=1)


class DummyAdapter(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)


def _modules_with_store() -> tuple[dict, dict]:
    store = EpisodicStore(8)
    modules = {
        "episodic": {
            "store": store,
            "adapter": DummyAdapter(8),
            "gate": WriteGate(tau=0.0),
        }
    }
    gating = {"episodic": GateCounters()}
    return modules, gating


def test_no_retrieval_during_teach() -> None:
    modules, gating = _modules_with_store()
    tasks = [Task(prompt="p", answer="a")]
    registry.reset()
    _evaluate(
        tasks,
        modules,
        tokenizer=DummyTokenizer(),
        model=DummyModel(),
        max_new_tokens=2,
        use_chat_template=False,
        system_prompt=None,
        retrieval_enabled=True,
        strategy=TeachStrategy(),
        gating=gating,
        isolate="none",
    )
    assert registry.get("episodic").requests == 0


def test_per_item_isolation_clears_store() -> None:
    modules, gating = _modules_with_store()
    tasks = [Task(prompt="p1", answer="a"), Task(prompt="p2", answer="b")]
    _evaluate(
        tasks,
        modules,
        tokenizer=DummyTokenizer(),
        model=DummyModel(),
        max_new_tokens=2,
        use_chat_template=False,
        system_prompt=None,
        retrieval_enabled=False,
        strategy=TeachStrategy(),
        gating=gating,
        isolate="per_item",
    )
    store = modules["episodic"]["store"]
    assert store.index.ntotal == 0
