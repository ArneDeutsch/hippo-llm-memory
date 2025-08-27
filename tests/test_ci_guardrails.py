"""Tests for automated CI guardrails."""

import json
from pathlib import Path

import pytest

from hippo_mem.eval import harness as eval_model
from scripts import test_consolidation


@pytest.mark.parametrize(
    "preset",
    [
        "configs/eval/memory/hei_nw.yaml",
        "configs/eval/memory/sgc_rss.yaml",
        "configs/eval/memory/smpd.yaml",
    ],
)
def test_retrieval_requests_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preset: str
) -> None:
    """Memory runs must perform at least one retrieval."""

    data_file = tmp_path / "data.jsonl"
    data_file.write_text("")

    cfg = eval_model.EvalConfig(
        suite="episodic",
        n=0,
        seed=0,
        preset=preset,
    )

    monkeypatch.setattr(eval_model, "_dataset_path", lambda s, n, seed: data_file)
    with pytest.raises(RuntimeError, match="retrieval.requests == 0"):
        eval_model.run_suite(cfg)


def test_refusal_rate_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """High refusal rate on span suites should raise an error."""

    # dataset
    data_file = tmp_path / "d.jsonl"
    data_file.write_text(json.dumps({"prompt": "p", "answer": "a"}) + "\n")
    monkeypatch.setattr(eval_model, "_dataset_path", lambda s, n, seed: data_file)

    class DummyTokenizer:
        pad_token_id = 0
        eos_token_id = 0
        pad_token = None
        eos_token = ""

        def __call__(self, prompt, return_tensors):  # pragma: no cover - helper
            import torch

            return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}

        def decode(self, ids, skip_special_tokens=True):  # pragma: no cover - helper
            return "I cannot help with that"

    class DummyModel:
        device = "cpu"
        config = type("C", (), {})()
        generation_config = type("G", (), {})()

        def generate(self, **inputs):  # pragma: no cover - helper
            import torch

            return torch.tensor([[1, 2, 3]])

        def to(self, device):  # pragma: no cover - helper
            return self

    monkeypatch.setattr(
        eval_model,
        "AutoTokenizer",
        type("T", (), {"from_pretrained": lambda *a, **k: DummyTokenizer()}),
    )
    monkeypatch.setattr(
        eval_model,
        "AutoModelForCausalLM",
        type("M", (), {"from_pretrained": lambda *a, **k: DummyModel()}),
    )

    cfg = eval_model.EvalConfig(
        suite="episodic",
        n=1,
        seed=0,
        preset="configs/eval/baselines/core.yaml",
        use_chat_template=True,
        max_new_tokens=8,
    )
    with pytest.raises(RuntimeError, match="refusal rate > 0.5"):
        eval_model.run_suite(cfg)


def test_consolidation_uplift_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Post phase should fail when EM uplift is below threshold."""

    pre_dir = tmp_path / "pre"
    pre_dir.mkdir()
    pre_data = {
        "suite": "episodic",
        "metrics": {"episodic": {"pre_em_raw": 0.0}},
    }
    (pre_dir / "metrics.json").write_text(json.dumps(pre_data))

    post_dir = tmp_path / "post"

    def fake_eval(cfg, outdir):  # pragma: no cover - helper
        data = {
            "suite": "episodic",
            "metrics": {"episodic": {"pre_em_raw": 0.0, "post_em_raw": 0.1}},
        }
        Path(outdir).mkdir(parents=True, exist_ok=True)
        (Path(outdir) / "metrics.json").write_text(json.dumps(data))

    monkeypatch.setattr(test_consolidation, "_prepare_model_with_adapter", lambda m, a: (m, None))
    monkeypatch.setattr(eval_model, "evaluate", fake_eval)

    args = [
        "--phase",
        "post",
        "--suite",
        "episodic",
        "--n",
        "50",
        "--seed",
        "1337",
        "--model",
        "models/tiny-gpt2",
        "--adapter",
        str(tmp_path / "adapter"),
        "--pre_dir",
        str(pre_dir),
        "--outdir",
        str(post_dir),
    ]
    with pytest.raises(RuntimeError, match=r"EM uplift < \+0.20"):
        test_consolidation.main(args)
