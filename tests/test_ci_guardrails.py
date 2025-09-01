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
        model="models/tiny-gpt2",
    )

    monkeypatch.setattr(eval_model, "_dataset_path", lambda s, n, seed, profile=None: data_file)
    with pytest.raises(RuntimeError) as exc:
        eval_model.run_suite(cfg)
    assert "retrieval.requests == 0" in str(exc.value)


def test_retrieval_tokens_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Memory runs must return retrieval tokens."""

    data_file = tmp_path / "data.jsonl"
    data_file.write_text("")

    cfg = eval_model.EvalConfig(
        suite="episodic",
        n=0,
        seed=0,
        preset="configs/eval/memory/hei_nw.yaml",
        model="models/tiny-gpt2",
    )

    monkeypatch.setattr(eval_model, "_dataset_path", lambda s, n, seed, profile=None: data_file)
    monkeypatch.setattr(
        eval_model.registry,
        "all_snapshots",
        lambda: {"episodic": {"requests": 1, "tokens_returned": 0}},
    )
    with pytest.raises(RuntimeError) as exc:
        eval_model.run_suite(cfg)
    assert "retrieval.tokens_returned == 0" in str(exc.value)


def test_refusal_rate_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """High refusal rate on span suites should raise an error."""

    # dataset
    data_file = tmp_path / "d.jsonl"
    data_file.write_text(json.dumps({"prompt": "p", "answer": "a"}) + "\n")
    monkeypatch.setattr(eval_model, "_dataset_path", lambda s, n, seed, profile=None: data_file)

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
        model="models/tiny-gpt2",
    )
    with pytest.raises(RuntimeError) as exc:
        eval_model.run_suite(cfg)
    assert "refusal rate > 0.5" in str(exc.value)


def test_consolidation_uplift_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Post phase should fail when EM uplift is below threshold."""

    pre_dir = tmp_path / "pre"
    pre_dir.mkdir()
    pre_data = {
        "suite": "episodic",
        "metrics": {"episodic": {"pre_em": 0.0}},
    }
    (pre_dir / "metrics.json").write_text(json.dumps(pre_data))

    meta = tmp_path / "stores" / "hei_nw" / "sid" / "store_meta.json"
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps({"replay_samples": 1, "source": "replay"}))

    post_dir = tmp_path / "post"

    def fake_eval(cfg, outdir):  # pragma: no cover - helper
        data = {
            "suite": "episodic",
            "metrics": {"episodic": {"pre_em": 0.0, "post_em": 0.1}},
        }
        Path(outdir).mkdir(parents=True, exist_ok=True)
        (Path(outdir) / "metrics.json").write_text(json.dumps(data))

    monkeypatch.setattr(test_consolidation, "_prepare_model_with_adapter", lambda m, a: (m, None))
    monkeypatch.setattr(test_consolidation.eval_model, "evaluate", fake_eval)

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
        "--allow-tiny-test-model",
        "--adapter",
        str(tmp_path / "adapter"),
        "--pre_dir",
        str(pre_dir),
        "--outdir",
        str(post_dir),
        "--uplift-mode",
        "fixed",
        "--min-em-uplift",
        "0.2",
    ]
    with pytest.raises(RuntimeError) as exc:
        test_consolidation.main(args)
    assert "EM uplift < +0.20" in str(exc.value)


def test_consolidation_ci_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CI-based uplift gate passes when deltas' CI excludes 0."""

    pre_dir = tmp_path / "pre"
    pre_dir.mkdir()
    pre_data = {"suite": "episodic", "metrics": {"episodic": {"pre_em": 0.0}}}
    (pre_dir / "metrics.json").write_text(json.dumps(pre_data))

    meta = tmp_path / "stores" / "hei_nw" / "sid" / "store_meta.json"
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps({"replay_samples": 1, "source": "replay"}))

    post_root = tmp_path / "post"
    seed_a = post_root / "seed_a"
    seed_a.mkdir(parents=True)
    existing = {
        "suite": "episodic",
        "metrics": {"episodic": {"pre_em": 0.0, "post_em": 0.1}},
        "delta": {"em": 0.1},
    }
    (seed_a / "metrics.json").write_text(json.dumps(existing))

    def fake_eval(cfg, outdir):  # pragma: no cover - helper
        data = {
            "suite": "episodic",
            "metrics": {"episodic": {"pre_em": 0.0, "post_em": 0.12}},
        }
        Path(outdir).mkdir(parents=True, exist_ok=True)
        (Path(outdir) / "metrics.json").write_text(json.dumps(data))

    monkeypatch.setattr(test_consolidation, "_prepare_model_with_adapter", lambda m, a: (m, None))
    monkeypatch.setattr(test_consolidation.eval_model, "evaluate", fake_eval)

    args = [
        "--phase",
        "post",
        "--suite",
        "episodic",
        "--n",
        "50",
        "--seed",
        "2025",
        "--model",
        "models/tiny-gpt2",
        "--allow-tiny-test-model",
        "--adapter",
        str(tmp_path / "adapter"),
        "--pre_dir",
        str(pre_dir),
        "--outdir",
        str(post_root / "seed_b"),
        "--uplift-mode",
        "ci",
    ]
    test_consolidation.main(args)
    metrics = json.loads((post_root / "seed_b" / "metrics.json").read_text())
    assert metrics["delta"]["em"] > 0


def test_consolidation_ci_mode_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CI gate fails when CI includes 0."""

    pre_dir = tmp_path / "pre"
    pre_dir.mkdir()
    pre_data = {"suite": "episodic", "metrics": {"episodic": {"pre_em": 0.0}}}
    (pre_dir / "metrics.json").write_text(json.dumps(pre_data))

    meta = tmp_path / "stores" / "hei_nw" / "sid" / "store_meta.json"
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps({"replay_samples": 1, "source": "replay"}))

    post_root = tmp_path / "post"
    seed_a = post_root / "seed_a"
    seed_a.mkdir(parents=True)
    existing = {
        "suite": "episodic",
        "metrics": {"episodic": {"pre_em": 0.0, "post_em": -0.05}},
        "delta": {"em": -0.05},
    }
    (seed_a / "metrics.json").write_text(json.dumps(existing))

    def fake_eval(cfg, outdir):  # pragma: no cover - helper
        data = {
            "suite": "episodic",
            "metrics": {"episodic": {"pre_em": 0.0, "post_em": 0.01}},
        }
        Path(outdir).mkdir(parents=True, exist_ok=True)
        (Path(outdir) / "metrics.json").write_text(json.dumps(data))

    monkeypatch.setattr(test_consolidation, "_prepare_model_with_adapter", lambda m, a: (m, None))
    monkeypatch.setattr(test_consolidation.eval_model, "evaluate", fake_eval)

    args = [
        "--phase",
        "post",
        "--suite",
        "episodic",
        "--n",
        "50",
        "--seed",
        "2025",
        "--model",
        "models/tiny-gpt2",
        "--allow-tiny-test-model",
        "--adapter",
        str(tmp_path / "adapter"),
        "--pre_dir",
        str(pre_dir),
        "--outdir",
        str(post_root / "seed_b"),
        "--uplift-mode",
        "ci",
    ]
    with pytest.raises(RuntimeError) as exc:
        test_consolidation.main(args)
    assert "CI includes 0" in str(exc.value)
