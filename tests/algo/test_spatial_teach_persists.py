from pathlib import Path

from hydra.utils import to_absolute_path  # noqa: F401
from omegaconf import OmegaConf

from hippo_eval.eval.harness import evaluate


def test_spatial_teach_writes_jsonl(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "suite": "spatial",
            "preset": "memory/smpd",
            "n": 2,
            "seed": 0,
            "mode": "teach",
            "persist": True,
            "store_dir": str(tmp_path / "stores" / "smpd"),
            "session_id": "smpd_test",
            "outdir": str(tmp_path / "out"),
            "strict_telemetry": True,
            "dataset_profile": None,
            "memory_off": False,
            "model": "models/tiny-gpt2",
            "pad_token_id": 0,
            "eos_token_id": 0,
            "max_new_tokens": 8,
            "use_chat_template": False,
            "system_prompt": "",
            "memory": {"spatial": {"gate": {"enabled": True}}},
        }
    )
    dataset = Path("data") / "spatial_50_0.jsonl"
    dataset.write_text(
        '{"prompt": "p1", "answer": "a1"}\n{"prompt": "p2", "answer": "a2"}\n',
        encoding="utf-8",
    )
    try:
        evaluate(cfg, tmp_path, preflight=False)  # in-process
    finally:
        dataset.unlink(missing_ok=True)
    f = tmp_path / "stores" / "smpd" / "smpd_test" / "spatial.jsonl"
    assert f.exists()
    # Must contain at least one non-blank line (meta or node)
    assert any(line.strip() for line in f.read_text(encoding="utf-8").splitlines())
