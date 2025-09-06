import shutil
import subprocess
import sys
from pathlib import Path


def test_smpd_store_population(tmp_path: Path) -> None:
    run_id = "run_ci"
    stores = Path("runs") / run_id / "stores"
    sid = "smpd_run_ci"
    dataset = Path("data") / "spatial_multi_50_0.jsonl"
    dataset.write_text(
        '{"prompt": "p1", "answer": "a1"}\n{"prompt": "p2", "answer": "a2"}\n',
        encoding="utf-8",
    )
    try:
        subprocess.run(
            [
                sys.executable,
                "scripts/eval_cli.py",
                "suite=spatial_multi",
                "preset=memory/smpd",
                "n=2",
                "seed=0",
                f"run_id={run_id}",
                "mode=teach",
                "persist=true",
                f"store_dir={stores}",
                f"session_id={sid}",
                "strict_telemetry=true",
                "model=models/tiny-gpt2",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/validate_store.py",
                "--algo",
                "smpd",
                "--kind",
                "spatial",
                "--run_id",
                run_id,
            ],
            check=True,
        )
    finally:
        dataset.unlink(missing_ok=True)
        shutil.rmtree(Path("runs") / run_id, ignore_errors=True)
