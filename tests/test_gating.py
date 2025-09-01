import json
import subprocess
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape


def test_teach_records_gate_counts(tmp_path: Path) -> None:
    outdir = tmp_path / "run"
    cmd = [
        sys.executable,
        "scripts/eval_model.py",
        "suite=episodic",
        "preset=memory/hei_nw",
        "n=1",
        "seed=1337",
        "model=models/tiny-gpt2",
        f"outdir={outdir}",
        "mode=teach",
        "dry_run=true",
    ]
    subprocess.run(cmd, check=True)
    data = json.loads((outdir / "metrics.json").read_text())
    stats = data["gating"]["episodic"]
    assert stats["attempts"] > 0
    assert stats["attempts"] >= stats["accepted"] + stats["blocked"] + stats["skipped"]


def test_gating_template_renders() -> None:
    env = Environment(
        loader=FileSystemLoader("reports/templates"),
        autoescape=select_autoescape(["md"]),
    )
    template = env.get_template("partials/gating.md.j2")
    text = template.render(
        gating={"episodic": {"attempts": 5, "accepted": 3, "blocked": 2, "skipped": 0}}
    )
    assert "| episodic | 5 | 3 | 2 |" in text
