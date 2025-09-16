# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
import csv
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from hippo_eval.eval import harness
from hippo_mem.testing import FAKE_MODEL_ID

GOLDEN = Path(__file__).resolve().parent / "golden"
CASES = [
    ("episodic_cross_mem", "memory/hei_nw"),
    ("semantic_mem", "memory/sgc_rss"),
    ("spatial_multi", "memory/smpd"),
]


def _strip(obj):
    if isinstance(obj, dict):
        return {
            k: _strip(v)
            for k, v in obj.items()
            if not any(tok in k for tok in ("time", "rss", "latency"))
        }
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


@pytest.mark.parametrize("suite,preset", CASES)
def test_harness_golden(tmp_path, suite, preset):
    cfg = OmegaConf.load("configs/eval/default.yaml")
    cfg.suite = suite
    cfg.preset = preset
    cfg.n = 5
    cfg.seed = 1337
    cfg.run_id = "golden"
    cfg.model = FAKE_MODEL_ID
    cfg = harness._load_preset(cfg)
    cfg = harness._apply_model_defaults(cfg)
    harness.evaluate(cfg, tmp_path, preflight=False)

    # metrics.json
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    gold_metrics = json.loads((GOLDEN / suite / "metrics.json").read_text())
    assert _strip(metrics) == _strip(gold_metrics)

    # meta.json
    meta = json.loads((tmp_path / "meta.json").read_text())
    gold_meta = json.loads((GOLDEN / suite / "meta.json").read_text())
    meta.pop("git_sha", None)
    gold_meta.pop("git_sha", None)
    assert _strip(meta) == _strip(gold_meta)

    # metrics.csv
    def read_csv(path: Path):
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            headers = [
                h
                for h in reader.fieldnames
                if not any(tok in h for tok in ("time", "rss", "latency"))
            ]
            rows = []
            for row in reader:
                rows.append({h: row[h] for h in headers})
        return headers, rows

    head, rows = read_csv(tmp_path / "metrics.csv")
    gold_head, gold_rows = read_csv(GOLDEN / suite / "metrics.csv")
    assert head == gold_head
    assert rows == gold_rows
