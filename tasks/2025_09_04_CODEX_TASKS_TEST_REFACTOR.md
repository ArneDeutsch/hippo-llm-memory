# Codex Tasks — Make Tests Fast by Default

Each task is self-contained and aligned with the current repo layout. Apply in order.

---

## T1 — Add `integration` and `smoke` markers, and auto-mark CLI tests

**Why:** Separate slow/CLI tests from the default unit+s moke suite; skip them unless explicitly requested.

**Edits:**

1) In `pyproject.toml` add markers:

```toml
[tool.pytest.ini_options]
markers = [
  "slow: mark tests as slow and excluded from default runs",
  "integration: mark CLI/Hydra/real harness tests",
  "smoke: mark minimal e2e smoke checks"
]
```

2) Update `tests/conftest.py` to add `--runintegration` and auto-mark any test collected from `tests/cli/` as `integration`:

```python
# tests/conftest.py
import os, sys
from pathlib import Path
import pytest

def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run tests marked as slow")
    parser.addoption("--runintegration", action="store_true", default=False, help="run integration tests")

def pytest_collection_modifyitems(config, items):
    skip_slow = not config.getoption("--runslow")
    skip_integration = not config.getoption("--runintegration")

    for item in items:
        fspath = str(getattr(item, "fspath", ""))
        # Auto-mark CLI folder as integration
        if "tests/cli/" in fspath.replace("\\", "/"):
            item.add_marker(pytest.mark.integration)

        if skip_slow and "slow" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="need --runslow to run slow tests"))
        if skip_integration and "integration" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="need --runintegration to run integration tests"))
```

**Acceptance:**  
- `pytest -q` runs without collecting CLI tests.  
- `pytest -q --runintegration` includes tests under `tests/cli/`.

---

## T2 — Collapse duplicate gate ablation tests

**Why:** `tests/algo/test_gate_ablation.py` and `tests/algo/test_gates_ablation.py` overlap and both are `slow`.

**Action:** Replace both with a single fast test that uses `hippo_eval.bench.run_suite` (no model load).

**Edits:**

- Delete:
  - `tests/algo/test_gate_ablation.py`
  - `tests/algo/test_gates_ablation.py`

- Create `tests/algo/test_gate_ablation_fast.py`:

```python
from omegaconf import OmegaConf
from hippo_eval.bench import run_suite

def _run(preset: str, suite: str, ablate_key: str | None = None):
    cfg = OmegaConf.create({
        "preset": preset,
        "suite": suite,
        "n": 2,
        "seed": 1337,
        "outdir": None,
        "ablate": {},
    })
    if ablate_key:
        cfg.ablate[ablate_key] = False
    return run_suite(cfg)  # Bench returns dataclass with metrics/meta

def test_gating_can_be_disabled():
    # relational (semantic) gate
    res = _run("memory/sgc_rss", "semantic", "relational.gate.enabled")
    assert res.meta["gating_enabled"] is False

def test_gating_enabled_by_default():
    res = _run("memory/sgc_rss", "semantic", None)
    assert res.meta["gating_enabled"] is True
```

**Acceptance:**  
- New test passes in default `pytest -q`.  
- Gate toggle is covered without subprocess or model load.

---

## T3 — Convert E2E smoke to in-process, and shrink `n`

**Why:** `tests/algo/test_end2end_smoke.py` shells out and loads the model; it can be done in-process and faster.

**Edits:** Replace the test body with a `smoke`-marked variant using `hippo_eval.bench` and `n=2`:

```python
# tests/algo/test_end2end_smoke.py
import pytest
from omegaconf import OmegaConf
from hippo_eval.bench import run_suite, write_outputs

@pytest.mark.smoke
def test_end_to_end_smoke(tmp_path):
    # Baseline
    base = OmegaConf.create({"preset": "baselines/core", "suite": "episodic", "n": 2, "seed": 1337})
    res_base = run_suite(base)
    out = tmp_path / "baseline"
    write_outputs(res_base, out)
    assert (out / "metrics.json").exists()

    # Memory preset
    mem = OmegaConf.create({"preset": "memory/hei_nw", "suite": "episodic", "n": 2, "seed": 1337})
    res_mem = run_suite(mem)
    out2 = tmp_path / "memory"
    write_outputs(res_mem, out2)
    assert (out2 / "metrics.json").exists()

    # Basic sanity: EM present and numeric
    assert "episodic" in res_base.metrics
    assert "episodic" in res_mem.metrics
```

**Acceptance:**  
- Marked `smoke` and runs by default.  
- No subprocess, no HF model load, runtime under ~1s locally.

---

## T4 — Auto-mark and refactor CLI tests

**Why:** 14 files under `tests/cli/` spawn subprocesses, many refer to missing `scripts/…` in this ZIP.

**Action:**  
- Leave them in place, but rely on **T1 auto-marking** so they run only with `--runintegration`.
- Refactor *two* representative CLI tests to avoid `scripts/…` paths and use module entry points.

**Edits:**

1) Update `tests/cli/test_scripts_help.py` to use module `-m` instead of `scripts/*.py`:

```python
# tests/cli/test_scripts_help.py
import subprocess, sys

SCRIPTS = [
    ["-m", "hippo_eval.bench", "--help"],
    ["-m", "hippo_eval.datasets.cli", "--help"],
]

def test_cli_help():
    for args in SCRIPTS:
        subprocess.run([sys.executable, *args], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
```

2) Update `tests/cli/test_eval_dryrun.py` to call `hippo_eval.bench.run_suite` in-process (keep marked `integration` via folder rule):

```python
import json
from omegaconf import OmegaConf
from hippo_eval.bench import run_suite, write_outputs

def test_eval_dryrun(tmp_path):
    cfg = OmegaConf.create({"preset": "baselines/core", "suite": "episodic", "n": 2, "seed": 0})
    res = run_suite(cfg)
    write_outputs(res, tmp_path)
    data = json.loads((tmp_path / "metrics.json").read_text())
    assert data["metrics"]["episodic"]["pre_em"] >= 0.0
```

**Acceptance:**  
- `pytest -q --runintegration` passes these tests fast (<1s each).  
- No reliance on `scripts/` paths.

---

## T5 — Provide a global fast-LLM monkeypatch (optional speed-up)

**Why:** A few harness-level unit tests need an object with `.generate()` but do not care about actual LM quality.

**Edits:** Add to `tests/conftest.py`:

```python
# tests/conftest.py
import types, torch, pytest

@pytest.fixture(scope="session", autouse=True)
def fast_llm_monkeypatch(monkeypatch):
    class DummyTok:
        pad_token_id = 0
        eos_token_id = 0
        def __init__(self): pass
        def __call__(self, text, return_tensors=None, add_special_tokens=False):
            import torch
            ids = torch.tensor([[1,2,3]])
            attn = torch.ones_like(ids)
            return types.SimpleNamespace(input_ids=ids, attention_mask=attn)
        def decode(self, ids, skip_special_tokens=True):
            return "ok"

    class DummyLM(torch.nn.Module):
        def __init__(self): super().__init__()
        def generate(self, input_ids=None, attention_mask=None, max_new_tokens=32, **kw):
            # Return a tiny deterministic "completion"
            return input_ids

    # Patch only if env is set, so nightly runs can exercise the real model.
    import os
    if os.environ.get("FAST_TESTS", "1") != "1":
        return

    monkeypatch.setattr("hippo_eval.harness.runner._h.AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyTok()))
    monkeypatch.setattr("hippo_eval.harness.runner._h.AutoModelForCausalLM", types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyLM()))
```

**Acceptance:**  
- Harness-based unit tests no longer load HF weights when `FAST_TESTS=1` (default).  
- Nightly can unset the env to use real models.

---

## T6 — Add small, fast unit tests for harness helpers

**Why:** Replace slow coverage with fast, targeted tests.

**New tests:**

- `tests/eval/test_harness_metrics_fast.py` — verifies `harness.metrics.collect_metrics` shape and key presence for a tiny synthetic run result (`n=2`), no model load.
- `tests/eval/test_harness_io_fast.py` — round-trip `write_meta`, `write_csv`, `write_metrics` to a temp dir with a small metrics dict.

**Skeletons:**

```python
# tests/eval/test_harness_metrics_fast.py
from hippo_eval.harness.metrics import collect_metrics

def test_collect_metrics_shape():
    # minimal fake snapshots/diags
    snaps = []
    diags = {"pre": {"em": 1.0}, "post": {"em": 1.0}}
    g = {"episodic": {"attempts": 1, "grants": 1}, "relational": {"attempts": 0, "grants": 0}, "spatial": {"attempts": 0, "grants": 0}}
    m = collect_metrics("episodic", n=2, snapshots=snaps, store_diags=diags, gating=g)
    assert "pre_em" in m and "post_em" in m

# tests/eval/test_harness_io_fast.py
from hippo_eval.harness.io import write_meta, write_metrics
from pathlib import Path
import json

def test_write_read_roundtrip(tmp_path: Path):
    meta = {"suite": "episodic", "preset": "baselines/core", "n": 2}
    write_meta(tmp_path, meta)
    write_metrics(tmp_path, {"metrics": {"episodic": {"pre_em": 1.0}}})
    assert json.loads((tmp_path / "meta.json").read_text())["suite"] == "episodic"
```

**Acceptance:**  
- New tests pass in the default suite and run in milliseconds.

---

## T7 — Simplify CI workflow

**Why:** Many overlapping smoke steps waste compute and time.

**Edits:** Replace the test section of `.github/workflows/ci.yaml` with:

```yaml
    - name: Unit & Smoke
      run: |
        export FAST_TESTS=1
        pytest -q -m "not slow and not integration"

    - name: Ablation smoke
      run: python tests/fixtures/ci/ablation_smoke.py
```

And create a new `.github/workflows/nightly.yaml` (scheduled) that runs:

```yaml
name: Nightly
on:
  schedule:
    - cron: "0 2 * * *" # 02:00 UTC daily
jobs:
  nightly:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with: { python-version: "3.10" }
      - name: Install
        run: pip install -e . -r codex-env/requirements.txt
      - name: Integration (CLI/Hydra) + Slow
        run: |
          pytest -q --runintegration --runslow
```

**Acceptance:**  
- Default PR CI only runs unit + smoke + ablation smoke.  
- Nightly covers integration + slow.

---

## T8 — (Optional) Coverage guard

**Why:** Keep confidence high after slimming tests.

**Edits:** Add to `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["hippo_eval", "hippo_mem"]
branch = true

[tool.coverage.report]
fail_under = 70
skip_empty = true
show_missing = true
```

And in CI add a step:

```yaml
- name: Coverage
  run: |
    coverage run -m pytest -q -m "not slow and not integration"
    coverage report -m
```

**Acceptance:**  
- Coverage report prints and enforces threshold.
