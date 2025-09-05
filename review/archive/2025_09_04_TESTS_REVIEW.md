# Test Suite Review — hippo-llm-memory

**Date:** 2025-09-04  
**Scope:** Inspect tests and CI in the provided repository ZIP; identify slow/overlapping tests; propose a lean, fast-by-default test strategy.

---

## TL;DR

- The suite contains **~142 test files**. A significant fraction call external CLIs via **`subprocess`** and/or invoke the heavy **Transformers model loader**, which dominates runtime.
- Only **one pytest marker** (`slow`) is currently registered. There is **no first-class “integration” or “smoke” marker**, so many CLI/e2e checks end up in the default selection (or get duplicated as shell steps in CI).
- We found **19 files marked `@pytest.mark.slow`**, **33 test files using `subprocess`**, and **17 files referring to non-existent `scripts/…` paths** in this ZIP (flaky in CI). There are **3 smoke-style e2e checks** spread across different places.
- The repository already ships a fast, LM-free harness in **`hippo_eval.bench`** (predictions == answers), which is ideal for _fast_ tests. We can migrate many CLI/e2e tests to that module or to in-process helpers in **`hippo_eval.harness`**.

---

## What makes tests slow

1. **Model load per test**  
   The real harness loads a HF model inside `hippo_eval.eval.harness` and `hippo_eval.harness.runner` every run. Even with the local **`models/tiny-gpt2/`**, initialization is the dominant cost.
2. **`subprocess` + Hydra process spin-up**  
   ~33 tests spawn Python processes to call `scripts/*.py` or `python -m …`. Process startup + Hydra config parsing adds up.
3. **Redundant e2e/smoke variants in CI**  
   The CI workflow (`.github/workflows/ci.yaml`) runs many nearly-identical “smoke” scripts (7b, 8a, 8, n50, n seeds, smoke eval) which heavily overlap in coverage.
4. **Script path skew**  
   In this ZIP there is no `scripts/` directory, yet 17 tests refer to `scripts/*.py`, guaranteeing failures unless CI injects those files. This is both brittle and slow.

---

## Existing fast building blocks we should lean on

- **`hippo_eval.bench`** — a light harness with a CLI (`python -m hippo_eval.bench`) and programmatic API (`run_suite`, `run_bench`, `run_matrix`), **no real LLM inference**. Perfect for plumbing and store/gating metrics.
- **In-process helpers** under `hippo_eval.harness.*` (e.g., `build_runner`, `run_suite`, `harness.metrics`, `harness.io`) for **unit-level** coverage.
- **Focused algo tests** (telemetry, gating, retrieval) that don’t need the LLM and already run fast.

---

## Concrete findings (from the ZIP)

- **Markers**: only `slow` is registered in `pyproject.toml`.  
- **`tests/conftest.py`** adds `--runslow` and skips `slow` by default — good, but **we need `integration` and `smoke`** too.
- **CLI-style tests** (in `tests/cli/`) and several `tests/algo/*` call `scripts/*.py` via `subprocess`. These should be:
  - migrated to programmatic calls into `hippo_eval.bench` or `hippo_eval.harness`, or
  - auto-marked `integration` and skipped by default.
- **Overlap**: `tests/algo/test_gate_ablation.py` and `tests/algo/test_gates_ablation.py` exercise nearly identical gating toggles; keep one canonical test.
- **E2E smoke**: `tests/algo/test_end2end_smoke.py` executes the full harness with `models/tiny-gpt2` and `n=5` via `subprocess`. This can be:
  - converted to programmatic calls, and
  - reduced to `n=2`, or
  - reimplemented against `hippo_eval.bench` for speed (still validates stores/gating/metrics).

Counts observed in the ZIP:
- **~142** files under `tests/`
- **19** files marked `slow`
- **33** files use `subprocess`
- **17** files reference `scripts/…` paths
- **3** smoke/e2e style checks scattered across locations

---

## Proposed strategy

1. **Three tiers**
   - **Unit (default)** — pure-Python, in-process, **no model load**, no subprocess. Should cover >80% of code paths.
   - **Smoke (default)** — 1–2 minimal end-to-end tests using `hippo_eval.bench` (or `harness` with a mock model) and `n≤2` to validate wiring, file layout, basic telemetry.
   - **Integration (opt-in)** — CLI/Hydra & real-harness invocations; run *only* on nightly or with a flag.

2. **Marking & collection**
   - Add PyTest markers: `integration`, `smoke` (and keep `slow`).
   - Update `conftest.py` to **auto-mark everything under `tests/cli/` as `integration`** and to skip `integration` by default unless `--runintegration` is passed.
   - Mark only **one** smoke test in e2e category; keep `tests/fixtures/ci/ablation_smoke.py` as a single ablation sanity check.

3. **Make tests faster**
   - **Remove `subprocess`** where possible — call `hippo_eval.bench.run_suite` or `hippo_eval.harness.run_suite` directly.
   - **Shrink `n` to 1–2** in smoke paths; unit tests should operate on tiny fixtures.
   - For the handful of harness-level unit tests that still need a model object, **monkeypatch `AutoModelForCausalLM` + `AutoTokenizer`** with a trivial mock once per session to avoid weight loads.

4. **CI simplification**
   - Replace 6+ “Smoke …” steps with **one `pytest -m "not slow and not integration"` step + one ablation check**.
   - Add a **nightly Integration** workflow to run `-m "integration"` and (optionally) `--runslow`.

---

## Expected impact

- Default CI runtime **drops sharply** by avoiding repeated HF model loads and extra processes.
- Failures become **less flaky** (no missing `scripts/`), and test intent is clearer (unit vs smoke vs integration).
- Coverage **stays high** because we lean on `hippo_eval.bench` + unit helpers instead of deleting checks.
