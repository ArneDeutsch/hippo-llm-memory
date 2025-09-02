# Codex Task Pack — Run-ID unification, Protocol Hardening, and Baseline Robustness

**Date:** 2025-09-02

> Goal: Eliminate RUN\_ID/date mismatches and Hydra quoting pitfalls, harden the pipeline, and update docs/tests accordingly. Each task is **atomic**, has **acceptance criteria**, and includes **how to test**.

---

## T1 — Introduce `run_id` as the sole IO key; deprecate `date` for IO

**Files**

* `hippo_mem/eval/harness.py`
* `scripts/eval_model.py` (CLI plumbing if present)
* `scripts/run_baselines.py`
* `scripts/report.py`
* `scripts/_env.sh`
* `configs/eval/default.yaml` (if it currently defines `date`)

**Changes**

1. Add a new config/CLI field `run_id: str` (required for pathing).
2. Enforce slug validation: `^[A-Za-z0-9._-]{3,64}$`. If invalid, raise `ValueError` with guidance.
3. Replace all occurrences of `runs/<date>/...` and `reports/<date>/...` with `runs/<run_id>/...` and `reports/<run_id>/...`.
4. Keep `date` (if provided) only as metadata in emitted JSON/markdown; it must **not** influence file paths.
5. Back-compat: if `run_id` is missing and `date` is provided, derive `run_id = _date_str(date)` and emit a **deprecation warning**.

**Acceptance**

* Running with `--run-id X` writes all outputs under `runs/X` and `reports/X`.
* Running with only `--date 20250902` still works, but logs a visible deprecation warning.

**How to test**

```bash
pytest -q tests/test_run_id_paths.py::test_paths_use_run_id
pytest -q tests/test_run_id_paths.py::test_date_backcompat_warns
```

---

## T2 — Add type guards for Hydra lists (presets, tasks, n\_values, seeds)

**Files**

* `hippo_mem/eval/harness.py`

**Changes**

* Before iterating, validate each field is a sequence (list/tuple). If it’s a string, raise:

  * `TypeError("presets must be a list: presets=[a,b], not a quoted string")`
  * Same for `tasks`, `n_values`, `seeds`.

**Acceptance**

* Quoted lists cause immediate, actionable failure.
* Unquoted lists proceed normally.

**How to test**

```bash
pytest -q tests/test_input_validation.py::test_reject_quoted_presets
pytest -q tests/test_input_validation.py::test_accept_list_presets
```

---

## T3 — Update aggregator to use `run_id` and improve diagnostics

**Files**

* `scripts/run_baselines.py`

**Changes**

1. Replace `--date` with `--run-id` (support `--date` as deprecated alias that maps to `run_id = _date_str(date)` and warns).
2. Build `root = runs/<run_id>/baselines`.
3. If no metrics found, raise:
   `FileNotFoundError(f"no baseline metrics under {root}; found: {', '.join(sorted(p.name for p in (root.parent).glob('*') if (root.parent/p.name/'baselines').exists())) or '<none>'}")`
4. Add a count summary on success and write a `baselines_ok.flag`.

**Acceptance**

* `python scripts/run_baselines.py --run-id TEST123` searches exactly `runs/TEST123/baselines`.
* Error message lists nearby available run\_ids.

**How to test**

```bash
pytest -q tests/test_aggregator.py::test_aggregator_uses_run_id
pytest -q tests/test_aggregator.py::test_aggregator_error_lists_candidates
```

---

## T4 — Report tool follows `run_id`

**Files**

* `scripts/report.py`

**Changes**

* Replace `--date` with `--run-id` (same back-compat shim as T3).
* Read from `runs/<run_id>` and write to `reports/<run_id>`.

**Acceptance**

* Reports land in `reports/<run_id>/index.md`.

**How to test**

```bash
pytest -q tests/test_report.py::test_report_paths_run_id
```

---

## T5 — Preflight error hint

**Files**

* `hippo_mem/eval/harness.py`

**Changes**

* When detecting missing `baselines/metrics.csv`, include an actionable hint:

  * `"... missing baseline metrics: runs/<run_id>/baselines/metrics.csv — generate via: python scripts/run_baselines.py --run-id <run_id> ..."`

**Acceptance**

* Error contains the exact command with the active `run_id`.

**How to test**

```bash
pytest -q tests/test_preflight_gate.py::test_preflight_missing_baseline_hints_command
```

---

## T6 — Make “baselines in teach mode” robust (optional but recommended)

**Files**

* `hippo_mem/eval/harness.py`

**Changes**

* If `preset` begins with `baselines/` and `mode=teach`, set `compute_metrics=True` so `metrics.json` includes `pre_*`.

**Acceptance**

* Baseline runs in teach mode still produce usable metrics.

**How to test**

```bash
pytest -q tests/test_baselines.py::test_baseline_metrics_in_teach
```

---

## T7 — Update EVAL protocol and README

**Files**

* `EVAL_PROTOCOL.md`
* `README.md`

**Changes**

* Replace all uses of `date` with `run_id` in commands.
* Ensure all Hydra lists are unquoted.
* Add a red **MUST** box about using `run_id` consistently and list the allowed slug regex.
* Include a minimal end-to-end example with `RUN_ID=20250902_50_1337_2025`.

**Acceptance**

* Copy-&-paste works end-to-end; baselines roll-up produced; no guardrails tripped.

**How to test**

* Manually run the protocol in CI (see T9).

---

## T8 — Tests for the new schema

**Files**

* `tests/test_run_id_paths.py` (new)
* `tests/test_input_validation.py` (new)
* `tests/test_aggregator.py` (new or updated)
* `tests/test_report.py` (updated)
* `tests/test_preflight_gate.py` (updated)

**Changes**

* Add tests described in T1–T6.

**Acceptance**

* `pytest` suite passes locally.

---

## T9 — CI pipeline hook to prevent regressions

**Files**

* `.github/workflows/ci.yml` (or your CI)
* `scripts/ci_smoke_eval.sh` (new)

**Changes**

* Add a smoke job that runs:

  1. Matrix baselines (`run_id=ci_smoke`)
  2. Aggregation (`--run-id ci_smoke`)
  3. One memory suite teach→replay→test with strict telemetry
  4. Report generation
* Fail CI if `baselines/metrics.csv` is missing or any `failed_preflight.json` exists.

**Acceptance**

* CI fails on any reintroduction of date/run\_id mismatch or missing roll-up.

**How to test**

* Push a PR that intentionally uses `--date`; CI should warn/fail.

---

## T10 — Cleanup utility for stray outputs

**Files**

* `scripts/cleanup_runs.py` (new)

**Changes**

* Provide a tool to list and optionally remove suspicious directories (single-char names, unmatched to slug regex).

**Acceptance**

* Running without `--force` only lists; with `--force` removes safely.

**How to test**

```bash
python scripts/cleanup_runs.py --list
python scripts/cleanup_runs.py --force
```

---

## T11 — Migration note & deprecation window

**Files**

* `CHANGELOG.md` (new)
* `README.md` (update)

**Changes**

* Document the `run_id` transition, the deprecation of `--date` for IO (kept as metadata), and the cut-off version/date.

**Acceptance**

* Users are informed and the path forward is unambiguous.

---

### Appendix — Reference diff snippets

**Hydra list guards**

```diff
# hippo_mem/eval/harness.py
+def _ensure_list(name, val):
+    if isinstance(val, str):
+        raise TypeError(f"{name} must be a list: use {name}=[a,b], not a quoted string")
+    return val

def main(cfg):
-    presets = cfg.get("presets")
+    presets = _ensure_list("presets", cfg.get("presets"))
-    tasks = cfg.get("tasks")
+    tasks = _ensure_list("tasks", cfg.get("tasks"))
```

**Run-ID validation**

```diff
SLUG_RE = re.compile(r"^[A-Za-z0-9._-]{3,64}$")
run_id = cfg.get("run_id")
if not run_id and cfg.get("date"):
    run_id = _date_str(cfg.get("date"))
    log.warning("`date` is deprecated for IO; using run_id=%s", run_id)
if not run_id or not SLUG_RE.match(run_id):
    raise ValueError("run_id must match ^[A-Za-z0-9._-]{3,64}$")
root_outdir = Path("runs") / run_id
```

These tasks, once implemented, will lock the system onto a single, simple naming scheme and eliminate the class of failures we’ve been chasing.

