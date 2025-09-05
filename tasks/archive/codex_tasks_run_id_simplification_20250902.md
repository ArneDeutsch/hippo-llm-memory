# Codex Task List — Simplify RUN_ID (remove DATE & digits fallback)
**Scope:** Replace the current RUN_ID/DATE mix with a single canonical **RUN_ID** that is **used verbatim** across the pipeline (baselines, memory runs, stores, reports). Remove underscore→digits normalization and all usages of `DATE`. Provide migration tooling and update docs/tests/CI.

**Target date:** ASAP  
**Owner:** Codex

---

## Guiding principles

- **Single source of truth:** `RUN_ID` is a slug used **unchanged** as the directory name under `runs/<RUN_ID>/`, `reports/<RUN_ID>/`, etc.
- **No silent fallbacks:** If `RUN_ID` is not provided, commands should either (a) use a safe explicit default (`dev`) **or** (b) fail with a helpful message (prefer explicit failure for reproducibility).
- **No implicit normalization:** Stop stripping underscores; stop “digits-only” fallbacks.
- **CLI override wins:** When provided both in env and on CLI, the CLI `run_id=...` must take precedence (append it last in overrides).

Slug policy: `^[A-Za-z0-9_-]{3,64}$`.

Breakage policy: keep a **one-shot migration helper**; after that, the code path is simplified (no compatibility probes).

---

## Task 1 — Define canonical RUN_ID contract (validation helper)
**Files:** `hippo_mem/utils/ids.py` (new), `hippo_mem/utils/__init__.py`  
**Change:** Add a tiny validator and normalizer (no mutation, only validation and error messages).

```python
# hippo_mem/utils/ids.py
import re

SLUG_RE = re.compile(r"^[A-Za-z0-9_-]{3,64}$")

def validate_run_id(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("RUN_ID must be a string")
    if not SLUG_RE.fullmatch(value):
        raise ValueError("Invalid RUN_ID. Use 3–64 chars from [A-Za-z0-9_-].")
    return value
```

**Acceptance:** unit test `tests/test_ids.py` covers valid/invalid examples.

---

## Task 2 — Remove DATE from config and env
**Files:** `configs/eval/default.yaml`, `scripts/_env.sh`, `docs/EVAL_PROTOCOL.md`, root `EVAL_PROTOCOL.md`

- `configs/eval/default.yaml`:
  - Replace:
    ```yaml
    run_id: ${oc.env:RUN_ID,${oc.env:DATE,null}}
    date: ${oc.env:DATE,null}
    ```
    with
    ```yaml
    run_id: ${oc.env:RUN_ID,null}
    ```
  - Remove the `date:` field entirely.

- `scripts/_env.sh`:
  - Replace current RUN_ID/DATE bootstrap with an **explicit requirement** (or default):
    ```bash
    #!/usr/bin/env bash
    set -euo pipefail
    : "${RUN_ID:?Set RUN_ID (simple slug like 'dev' or 'exp_42')}"
    RUNS="runs/$RUN_ID"
    REPORTS="reports/$RUN_ID"
    STORES="$RUNS/stores"
    ADAPTERS="adapters/$RUN_ID"
    : "${MODEL:=Qwen/Qwen2.5-1.5B-Instruct}"
    export MODEL HF_MODEL_PATH="$MODEL"
    HEI_SESSION_ID="hei_${RUN_ID}"
    mkdir -p "$RUNS" "$REPORTS" "$STORES" "$ADAPTERS"
    ```
    > If you prefer a default rather than strict requirement, simply set `export RUN_ID=dev` before sourcing.

- Docs (`docs/EVAL_PROTOCOL.md` and root `EVAL_PROTOCOL.md`):
  - Remove all references to DATE and “digits-only” fallbacks.
  - Update “Choosing a run identifier” to:  
    ```bash
    export RUN_ID=my_experiment   # slug [A-Za-z0-9_-], 3–64 chars
    source scripts/_env.sh
    ```
  - Clarify: “All outputs go to `runs/$RUN_ID/…`, no normalization happens.”

**Acceptance:** `grep -R "DATE" .` only finds mentions in CHANGELOG or historical notes.

---

## Task 3 — Make preflight check exact (no digits-only)
**Files:** `hippo_mem/eval/harness.py`, `tests/test_preflight.py`

- In `preflight_check()`, replace the baselines probe with a single path:
  ```python
  rid = str(cfg.get("run_id"))
  baseline = Path("runs") / rid / "baselines" / "metrics.csv"
  if not baseline.exists():
      failures.append(f"missing baseline metrics: {baseline} — generate via:\n  "
                      f"python scripts/run_baselines.py --run-id {rid}")
  ```

- Update `tests/test_preflight.py`:
  - Remove the test that accepts digits-only baselines.
  - Keep/adjust tests that require exact match under `runs/<RUN_ID>/`.

**Acceptance:** tests pass; `failed_preflight.json` only lists the exact path.

---

## Task 4 — Stop mutating `store_dir`/`run_id` in the harness shims
**Files:** `scripts/eval_model.py`, `hippo_mem/eval/harness.py`, `scripts/eval_cli.py`, `scripts/store_paths.py`, `hippo_mem/utils/stores.py`, `tests/test_store_paths.py`

- **General rule:** do **not** infer/alter `run_id`; do **not** append algo to `store_dir` unless it’s explicitly missing and we have to select an algo (keep this behavior if already relied upon).

- In `scripts/eval_model.py` main wrapper, ensure any resolution for `store_dir` does **not** change `run_id`. Preserve existing logic that maps `store_dir` ⇒ algo dir but **remove** any underscore/digits normalization.

- In `scripts/eval_cli.py`:
  - Add an explicit `--run-id` argument (optional). If provided, append `run_id=<value>` **after** other overrides so CLI wins.
  - Echo the resolved `run_id` for traceability when `--verbose` is set.

- In `hippo_mem/eval/harness.py` `evaluate()`:
  - Ensure we never re-derive an alternate RID. Only use `cfg.run_id` verbatim.

**Acceptance:** `tests/test_store_paths.py` still passes; manual runs show identical `runs/<RUN_ID>/…` layout irrespective of underscores/digits composition.

---

## Task 5 — Baselines entry points: require `--run-id`, remove `--date`
**Files:** `scripts/run_baselines.py`, `hippo_mem/eval/baselines.py`, `docs/baselines.md`

- Remove `--date` option and any transformations of RID.
- Validate `--run-id` with `validate_run_id()`.
- Write output strictly under `runs/<RUN_ID>/baselines/` and `runs/<RUN_ID>/presets/...` etc.

**Acceptance:** Running:
```bash
python scripts/run_baselines.py --run-id my_exp
```
creates `runs/my_exp/baselines/metrics.csv` and no other variant.

---

## Task 6 — Remove digits-only logic from tooling
**Files:** `scripts/tools/triage_preflight.sh`, `scripts/smoke_*.sh`

- `triage_preflight.sh`:
  - Delete the section that computes `DIGITS="${RID//_/}"` and probes `runs/$DIGITS/baselines/...`.
  - Keep the rest (meta/store checks).

- `scripts/smoke_*.sh`:
  - Ensure each script sets `RUN_ID` explicitly and does not reference `DATE`.

**Acceptance:** `grep -R DIGITS scripts` returns nothing. Smoke scripts run green with a static `RUN_ID`.

---

## Task 7 — Tests & fixtures update
**Files:** `tests/test_preflight.py`, `tests/test_store_resolution.py`, `tests/test_aggregator.py` (if they assert digits fallback), `tests/conftest.py`

- Remove/adjust any assertions that expect digits-only acceptance.
- Add `tests/test_ids.py` for the new validator (Task 1).

**Acceptance:** `make test` passes locally; CI green.

---

## Task 8 — Documentation sweep
**Files:** `README.md`, `EVAL_PROTOCOL.md`, `docs/EVAL_PROTOCOL.md`, `docs/baselines.md`, any `review/*.md` that still prescribes DATE/digits.

- Replace examples to always use a simple `RUN_ID` slug (e.g., `dev`, `m9_eval`, `exp_42`).
- Remove references to DATE and digits-only fallbacks.
- Add a short “RUN_ID contract” box.

**Acceptance:** `grep -R "digits-only\|DATE" docs README.md EVAL_PROTOCOL.md` returns nothing relevant.

---

## Post-merge smoke checklist

- `RUN_ID=dev` end-to-end (baselines → memory teach → replay → test → report) works.
- `failed_preflight.json` lists only *one* baseline path candidate (no digits variant).
- `scripts/tools/triage_preflight.sh` shows clear diagnostics without digits guesses.
- CI job `ci_smoke_eval.sh` passes.

---

## Notes / Risks

- **Breaking change:** old runs named in digits-only form will not be discovered by preflight anymore. Either regenerate baselines or run the migration helper once.
- **Safety:** `validate_run_id()` prevents path traversal and weird Unicode—keep the slug rule strict.
- **Reproducibility:** Prefer explicit failure if `RUN_ID` is missing. It nudges users into naming runs deliberately.
