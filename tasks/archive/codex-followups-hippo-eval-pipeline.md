# Follow‑up Codex Tasks — Hippo‑LLM Evaluation Pipeline
_Generated: 2025-08-30 15:47 UTC_

These tasks close the remaining operational gaps so the improved pipeline yields **meaningful** validation data by default. Each task includes file paths, concrete steps, and acceptance criteria.

---

## T1 — Enforce strict telemetry invariants in protocol runs
**Why:** Prevent inflated hit‑rates from slipping through when a store is empty or irrelevant.

**Changes**
- Update **EVAL_PROTOCOL.md** memory runs to pass `--strict-telemetry` via `scripts/eval_cli.py` (teach/test/replay phases).
- Update **scripts/smoke_n50.sh** to add `--strict-telemetry` on all `eval_cli.py` invocations.

**Acceptance**
- Running the smoke script with an empty store fails with a clear telemetry error (e.g., `hits > total_k` or mismatch) and passes after a normal run.
- `reports/<RUN_ID>/*/summary.md` shows non‑inflated `hit_rate_at_k` values.

---

## T2 — Make difficult profiles the explicit default for saturating suites
**Why:** Avoid EM(norm)≈1.0 on `episodic_cross` and `episodic_capacity` so uplift is measurable.

**Changes**
- In **EVAL_PROTOCOL.md**, inline the recommended flags where they are needed, e.g.:  
  `suite=episodic_cross dataset_profile=hard` and `suite=episodic_capacity dataset_profile=hard`.
- Add a short **“Profiles”** box in EVAL_PROTOCOL’s prelude with a table mapping suites → recommended profile.

**Acceptance**
- Fresh runs on `baselines/core` with `n=50, seed=1337` yield EM(norm) **< 0.95** for `episodic_cross` and `episodic_capacity`.

---

## T3 — Guarantee per‑suite `post_*` and `delta_*` metrics (and assert in reports)
**Why:** Some older artefacts had only a top‑level `delta` block.

**Changes**
- In **hippo_mem/eval/harness.py**, ensure per‑suite `post_*` and `delta_*` keys are always written on replay to the **same** `outdir` as `pre`.
- In **scripts/report.py**, add a guard: if a suite lacks any `post_*` keys, print a clear error and exit non‑zero (behind `--strict` flag).
- Extend **tests/test_metrics_post_delta.py** to also assert `delta_*` keys (not just `delta_em`).

**Acceptance**
- `scripts/smoke_n50.sh` passes and `jq` finds both `post_em` and `delta_em` in all memory suite metrics.
- `scripts/report.py --strict --date <RUN_ID>` fails if any suite is missing `post_*`.

---

## T4 — Clean repository of stale artefacts to avoid confusion
**Why:** Old `runs/` and `reports/` mask whether fixes are effective.

**Changes**
- Remove `runs/` and `reports/` completely.
- Add a `make clean-runs` target to remove `runs/*` and `reports/*`.
- Keep tiny fixtures under `tests/data/` only.

**Acceptance**
- A fresh clone has no run/report artefacts; CI still passes using generated data.

---

## T5 — Clarify CI baselines vs. model baselines
**Why:** `scripts/run_baselines_bench.py` leverages the bench harness, which is great for CI but not for model‑backed baselines.

**Changes**
- Rename the CI helper to **scripts/run_baselines_bench.py** and update references in docs that point to CI usage.
- Keep **EVAL_PROTOCOL.md** baseline examples on `scripts/eval_model.py` with `model=$MODEL` to avoid ambiguity.

**Acceptance**
- Searching for the old script name in docs returns no hits (rename complete).
- No user attempts to run bench baselines for scientific comparisons.

---

## T6 — Integrate store layout validation into replay steps
**Why:** Prevent “missing store” errors during HEI‑NW/other replays.

**Changes**
- In **EVAL_PROTOCOL.md** memory phases, insert a step:  
  `python scripts/validate_store.py --algo hei_nw` (and equivalents for other modules) before replay.
- Add this check to **scripts/smoke_n50.sh** too.

**Acceptance**
- Replay phases fail early with a clear message if the expected `runs/$RUN_ID/stores/<algo>/<session>` structure is missing.

---

## T7 — Report should always include an Uplift section
**Why:** Make uplift visible even for a single seed (CI/quick runs).

**Changes**
- In **scripts/report.py**, unconditionally render an **“## Uplift”** section; when only one seed is available, show `± 0.000` and include a note “single‑seed run: CI bands unavailable”.

**Acceptance**
- `reports/<RUN_ID>/<suite>/summary.md` includes an Uplift section in all suites.

---

## T8 — Protocol: make “hard profile + strict telemetry” copy‑pasteable
**Why:** Reduce operator error.

**Changes**
- Add a ready‑to‑paste block in **EVAL_PROTOCOL.md**:
  ```bash
  # Recommended flags for sensitive suites
  suite=episodic_cross dataset_profile=hard --strict-telemetry
  suite=episodic_capacity dataset_profile=hard --strict-telemetry
  ```

**Acceptance**
- Reviewers can execute the block verbatim; resulting reports show non‑saturated metrics and valid telemetry.

