# Hippo‑LLM Evaluation Pipeline — Deep Analysis (n=50, seed=1337)
_Generated: 2025-08-30 15:47 UTC_

## Context
You asked for a thorough check that the tasks in `tasks/evaluation_pipeline_improvements.md` actually mitigate the issues you documented in `review/hippo-pipeline-review_n50_seed1337.md` and that the resulting data is meaningful for validating the three hippocampus‑inspired algorithms (HEI‑NW, SGC‑RSS, SMPD). I inspected the ZIP repository, relevant scripts, tests, and the existing run artefacts under `runs/` and `reports/` that accompany the ZIP.

## Executive summary
**Good news:** Most improvements are **implemented** (code + tests) and are sufficient *if the updated pipeline is run end‑to‑end*. However, the artefacts included in the ZIP (`runs/data50`, `reports/*`) look **stale** (generated before the fixes), so they still exhibit the original problems (e.g., episodic hit‑rate **1.000**, no `post_*` in suite metrics, missing **Uplift** sections, saturated suites). Re‑running with the updated protocol (using the `hard` dataset profiles where indicated, and enabling strict telemetry) should produce meaningful validation data.

**Verdict:** Implementation quality is **good**; **remaining risk is operational** (protocol defaults, old artefacts in repo, a couple of small consistency gaps). See “Gaps & follow‑ups” at the end.

---

## Issue‑by‑issue assessment vs. the improvement tasks

Below I track each task from `tasks/evaluation_pipeline_improvements.md` and judge (1) whether it’s implemented, (2) evidence, and (3) whether it suffices to mitigate the linked issue in your review.

### 1) Fix episodic retrieval hit‑rate inflation
- **Status:** **Implemented** (code + tests).
- **Evidence:** New telemetry layer `hippo_mem/common/telemetry.py` with invariants and `record_stats(...)` called from the common retrieval path `hippo_mem/common/retrieval.py`. Unit tests exist: `tests/test_telemetry_sanity.py` and `tests/test_episodic_retrieval.py`.
- **Mitigation sufficiency:** Yes, provided you **re‑run**. The checked‑in reports still show `hit_rate_at_k = 1.000` (e.g., `reports/20250829_1307/episodic/summary.md`), which indicates they were generated **before** this fix. With the current code, empty/irrelevant stores should log `hits=0 → hit_rate_at_k=0.0` while still returning placeholder tokens.

### 2) Make consolidation uplift gate configurable + CI‑based
- **Status:** **Implemented**.
- **Evidence:** `scripts/test_consolidation.py` exposes `--uplift-mode` (`threshold|ci`), `--min-uplift`, and `--alpha`. CI tests in `tests/test_ci_guardrails.py` cover pass/fail logic.
- **Mitigation sufficiency:** Yes. This addresses your “+0.20 uplift gate is too aggressive” point by making it tunable and CI‑friendly.

### 3) Ensure replay writes `post_*` and `delta_*` metrics
- **Status:** **Partially implemented** in artefacts; **implemented** in code + tests.
- **Evidence:** Harness computes and writes both `post_*` and `delta_*` (see `hippo_mem/eval/harness.py`, `scripts/eval_model.py`). Test `tests/test_metrics_post_delta.py` asserts both keys exist in per‑suite metrics. **However**, the run artefacts inside the ZIP (e.g., `runs/data50/consolidation/post/metrics.json`) contain a top‑level `"delta": {...}` block but **no per‑suite `post_*` fields**. The presence of the test suggests the code now writes them; the artefacts are likely pre‑fix.
- **Mitigation sufficiency:** Yes once re‑run. Also see follow‑up to **enforce** the presence of per‑suite `post_*`/`delta_*` in the report step.

### 4) Report uplift tables with CIs and plots
- **Status:** **Implemented** (code), but **not present in the checked‑in reports**.
- **Evidence:** `scripts/report.py` renders per‑suite **Uplift** sections (pre/post/Δ with 95% CI) and saves `uplift.png`. The included reports under `reports/*` **lack** an “## Uplift” section, confirming they predate the change.
- **Mitigation sufficiency:** Yes after regeneration.

### 5) Standardize RUN_ID prelude + stable session IDs
- **Status:** **Implemented**.
- **Evidence:** `scripts/_env.sh` defines a consistent `RUN_ID` → `DATE` mapping, derives `$RUNS/$REPORTS/$STORES`, and exports `HEI_SESSION_ID`. `scripts/store_paths.py` centralizes session/dir derivation; used by `scripts/validate_store.py`.
- **Mitigation sufficiency:** Yes; resolves path mismatches you saw with HEI‑NW.

### 6) Add dataset difficulty profiles to avoid saturation
- **Status:** **Implemented** (generators + docs), **not applied** in included runs.
- **Evidence:** `configs/eval/default.yaml` includes `dataset_profile`, and `hippo_mem/eval/harness.py` threads it through. `EVAL_PROTOCOL.md` docs explicitly instruct using `dataset_profile=hard` for `episodic_cross` and `episodic_capacity`. The included reports show saturation (e.g., `episodic_cross`: EM(norm) **1.0** baseline), indicating the runs used the `base` profile.
- **Mitigation sufficiency:** Yes if you run the **hard** profile for those suites.

### 7) Gate ON/OFF ablation + reporter section
- **Status:** **Implemented**.
- **Evidence:** Ablation runs exist under `runs/data50/ablate/*` and surface in `reports/*/index.md`. Gate telemetry sections appear in suite summaries.
- **Mitigation sufficiency:** Yes.

### 8) Use EM(raw) as primary metric for `semantic`; keep both
- **Status:** **Implemented**.
- **Evidence:** `scripts/report.py` sorts **semantic** summaries by EM(raw) and displays both EM(raw) and EM(norm), per your rationale.
- **Mitigation sufficiency:** Yes; avoids normalization masking.

### 9) Tighten telemetry sanity checks
- **Status:** **Implemented**.
- **Evidence:** `hippo_mem/common/telemetry.py` with strict mode; tests in `tests/test_telemetry_sanity.py` cover invariants (`hits<=total_k`, hit‑rate consistency, etc.).
- **Mitigation sufficiency:** Yes; add usage in protocol (see follow‑ups).

### 10) Reporter: Retrieval section reflects fixed semantics
- **Status:** **Implemented** (code).
- **Evidence:** `scripts/report.py` explains that “cue‑only fallbacks are excluded from telemetry” and renders `requests|hits|hit_rate_at_k|tokens_returned|avg_latency_ms` pulled from the telemetry snapshots.
- **Mitigation sufficiency:** Yes; regenerate reports after re‑run.

### 11) Smoke E2E script for n=50, seed=1337 (pre→replay→report)
- **Status:** **Implemented**.
- **Evidence:** `scripts/smoke_n50.sh` runs baselines + memory (teach/test/replay) and asserts **both** `post_em` and `delta_em` exist using `jq`. It finishes by calling `scripts/report.py`.
- **Mitigation sufficiency:** Yes for CI and quick local verification.

### 12) Store/session layout helper + validator
- **Status:** **Implemented** (and tested).
- **Evidence:** `scripts/store_paths.py`, `scripts/validate_store.py`, and tests `tests/test_validate_store.py` ensure the expected `runs/$RUN_ID/stores/<algo>/<session>/` layout exists before replay.
- **Mitigation sufficiency:** Yes; addresses the earlier HEI‑NW path mismatch.

### 13) Documentation updates (protocol & plan)
- **Status:** **Implemented**.
- **Evidence:** Updated `EVAL_PROTOCOL.md` and `EVAL_PLAN.md` (+ `docs/` variants) include the RUN_ID prelude, dataset profiles, full run matrices, ablations, and consolidation steps.
- **Mitigation sufficiency:** Yes; they’re clear and actionable.

### 14) (Optional) Bootstrap CI over seeds
- **Status:** **Implemented** (tests) and **documented**.
- **Evidence:** `scripts/test_consolidation.py` exposes CI mode; tests in `tests/test_ci_guardrails.py` cover it; docs mention the CI path.
- **Mitigation sufficiency:** Yes when running ≥2 seeds.

---

## Artefact spot‑checks (from ZIP) vs. expectations

- **Retrieval telemetry (episodic):** `hit_rate_at_k=1.000` still present in `reports/20250829_1307/episodic/summary.md` → **stale**. With new code, this should be ≤ observed recall and be **0.0** when no traces are recalled.
- **Replay metrics:** Example `runs/data50/consolidation/post/metrics.json` has a top‑level `"delta"` but **no per‑suite `post_*`** keys → **stale** relative to the code & tests.
- **Uplift sections/plots:** Missing in `reports/*` → **stale**.
- **Suite saturation:** `episodic_cross` and `episodic_capacity` show EM(norm) near or at 1.0 for baselines → must switch to `dataset_profile=hard` as the docs recommend.

---

## Gaps & follow‑ups (what to change to get meaningful validation *by default*)
1) **Enforce strict telemetry during runs.** The code supports strict mode; make the protocol pass `--strict-telemetry` via `scripts/eval_cli.py` for memory presets so inflated hit‑rates fail fast.
2) **Default the difficult profiles where needed.** In `EVAL_PROTOCOL.md`, put concrete call examples that pass `dataset_profile=hard` for `episodic_cross`/`episodic_capacity` (not just a note).
3) **Guarantee per‑suite `post_*`/`delta_*` are written.** The harness implements this and tests assert it; ensure the protocol’s replay step always calls the same outdir as `pre` and that the report step **fails** if any suite lacks `post_*` (the smoke script already does this—extend that check to the main protocol too).
4) **Remove stale run artefacts from the repo.** Add `runs/` and `reports/` to `.gitignore` and ship only a tiny fixture under `tests/data/` to avoid mixing old outputs with new ones.
5) **Name clarity for CI baselines.** `scripts/run_baselines.py` currently calls the *bench* harness (suitable for CI), which can confuse readers who expect real models. Consider renaming to `run_baselines_bench.py` and keep `EVAL_PROTOCOL.md` on `scripts/eval_model.py` for true baselines.

See the companion “Codex follow‑up tasks” document for concrete, ready‑to‑execute tasks covering the points above.
