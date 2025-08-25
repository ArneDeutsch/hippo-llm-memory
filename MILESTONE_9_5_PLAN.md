# Milestone 9.5 – Cross-session Consolidation
_Generated: 2025-08-25 11:33_

## 0) Goal (what we ship)
Demonstrate that memory modules persist across runs and improve **delayed recall** through replay. The harness gains
store persistence, a two-phase protocol (`teach → replay → test`), and stronger acceptance checks.

## 1) Gate (exit criteria)

- **Persistence**
  - Episodic, relational, and spatial stores expose `save(dir, session_id)` / `load(dir, session_id)`.
  - CLI flags `--store_dir`, `--session_id`, `--persist`, and `--mode={teach,replay,test}` are supported.
- **Replay policy & telemetry**
  - Scheduler implements `replay.policy={uniform,priority,spaced}`, `replay.rate`, `replay.noise_level`, and `replay.max_items`.
  - `metrics.json` records replay counts and gate counters; `retrieval.requests > 0` and refusal rate ≤0.5.
- **Delayed recall**
  - On `episodic@50`, test runs after a teach phase show **EM ≥ core + 0.20**.
- **Baselines & suites**
  - Baseline presets `baselines/longctx`, `baselines/rag`, and `baselines/span_short` run under the same protocol.
  - Minimal relational and spatial suites (`n=50`) execute with and without gates; results include ON→OFF deltas.
- **CI/Smoke**
  - Fast test reproduces a teach→test cycle and fails if telemetry fields are zero or refusal rate >0.5.
- **Docs**
  - `README.md`, `PROJECT_PLAN.md`, and `EVAL_PLAN.md` describe persistence flags, replay policies, and acceptance checks.

## 2) Work packages

1. **Store persistence API** – implement JSONL/Parquet backends; ensure thread-safe open/close.
2. **Harness modes** – extend `scripts/eval_model.py` to handle `teach`, `replay`, and `test` modes with persistence.
3. **Replay scheduler** – add policy selection and noise/interval controls.
4. **Gate instrumentation** – count write attempts, accepts/rejects, and retrieved-before/after gating.
5. **Evaluation suites** – provide `tasks/relational_50.jsonl` and `tasks/spatial_50.jsonl` mirroring episodic format.
6. **Baselines & decoding** – add `baselines/span_short.yaml`; ensure memory presets mirror the same decoding profile.
7. **CI & tests** – unit tests for store save/load round-trip, replay policy sampling, and refusal-rate guard.
8. **Documentation** – update high-level docs and milestone plans.

## 3) Notes
Cross-session results complement Milestone 9’s intra-session evaluations and pave the way for Milestone 10’s publication work.
