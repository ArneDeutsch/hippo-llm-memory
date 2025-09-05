# Codex Follow‑ups — Ready‑to‑Run Tasks (pre‑EVAL rerun)

_Generated 2025-08-31 15:44 UTC_

These tasks finish the last mile before re‑running `EVAL_PROTOCOL.md`.

---

## T‑01 Wire spatial gate telemetry (blocker)
**Goal:** Increment gate counters for the spatial path just like episodic/relational.

**Code changes**
- File: `hippo_mem/spatial/gating.py` (or `hippo_mem/spatial/map.py` depending on where decisions are finalized)
  - Import: `from hippo_mem.common.telemetry import gate_registry`
  - In the code path where a spatial gate **decision** is made (e.g., `decide(prev_ctx, ctx, graph)`), add:
    - `stats = gate_registry.get("spatial")`
    - `stats.attempts += 1` for every decision attempt
    - If accepted for write: `stats.inserted += 1` (or `routed_to_episodic`/`aggregated` if your spatial design uses those semantics)
- Ensure this path is exercised in `harness._run_replay(...)` when spatial memory is enabled.

**Tests**
- Add: `tests/test_spatial_gate_telemetry.py`
  - Run a tiny spatial replay with `preset=configs/eval/memory/smpd.yaml`, `n=5`, `seed=1337` (with tiny model), assert:
    - `metrics["gates"]["spatial"]["attempts"] > 0`
    - `metrics["gates"]["spatial"]["accepts"] >= 0`

**Acceptance**
- `pytest -k spatial_gate_telemetry` passes.
- `scripts/smoke_n50.sh` produces non‑zero spatial gate attempts in metrics.

---

## T‑02 Guard baseline gating invariants (nice‑to‑have)
**Goal:** Baselines must show zero gate attempts across all memories even if code changes later.

**Code changes**
- Optional: at end of `evaluate(...)` in `harness.py`, when `is_memory_preset(cfg.preset)` is **False**, set `metrics["gates"]` to a zeroed snapshot to avoid accidental leakage.

**Tests**
- Extend `tests/test_baselines_have_no_memory.py` to assert `all(v["attempts"] == 0 for v in metrics["gates"].values())`.

**Acceptance**
- Test passes; reports show zeros for baselines.

---

## T‑03 Tiny docs nits (optional)
- In `EVAL_PROTOCOL.md`, add an explicit note: “Baselines MUST NOT be run with `persist=true`.”
- In `EVAL_PROTOCOL.md` telemetry invariants, add: “Spatial gate attempts must be > 0 on memory presets.”

---

## Done Criteria for this batch
- Spatial gate counters appear in `metrics.json` and in reports.
- Baselines remain zero‑telemetry for retrieval and gates.
- Smoke run (`scripts/smoke_n50.sh`) completes cleanly with strict telemetry on.

