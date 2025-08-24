# Hippo‑LLM Memory — Unified Refactoring Plan (v1.0)

_Date:_ 2025-08-24

A **consistent, low‑risk plan** that aligns with `research/experiment-synthesis.md` (ground truth) and `DESIGN.md`, and matches the current code layout under `hippo_mem/*`. The plan emphasizes **deduplication**, **API coherence**, **observability**, and **safe, incremental rollout** with strict test parity and performance guardrails.

**Scope.** Applies to the three memory systems (HEI‑NW / episodic, SGC‑RSS / relational, SMPD / spatial), their gates, retrieval/packing, adapters, maintenance threads, provenance & telemetry, and rollback semantics.

---

## A. Architectural baseline (from design & repo)

* **Uniform fusion point:** all memories return **`MemoryTokens`** that adapters cross‑attend into the LLM.
* **Shared patterns:** _retrieve → pad/project → build mask/meta → telemetry → MemoryTokens_; _gate → (action, reason, score?) → provenance_; _start/stop background maintenance_; _rollback N steps_.
* **Current code confirms duplication** in:
  * Retrieval packers: `hippo_mem/episodic/retrieval.py`, `hippo_mem/relational/retrieval.py`, `hippo_mem/spatial/retrieval.py`.
  * Gates & provenance: `hippo_mem/*/gating.py`.
  * Adapters: `hippo_mem/episodic/adapter.py`, `hippo_mem/spatial/adapter.py` (very similar attention/mask plumbing).
  * Maintenance/rollback: `hippo_mem/episodic/store.py`, `hippo_mem/relational/kg.py`, `hippo_mem/spatial/map.py`.
  * Telemetry registry: `hippo_mem/common/telemetry.py` (small, but update calls are duplicated).

---

## B. Cross‑cutting design principles

1. **API stability first.** Keep public names/entry points; add **shims** and remove after two releases.
2. **Single‑source of truth.** Centralize padding+projection, metadata fields, provenance shape, and thread lifecycle.
3. **No performance regressions.** Microbench adapters; keep tensor copies minimal in retrieval.
4. **Test‑guided migration.** Land changes in small PRs with golden tests and strict parity on fixed seeds.
5. **Observability upgrades.** Consistent telemetry/units and provenance keys across memory types.

---

## C. Work packages (WPs)

### WP1 — Retrieval pipeline consolidation
**Objective.** Remove repeated _retrieve → pad/project → meta/telemetry → MemoryTokens_ code across episodic/relational/spatial.

**Deliverables.**
* `hippo_mem/common/retrieval.py`
  * `def pad_and_project(vecs: np.ndarray, hits: int, limit: int, base_dim: int, proj: nn.Module, device, dtype) -> tuple[torch.Tensor, torch.BoolTensor]`
  * `def build_meta(kind: str, start: float, hits: int, k: int, **extra) -> dict[str, float]`
  * `def retrieve_and_pack_base(retrieve_fn, *, k: int, device, dtype, proj: nn.Module, build_meta_fn, telemetry_key: str) -> MemoryTokens`

**Refactors.**
* `episodic/relational/spatial/retrieval.py`: replace bespoke helpers (`_pad_and_pack`, `_pack_vectors`, batch loops, meta dicts, telemetry updates) with calls to the above.
* Keep thin wrappers: `episodic_retrieve_and_pack(...)`, `relational_retrieve_and_pack(...)`, `spatial_retrieve_and_pack(...)` that supply store‑specific `retrieve_fn` and `build_meta_fn`.

**Acceptance.**
* Test parity on `tests/test_*retrieval*.py` (k, hits, masks, shapes, meta fields).
* Wall‑time within ±2% on CPU microbench; no GPU‑path regressions.
* LOC and cyclomatic complexity drop recorded.

---

### WP2 — Gating API unification
**Objective.** Standardize gate return types and logging across memory types.

**Deliverables.**
* `hippo_mem/common/gates.py`
  * ```python
    @dataclass
    class GateDecision:
        action: str
        reason: str
        score: float | None = None
    GateResult = GateDecision  # alias for compatibility
    class MemoryGate(Protocol):
        def decide(self, *args, **kwargs) -> GateDecision: ...
    ```
* Back‑compat shim that still allows tuple‑unpacking for one transition release.
* Update gates in `episodic/`, `relational/`, `spatial/` to return `GateDecision`.

**Provenance helper (paired):**
* `hippo_mem/common/provenance.py`
  * `def log_gate(logger, mem: str, decision: GateDecision, payload: dict) -> None`

**Acceptance.**
* `tests/test_*gating*.py` updated for object return (plus tuple shim tests).
* Provenance NDJSON fields stable (`mem, action, reason, score, payload`).

---

### WP3 — Store lifecycle & rollback
**Objective.** Deduplicate thread lifecycle and rollback across stores while preserving existing semantics.

**Deliverables.**
* `hippo_mem/common/maintenance.py`
  * `class BackgroundTaskManager: start(interval: float) -> None; stop() -> None` (idempotent; joins threads; uses `threading.Event`)
* `hippo_mem/common/lifecycle.py`
  * `class StoreLifecycleMixin:` default `start_background_tasks`, `stop_background_tasks`, `log_status`; composes a `BackgroundTaskManager` with an overridable `_maintenance_tick(event)`.
* `hippo_mem/common/history.py`
  * `class RollbackMixin:` generic `rollback(self, n:int=1)`; defines standard `HistoryEntry` dataclass.

**Refactors.**
* Make `EpisodicStore`, `KnowledgeGraph`, `PlaceGraph` inherit `StoreLifecycleMixin` (+ `RollbackMixin` where applicable).
* Keep `relational/maintenance.py::MaintenanceManager` internally but delegate to mixin entry points.

**Acceptance.**
* Start/stop/stop‑again tests (idempotent) for all three stores.
* Rollback round‑trip tests (apply N ops → rollback N → byte‑equal snapshot or invariant checks).
* No change to external scheduling parameters; logs unaffected.

---

### WP4 — Cross‑attention adapter base
**Objective.** Factor shared attention plumbing across episodic & spatial adapters; keep relational adapter semantics unchanged.

**Deliverables.**
* `hippo_mem/common/attn_adapter.py`
  * `class CrossAttnAdapter(nn.Module):` LoRA q/k/v/o handling, KV head shaping (MQA/GQA), key‑padding mask build from `MemoryTokens.mask`, optional Flash‑Attn flag, residual gating when memory is empty.
* Refactor:
  * `hippo_mem/episodic/adapter.py` → subclass `CrossAttnAdapter` (enable Flash‑Attn if available/configured).
  * `hippo_mem/spatial/adapter.py` → subclass `CrossAttnAdapter` (GQA‑friendly shapes).

**Acceptance.**
* Golden numerical parity on fixed seeds in `tests/test_adapters.py` and `tests/test_adapter_wiring.py`.
* Microbench: <2% slowdown CPU path, no GPU perf regressions; KV memory unchanged.

---

### WP5 — Light observability cleanup
**Objective.** Small, safe consistency upgrades.

**Deliverables.**
* `hippo_mem/common/telemetry.py`: add `record_stats(kind: str, **metrics) -> None` thin helper and use from retrieval wrappers.
* `MemoryTokens` factory is **optional**; keep construction inline unless we later expand fields (avoid premature abstraction).

**Acceptance.**
* Existing telemetry tests pass; names/units unchanged.
* Minimal churn; defer deeper telemetry work until signals stabilize.

---

## D. Sequencing & rollout (4 phases)

**Phase 1 (safe wins):**
1. WP1 Retrieval pipeline consolidation
2. WP5 Observability cleanup (only retrieval call sites)

**Phase 2 (API coherence):**
3. WP2 Gating API unification (+ provenancing)

**Phase 3 (infra dedupe):**
4. WP3 Store lifecycle & rollback

**Phase 4 (heavier refactor with perf guardrails):**
5. WP4 Cross‑attention adapter base

Each phase should be a separate PR. Keep shims for one release, then remove once downstream code migrates.

---

## E. Success metrics & guardrails

* **Correctness parity:** all unit/integration tests green; golden outputs byte‑equal or within numerical tolerances.
* **Complexity/size:** record `radon cc -s -a`, total LOC, and per‑module LOC deltas before/after; target ≥ **–250 LOC** and lower avg complexity.
* **Performance:** retrieval & adapter microbenches within budget (±2% CPU, no GPU regression).
* **Observability consistency:** telemetry/provenance schemas consistent across modules.
* **Docs:** update docstrings, module headers (“Algorithm Card” sections), and any READMEs touched.

---

## F. Concrete “first diffs” to land

1) **Add `common/retrieval.py` + refactor episodic/relational to call it**  
   * New: `pad_and_project`, `build_meta`, `retrieve_and_pack_base`  
   * Replace: `_pad_and_pack`, `_pack_vectors`, inline meta & telemetry  
   * Tests: `tests/test_*retrieval*.py`

2) **Introduce `GateDecision` + `MemoryGate` + `log_gate`**  
   * New: `common/gates.py`, `common/provenance.py::log_gate`  
   * Refactor: all `*/gating.py` to return `GateDecision` and call `log_gate`  
   * Add tuple shim for backward‑compat tests

3) **Thread lifecycle mixin + background manager (+ optional rollback mixin)**  
   * New: `common/maintenance.py`, `common/lifecycle.py`, `common/history.py`  
   * Refactor: `episodic/store.py`, `relational/kg.py`, `spatial/map.py`

4) **Cross‑attention adapter base**  
   * New: `common/attn_adapter.py`  
   * Refactor: `episodic/adapter.py`, `spatial/adapter.py`  
   * Add microbench & golden tests

---

## G. Risk register & mitigations

| Risk | Where | Mitigation |
|---|---|---|
| Shape/mask broadcast bugs | WP4 adapters | Add strict shape asserts; golden tests with fixed seeds & random masks. |
| Silent telemetry schema drift | WP1/WP5 | Centralize field names in `build_meta`; add schema check in tests. |
| Thread lifecycle edge cases | WP3 | Idempotent `start/stop`; ensure `join` on stop; add double‑start/stop tests. |
| KG maintenance timing semantics | WP3 | Keep store‑specific intervals; wrap `MaintenanceManager` w/o behavior change. |
| Hopfield completion interaction | WP1 episodic | Keep completion opt‑in via `TraceSpec.params`; regression test with/without. |

---

## H. Mapping: current code → new abstractions

* **Retrieval packers** → `common/retrieval.py` used by `episodic/relational/spatial/retrieval.py`.
* **Gates** → return `GateDecision`; share `log_gate` helper.
* **Maintenance/rollback** → `StoreLifecycleMixin` + `BackgroundTaskManager` (+ `RollbackMixin`).
* **Adapters** → subclass `common/CrossAttnAdapter` where applicable.
* **Telemetry** → thin helper; otherwise unchanged.

---

## I. Back‑compat & deprecation

* Keep tuple unpacking from gates for one release (`(action, reason)`); log deprecation warning once per process.
* Export legacy names from old modules during migration (re‑exports) and remove after downstream code is updated.

---

## J. Acceptance checklist (per phase)

- [ ] All tests green; new tests added for helpers/mixins/bases.  
- [ ] `radon cc` and LOC deltas captured in PR description.  
- [ ] Microbench numbers attached for WP4.  
- [ ] Docs and headers updated; CHANGELOG entries for shims/deprecations.  

---

_This unified plan merges and reconciles prior proposals by: centralizing retrieval (RetrievalPipeline / pad_project / meta), unifying gating via `GateDecision` + provenance helper, deduplicating store lifecycle with a manager+mixin combo, and extracting a robust cross‑attention adapter base. Telemetry changes are deliberately light‑touch to minimize risk._
