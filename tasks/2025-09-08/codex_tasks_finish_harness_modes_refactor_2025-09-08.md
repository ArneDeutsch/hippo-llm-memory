# Finish Harness → Modes Refactor (Phase 2 & 3)
**Repo:** `hippo-llm-memory`  
**Scope:** `hippo_eval/eval/harness.py` ↔ `hippo_eval/eval/modes.py` (+ runner, tests)  
**Goal:** Move mode-specific logic *out of* `harness.py` into concrete strategies in `modes.py`, delete duplicated/legacy branches, and prove delegation with tests and guards — **no functional or CLI changes.**

> Use the “Refactor Transaction” phases, guardrails, and acceptance checks from `tasks/refactoring_guidelines.md` as the preface for this task. Enforce **No‑Stub**, **Delegation Proof**, **Branch Elimination Target**, **Coverage Freeze**, and **Lint/Type Clean** exactly as written.

---

## 0) Current state (baseline to confirm before you start)
- `hippo_eval/eval/harness.py` ≈ **1536 LOC**; still contains retrieval, prompt building, text generation, and ingest/gating blocks inline.
- `hippo_eval/eval/modes.py` ≈ **109 LOC**; strategies mostly carry flags (`retrieval_enabled`, `ingest_enabled`, etc.) and hooks are essentially empty.
- `hippo_eval/harness/runner.py::evaluate` hard‑codes `TestStrategy()` when calling `_evaluate(...)` (should be factory‑driven).

Record these numbers with the snapshot script in Task 1 and treat them as the baseline to beat.

---

## Tasks (follow the phases strictly)

### 1) Snapshot (pre‑refactor metrics)
- Add/extend `scripts/refactor_guard.py` to compute and write `artifacts/refactor_baseline.json` containing:
  - LOC for `harness.py`, `modes.py`.
  - Count of long functions (>60 LOC) per file.
  - Count of occurrences in `harness.py` of these anchors (they must go to 0 post‑refactor):
    - `active_adapters[name].retrieve(`
    - `apply_chat_template(`
    - `generate_text(`
    - `.teach(`
  - Public API signatures of `evaluate(...)`, `_evaluate(...)`, `_run_replay(...)`, and the CLI wrapper(s).
- Run `pytest -q --maxfail=1` and write coverage summary to the baseline JSON.

### 2) Expand the `ModeStrategy` surface (non‑stub)
In `hippo_eval/eval/modes.py`, replace the placeholder hooks with this **Protocol** and concrete strategies:

```py
from typing import Protocol, Tuple, List, Dict, Any, Optional
from hippo_mem.common import MemoryTokens
from hippo_mem.common.gates import GateCounters

class ModeStrategy(Protocol):
    retrieval_enabled: bool
    ingest_enabled: bool
    load_store: bool
    replay_mode: bool

    # Called once per run
    def before_run(self, cfg, modules, *, suite: Optional[str]) -> None: ...
    # Retrieval across adapters (E/R/S) and injection into adapters
    def retrieve(self, cfg, modules, item, *, context_key: str, adapters) -> Tuple[
        List[str],         # injected_context
        List[tuple[int,int] | None],  # positions
        List[str | None],  # sources
        List[str],         # topk_keys
        bool,              # mem_hit
        float,             # mem_latency_ms
        List[MemoryTokens] # mems
    ]: ...
    # Prompt construction
    def build_prompt(self, tokenizer, system_prompt, item, *, use_chat_template: bool) -> str: ...
    # Text generation
    def generate(self, model, tokenizer, prompt, *, max_new_tokens: int, long_context: bool) -> Tuple[str, int, int]: ...
    # Ingestion/gating per adapter
    def ingest(self, cfg, modules, item, *, adapters, suite: str, gating: Dict[str, GateCounters]) -> None: ...
    # Optional per‑task hook for telemetry updates
    def after_task(self, item, pred: str, *, telemetry) -> None: ...
    # Called once per run
    def after_run(self, inputs) -> None: ...
```

Implement `TeachStrategy`, `TestStrategy`, and `ReplayStrategy` as `@dataclass`es setting the appropriate flags; bodies must **move real logic** (see Tasks 3–6).

> **Guard:** Do **not** leave `pass`/empty/`return None` placeholder bodies. If a hook is not needed in a given strategy, call into a shared helper that performs a trivial but real action (e.g., snapshot/merge telemetry), or raise `NotImplementedError` and ensure the call‑site is unreachable for that mode.

### 3) Move retrieval into strategies
- Cut‑paste the retrieval block from `_evaluate` (anchor: `if retrieval_enabled and modules:` through the assembly of `MemoryTokens` and adapter injection) into a helper used by `ModeStrategy.retrieve(...)`.
- Preserve semantics:
  - Collate `mems`, accumulate `mem_latency`, collect `router_path`, and extend `topk_keys`.
  - Build the unified `MemoryTokens` tensor and call each module’s adapter to inject memory.
- Return `(injected_context, positions, sources, topk_keys, mem_hit, mem_latency, mems)` from the hook.
- Delete the old block from `harness.py` and replace with a single call to `strategy.retrieve(...)`.

### 4) Move prompt building into strategies
- Cut‑paste the `apply_chat_template(...)` usage (`use_chat_template`/`system_prompt` handling) into `ModeStrategy.build_prompt(...)`.
- Delete the old block from `harness.py` and call the hook instead.

### 5) Move generation into strategies
- Cut‑paste the `generate_text(...)` call and `long_context` logic into `ModeStrategy.generate(...)`.
- Delete the old call from `harness.py` and call the hook instead.

### 6) Move ingestion/gating into strategies
- Cut‑paste the ingestion write phase guarded by gates (anchors: `# write phase guarded by gates` through the end of adapter `teach(...)` calls and the relational gate post‑processing) into `ModeStrategy.ingest(...)`.
- Preserve the special cases already present (e.g., spatial “Start (0,0)” teach item when ingest is disabled).
- Delete the old block from `harness.py` and call the hook instead.

### 7) Wire strategy selection end‑to‑end
- Ensure a **single** factory is used everywhere: `get_mode_strategy(Mode(cfg.mode))`.
- In `hippo_eval/harness/runner.py::evaluate`, **remove** the `TestStrategy()` constructor and use the factory derived from `base_cfg`/`self.cfg.mode`.
- In `harness._evaluate(...)`, delete any leftover `mode` branching; rely solely on the `strategy` object and its flags.

### 8) Keep signatures & CLI stable
- Do **not** change public function signatures or CLI parameters.
- Ensure `scripts/eval_model.py` and tests still work with the same arguments.

### 9) Tests — prove delegation & parity
- Add `tests/eval/test_mode_delegation.py`:
  - Monkeypatch each hook on the active strategy with a spy and assert it is called at least once per run.
  - Negative test: Temporarily patch `TeachStrategy.ingest` to `raise NotImplementedError` and assert the run fails with a clear message.
- Extend `tests/eval/test_modes.py` to assert the flags **and** a minimal smoke run per mode uses the correct strategy.
- Golden parity test: Run a tiny suite (`SIZES=50, SEEDS=1337`) comparing pre‑vs‑post metrics (EM/raw+norm, latency mean) within tolerances (±0.5pp for EM, ±10% latency).

### 10) Guard script (post‑refactor checks)
- Extend `scripts/refactor_guard.py` to **fail** if:
  - Any of the retrieval/prompt/generation/ingest anchors still appear in `harness.py`.
  - Any strategy hook has an empty body or only `pass`/`return None`.
  - `harness.py` LOC is **not reduced by at least 25%** vs baseline (target ≤ 1150 LOC).
- Run it in CI before unit tests.

### 11) Lint, type, and docs
- Run `ruff`, `black --check`, and ensure type hints on new/changed public functions.
- Add/refresh docstrings in `modes.py` and at the top of `harness.py` summarizing what moved where and why.

### 12) Verify & document
- Recompute metrics (Task 1) into `artifacts/refactor_post.json` and assert:
  - `harness.py` LOC ≤ **1150** (≈−25% from baseline).
  - **0** occurrences in `harness.py` of the four anchors.
  - Coverage for `harness.py`+`modes.py` **≥ baseline**.
- Update an internal `CHANGELOG` section describing the refactor.

---

## Acceptance Criteria (machine‑checkable)
- `harness.py` shows **≤ 1150 LOC** and **0** occurrences of: `active_adapters[name].retrieve(`, `apply_chat_template(`, `generate_text(`, `.teach(`.
- No `if/elif` branching on mode in `harness.py`; decisions are polymorphic through `ModeStrategy`.
- Each hook (`before_run`, `retrieve`, `build_prompt`, `generate`, `ingest`, `after_task`, `after_run`) is **non‑stub** and executed in at least one test (use spies).
- CLI and public APIs unchanged; existing CLI tests pass.
- CI green for:
  - `make lint`
  - `pytest -q -m "not slow and not integration"`
  - `pytest -q --runintegration --runslow`
  - `scripts/ci_smoke_eval.sh`
- Coverage **not lower** than baseline for the touched modules.

---

## How to Verify (commands)
```bash
python scripts/refactor_guard.py --snapshot artifacts/refactor_baseline.json
pytest -q -m "not slow and not integration"
pytest -q --runintegration --runslow
scripts/ci_smoke_eval.sh
python scripts/refactor_guard.py --check artifacts/refactor_baseline.json artifacts/refactor_post.json
```

---

## Notes
- Prefer extracting small, pure helpers in `modes.py` over re‑using large `harness.py` helpers; this keeps coupling low.
- If shared behavior is substantial, add a private `_CommonStrategy` with reusable helpers and have `Teach|Test|ReplayStrategy` delegate internally.
- Leave `_run_replay(...)` in `harness.py` unchanged for now, but prefer to call strategy hooks from there if feasible in a follow‑up.
