# The Refactor Transaction (always follow these phases)

1. **Snapshot (pre‑refactor metrics)**
   - Record: file LOC, number of functions > 60 LOC, `if`/`elif` chains that switch on mode/strategy, and public API signatures.
   - Run: `pytest -q` and capture coverage; save baseline artifacts (e.g., `artifacts/refactor_baseline.json`).

2. **Extract & Redirect (mechanical, no behavior change)**
   - Create interfaces/classes *with the exact hook surface you need*.
   - **Move concrete logic** (cut‑paste) from old module to hook implementations. **Do not leave copies behind.**
   - Redirect call‑sites to use the new hooks; delete the old code paths.

3. **Consolidate & Simplify**
   - Remove obsolete branches, dead code, duplicated helpers.
   - Tighten visibility; minimize cross‑module imports.

4. **Verify (post‑refactor metrics)**
   - Recompute metrics from step 1 and **assert improvements** (or equality where required).
   - Run fast and slow tests, smoke scripts, and CI integration commands from this repo.

5. **Document and Cleanup**
   - Update docstrings and module‐level narrative (what moved where and why).
   - Add a brief `CHANGELOG` entry for internal maintainers.
   - Remove temporary artifacts created for the refactoring only (snapshot, metrics, e.g. `artifacts/refactor_baseline.json`)

## Guardrails the agent **must enforce**

- **No‑Stub Rule**: Newly introduced hooks/classes **must not** contain placeholder bodies that simply `return` or pass. If a hook is expected to execute logic, it must either:
  - include the migrated logic, or
  - raise `NotImplementedError` **until** the caller is proven unreachable by tests.
- **Delegation Proof**: Add a test that **counts hook invocations** per run (e.g., a Spy/monkeypatch) to prove `evaluate(...)` delegates into strategies/adapters.
- **Branch Elimination Target**: The originating file must show a **strict reduction** in `mode` branching. Example acceptance: “No `if cfg.mode`/`if mode` branches remain in `harness.py`; decisions are polymorphic.”
- **API Compatibility**: CLI and function signatures exposed from the package **must not change** (unless the task explicitly allows).
- **Coverage Freeze**: Overall coverage must not drop; critical modules (the refactored file + new module) have **>= baseline coverage**.
- **Lint/Type Clean**: `ruff`, `black --check`, and type hints for all new/changed public functions.

## Task Template for Future Refactors (fill every section)

**Title**: Refactor _<area>_ by extracting _<object>_ and delegating logic

**Scope**  
- *No functional changes.* Preserve CLI and public APIs.
- Move logic from _<old modules/functions>_ to _<new modules/classes>_.

**Plan (checklist)**  
- [ ] Record baseline metrics (LOC, long functions, branch count on key discriminators like `mode`).  
- [ ] Create interfaces/classes and **move** concrete logic into them.  
- [ ] Replace call‑sites to use the new types. Remove old branches.  
- [ ] Update/extend tests to prove the delegation occurs.  
- [ ] Run linters, unit, slow, and integration tests.  
- [ ] Write migration notes in module docstrings.

**Acceptance Criteria (machine‑checkable)**  
- [ ] The refactored source file has **≤ X LOC** (or **−Y%** vs baseline).  
- [ ] **0** occurrences of `if`/`elif` branching on the removed discriminator in the refactored file.  
- [ ] At least **N calls** to each new hook during `pytest -q` (enforced via a spy).  
- [ ] Coverage of refactored area **≥ baseline**.  
- [ ] CI targets all pass: `make lint && pytest -q -m "not slow and not integration" && pytest -q --runintegration --runslow && scripts/ci_smoke_eval.sh`.

**Verification Script (drop into `scripts/refactor_guard.py`)**  
- Counts LOC, long functions, and pattern occurrences (`grep`/AST).  
- Fails if hooks are stubs (detects empty bodies or `pass`/`return None` only).  
- Emits a JSON report consumed by tests to assert thresholds.

## Concrete example (this repo) — Harness → Modes

**Intent**: Move *mode‑dependent* logic out of `hippo_eval/eval/harness.py` into `hippo_eval/eval/modes.py` strategies.

**Required surface (non‑stub)** in `modes.py`:
```py
class ModeStrategy(Protocol):
    retrieval_enabled: bool
    ingest_enabled: bool
    def before_run(self, cfg, modules, tasks, *, suite: str|None) -> None: ...
    def retrieve(self, cfg, modules, item, *, context_key: str, adapters) -> tuple[list[str], list[str], list[str], MemoryTokens|None]: ...
    def build_prompt(self, tokenizer, system_prompt, item, *, use_chat_template: bool) -> str: ...
    def generate(self, model, tokenizer, prompt, *, max_new_tokens: int, long_context: bool) -> tuple[str, int, int]: ...
    def ingest(self, cfg, modules, item, *, adapters, suite: str, gating) -> None: ...
    def after_task(self, item, pred, *, telemetry) -> None: ...
    def after_run(self, inputs) -> None: ...
```
**Delegation requirement** in `harness._evaluate`:
- Replace in‑line retrieval/ingest/generation with calls to the strategy hooks above.  
- Remove `if cfg.mode`/`if strategy.replay_mode` branching from `harness.py` (asserted by guard script).

**Tests to add/modify**:
- `tests/eval/test_modes.py`: add spies to assert each hook is called on at least one task per run.  
- Golden parity test: Compare metrics from pre‑refactor (commit tag) vs post‑refactor on `SIZES=50, SEEDS=1337` small suites.  
- Negative test: make `TeachStrategy.ingest` raise `NotImplementedError` in a temporary stub and assert the run fails — prevents stub regressions.

## Anti‑patterns (fail the task if detected)
- Creating enums/empty classes without moving call‑sites.  
- Leaving old branches in place “for compatibility”.  
- Duplicating logic across old and new modules.  
- Reducing test scope to make the change pass CI.

## Deliverables Checklist
- [ ] Code change implementing the refactor with real logic moved.  
- [ ] Updated tests proving delegation and preserving behavior.  
- [ ] `scripts/refactor_guard.py` + JSON report committed.  
- [ ] Short docstring/narrative in the refactored modules.  
- [ ] Green CI with unchanged CLI.

---

*Use this document as a preface for every refactor task you give to Codex. The combination of explicit **mechanics**, **metrics**, and **verification** is what forces an actual refactor rather than a superficial scaffold.*
