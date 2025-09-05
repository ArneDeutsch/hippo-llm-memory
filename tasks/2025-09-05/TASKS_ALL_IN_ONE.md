# ALL-IN-ONE — Codex Tasks for Memory-First Pipeline Rework


---

## TASKS_01_M1_Documentation_Update.md

# T01 — Documentation-first refactor (Milestone M1)

**Goal**  
Align the repo docs with the reworked, memory-first evaluation approach so contributors know what to build and how to run it _before_ we touch code.

**Source of truth**  
- Review findings show that current suites keep facts in the same prompt and blur memory effects; we must split teach/test, enforce isolation, and add guardrails.  
- The stepwise plan (M1) requires updated `EVAL_PLAN.md`, `DESIGN.md`, and `EVAL_PROTOCOL.md` describing paired datasets, modes, isolation levels, guardrails, and smoke steps.

---

## Deliverables

1. **EVAL_PLAN.md** (revise)
   - Add a section **“Paired Datasets (Teach/Test)”** defining `*_teach.jsonl` (facts/trajectories only) vs `*_test.jsonl` (queries only).
   - Define **modes** and defaults per suite: `teach` (retrieval off), `replay` (optional), `test` (retrieval on).
   - Define **isolation levels**: `per_item`, `per_episode`, `none` (with examples).
   - Guardrails:
     - “No facts in test prompts”. If violated, fail the run.
     - For **memory-required** suites, baseline EM ceiling **≤ 0.2** (fail if higher, unless manually overridden).
   - State acceptance gates and smoke-test commands with `n=50, seed=1337`.

2. **DESIGN.md** (revise)
   - Describe a per-example **`context_key`** (e.g., `episode_id` or wall-clock) carried through `write` and `retrieve` to enable attribution and leakage checks.
   - Describe **justification logging**: when a prediction uses memory, emit trace IDs and a context-match rate.
   - Describe **leakage probes**: contradictions across items to ensure retrieval chooses the right context.

3. **EVAL_PROTOCOL.md** (revise)
   - Provide a **recipe per suite**: `build → teach → (replay) → test → report`.
   - Include exact smoke commands for `semantic_mem`, `episodic_cross_mem`, and `spatial_multi` once available.
   - Keep a **Legacy Path** appendix, marked deprecated, to ease transition.

---

## File Paths

- `hippo-llm-memory-main/hippo-llm-memory-main/EVAL_PLAN.md`
- `hippo-llm-memory-main/hippo-llm-memory-main/DESIGN.md`
- `hippo-llm-memory-main/hippo-llm-memory-main/EVAL_PROTOCOL.md`

---

## Steps

1. Edit `EVAL_PLAN.md`:
   - Add a new top-level section **“Memory-First Suites”** with bullets for `semantic_mem`, `episodic_cross_mem`, `spatial_multi`.
   - Add a table mapping suites → default `mode` sequences and default `--isolate` values.
   - Add **Guardrails** and **Acceptance** subsections (copy-paste runnable commands).

2. Edit `DESIGN.md`:
   - Add a new subsection **“Context-Keyed Memory Access”** explaining `context_key` for both write and retrieval and how it flows through adapters and telemetry.
   - Add a new subsection **“Justification & Leakage Telemetry”** listing the fields and examples.

3. Edit `EVAL_PROTOCOL.md`:
   - Provide **copy-paste** command blocks for teach/test per suite.
   - Provide a **smoke** block that runs `n=50, seed=1337` for each new suite.
   - Provide a **migration** note that legacy suites will be removed after two green runs.

---

## Acceptance

- All three documents updated as above.
- Commands are **copy-pasteable** and reference real scripts (`scripts/eval_model.py`, `scripts/eval_cli.py`).
- CI (if present) passes link/spellcheck on docs.
- Team can follow the protocol to run a smoke without further clarification.


---

## TASKS_02_M2_Harness_Isolation_and_Flags.md

# T02 — Harness: isolation controls, teach/test discipline, telemetry (Milestone M2)

**Goal**  
Add harness-level controls to (1) disable retrieval during teach, (2) isolate memory per item or episode, and (3) plumb a `context_key` with attribution telemetry.

---

## Relevant Code (current repo)

- **Entry**: `scripts/eval_model.py` → forwards to `hippo_eval.eval.harness.main`
- **Harness**: `hippo_eval/eval/harness.py` uses helpers from `hippo_eval/harness/*`
- **Runner/IO/Metrics** (already modularized):
  - `hippo_eval/harness/runner.py`
  - `hippo_eval/harness/io.py`
  - `hippo_eval/harness/metrics.py`
- **Stores/Adapters**:
  - Episodic: `hippo_mem/episodic/{store.py,retrieval.py}`
  - Relational: `hippo_mem/relational/{backend.py,retrieval.py}`
  - Spatial: `hippo_mem/spatial/{map.py,retrieval.py}`

---

## Required Changes

1. **CLI flags and config plumbing**
   - Add two flags (with defaults for memory suites):
     - `--no-retrieval-during-teach` (bool; default **true** for memory suites).
     - `--isolate={per_item|per_episode|none}` (default **per_item** for semantic/episodic, **per_episode** for spatial).
   - Surface via:
     - `scripts/eval_model.py` (OmegaConf overrides)
     - `scripts/eval_cli.py` (pass-through)
     - `hippo_eval/eval/harness.py` (read and apply)

2. **Context key plumbing**
   - Introduce a string `context_key` computed per example (default: `example.get("episode_id") or example.get("qid")`).
   - Pass `context_key` to store writes and retrieval calls.

3. **Isolation implementation**
   - Implement per-item/episode **clear or fork** of the store:
     - Add store utility in `hippo_eval/eval/store_utils.py` (or extend it) to `fork_store()` and `clear_store()`.
     - For `per_item`, fork/clear after each test QID; for `per_episode`, only at episode boundaries (if present in dataset).

4. **Telemetry**
   - Count and log:
     - `retrieval.requests`
     - `writes.count`
     - `store.size_before/after`
     - `justification.context_match_rate` (see below)

5. **Justification / context-match**
   - Extend retrieval wrappers to return selected trace IDs and any attached `context_key` so we can compute `% retrieved traces whose context matches this QID's context_key`.
   - Emit to per-item JSONL telemetry and aggregate to suite-level metrics.

---

## Implementation Sketch

- **`hippo_eval/eval/harness.py`**
  - Parse new flags, set defaults based on suite/preset.
  - During `mode="teach"`:
    - Enforce `no-retrieval-during-teach == True` by bypassing memory adapters' retrieval step (adapter flag or wrapper).
  - Between items:
    - Apply isolation policy using `fork_store()` / `clear_store()`.
  - Per example:
    - Compute `context_key` and attach to write/retrieve calls.

- **`hippo_mem/*/*/retrieval.py` & `store.py` (episodic/relational/spatial)**
  - Accept optional `context_key` in `write(...)` and filter by `context_key` in `retrieve(..., context_key=...)` when provided.
  - Return metadata including `trace_id` and (if stored) the trace `context_key` for justification checks.

- **`hippo_eval/harness/metrics.py`**
  - Calculate `context_match_rate` = matched_traces / total_traces for answered items.

---

## Tests & Smoke

1. **Unit-ish tests** (lightweight)
   - Add tests in `tests/` that mock a small store to verify:
     - `per_item` isolation forks/clears between consecutive items.
     - `no-retrieval-during-teach` results in `retrieval.requests == 0` for teach.
     - Retrieval returns trace meta so `context_match_rate` can be computed.

2. **Manual smoke**
   ```bash
   RUN_ID=m2_smoke
   python scripts/eval_cli.py suite=semantic preset=memory/sgc_rss_mem \
     mode=teach --no-retrieval-during-teach=true --isolate=per_item \
     outdir=runs/$RUN_ID/semantic_teach store_dir=stores/$RUN_ID session_id=sem

   python scripts/eval_cli.py suite=semantic preset=memory/sgc_rss_mem \
     mode=test --isolate=per_item \
     outdir=runs/$RUN_ID/semantic_test store_dir=stores/$RUN_ID session_id=sem
   ```

**Pass criteria**
- Teach logs: `retrieval.requests == 0`.
- Test logs: `retrieval.requests > 0`, telemetry contains `context_match_rate`.


---

## TASKS_03_M3_Datasets_Semantic_and_Episodic_Splits.md

# T03 — Datasets: memory-required semantic & episodic_cross (Milestone M3)

**Goal**  
Create paired teach/test datasets that **cannot be solved from the prompt alone** and wire them into the CLI.

---

## Relevant Code (current repo)

- Generators: `hippo_eval/tasks/generators.py`
  - `generate_semantic`, `generate_episodic_cross`, etc.
- CLI: `hippo_eval/datasets/cli.py`
- Loaders: `hippo_eval/datasets/loaders.py`
- Legacy spatial one-shot: `hippo_eval/tasks/spatial/generator.py` (unchanged here)

---

## Required Changes

1. **Semantic (new memory-required variant)**
   - Add a mode `require_memory=True` for `generate_semantic`:
     - **Teach output (`semantic_teach.jsonl`)**: emit tuples/facts only, e.g. `{ "fact": "StoreB is in Berlin.", "entity": "StoreB", "context_key": "...", ... }`
     - **Test output (`semantic_test.jsonl`)**: emit questions only, e.g. `{ "prompt": "Where is StoreB?", "answer": "Berlin", "context_key": "...", ... }`
   - Ensure **no facts** appear in test prompts.

2. **Episodic Cross (true cross-session)**
   - Replace in-prompt `"--- FLUSH ---"` pattern with real split:
     - **Teach**: emit the salient episode fact only (no question).
     - **Test**: emit a question that requires the previously taught fact (no supporting facts).

3. **CLI wiring**
   - Extend `hippo_eval/datasets/cli.py` with `--require_memory` and output to paired files:
     - `datasets/<suite>_mem/<suite>_teach.jsonl`
     - `datasets/<suite>_mem/<suite>_test.jsonl`
   - Add a guard that fails generation if any test prompt contains tokens overlapping the teach facts’ schema keywords.

4. **Validation script (lightweight)**
   - Add `scripts/validate_prompts.py` that checks “no facts in test prompts” and basic schema separation.

---

## File-Level Instructions

- **`hippo_eval/tasks/generators.py`**
  - Update `generate_semantic(...)` signature to accept `require_memory: bool = False`.
  - If `require_memory`:
    - Return a dict: `{ "teach": [ ... ], "test": [ ... ] }` instead of flat list.
    - Each teach item should include a `context_key` (use deterministic key like `f"sem/{i:05d}"`). The corresponding test item **must** carry the same `context_key`.
  - Add `generate_episodic_cross_mem(...)` that yields `{ "teach": [...], "test": [...] }` with aligned `context_key` across items.

- **`hippo_eval/datasets/cli.py`**
  - Add `--require_memory` flag.
  - When set, write the two JSONL files and update the dataset card.

- **`scripts/validate_prompts.py`** (new)
  - Read a test JSONL, scan for leakage of teach `entity`/`fact` strings.
  - Exit non-zero with a helpful message on violation.

---

## Tests & Smoke

1. Build & Validate
   ```bash
   python scripts/build_datasets.py suite=semantic --require_memory=true --out datasets/semantic_mem/
   python scripts/validate_prompts.py datasets/semantic_mem/semantic_test.jsonl

   python scripts/build_datasets.py suite=episodic_cross --require_memory=true --out datasets/episodic_cross_mem/
   python scripts/validate_prompts.py datasets/episodic_cross_mem/episodic_cross_test.jsonl
   ```

2. Baseline check (should be **low**)
   ```bash
   python scripts/eval_cli.py suite=semantic_mem preset=baseline n=50 seed=1337 outdir=runs/m3_semantic_baseline
   python scripts/eval_cli.py suite=episodic_cross_mem preset=baseline n=50 seed=1337 outdir=runs/m3_epcross_baseline
   ```

**Pass criteria**
- Baseline EM ≤ 0.20 on both suites.
- Files are emitted with matching `context_key` across teach/test.


---

## TASKS_04_M4_SMPD_MultiEpisode_and_Metrics.md

# T04 — Spatial/SMPD: multi-episode environment, replay-to-policy, metrics (Milestone M4)

**Goal**  
Replace the one-shot shortest-path puzzles with a **fixed topology** explored across **multiple episodes**; evaluate planning and macro reuse with learning curves.

---

## Relevant Code (current repo)

- One-shot generator: `hippo_eval/tasks/spatial/generator.py`
- Spatial memory module: `hippo_mem/spatial/{map.py, macros.py, retrieval.py}`
- Reporting: `hippo_eval/reporting/{report.py, tables.py, plots/*}`

---

## Required Changes

1. **Environment (new)**
   - `hippo_eval/tasks/spatial/env.py`: sample a fixed grid/topology with an optimal path oracle and obstacle sampler.

2. **Generator (new)**
   - `hippo_eval/tasks/spatial/generator_multi.py` produces `{ teach: [...episodes...], test: [...episodes...] }` for a single topology.
     - Teach episodes: progressive exploration to improve coverage.
     - Optional replay episodes: expose successful trajectories for distillation.
     - Test episodes: start/goal pairs **inside explored regions**, **no map hints** in the prompt.
   - Emit `context_key` per topology and `episode_id` per episode.

3. **Metrics**
   - Add success rate, optimality gap (∆steps vs oracle), mean plan length, and per-episode learning curves.
   - Add macro reuse KPI: plan length/latency improvement before vs after replay-to-policy.

4. **Harness integration**
   - New suite name: `spatial_multi` with `teach → replay → test` modes.
   - Defaults: `--isolate=per_episode` (stores persist across episodes of same topology; reset between topologies).

---

## File-Level Instructions

- **`hippo_eval/tasks/spatial/env.py`** (new)
  - Implement grid builder with obstacles and an A*/Dijkstra oracle.

- **`hippo_eval/tasks/spatial/generator_multi.py`** (new)
  - Yield a dict with keys `{ "topology_id", "episodes": [...] }` for teach/test.
  - Each episode item should include `{ "prompt", "answer", "episode_id", "context_key": topology_id }`.

- **`hippo_eval/harness/metrics.py`** (extend)
  - Add aggregation for success rate, optimality gap, and learning curves (episode index on x-axis).

- **`hippo_eval/reporting/report.py` & `plots/*`** (extend)
  - Render a **learning curve** figure and a macro reuse table for `spatial_multi`.
  - Include before/after replay quick stats in the top summary.

---

## Tests & Smoke

```bash
RUN_ID=m4_spatial
python scripts/eval_cli.py suite=spatial_multi preset=memory/smpd mode=teach outdir=runs/$RUN_ID/teach
python scripts/eval_cli.py suite=spatial_multi preset=memory/smpd mode=replay outdir=runs/$RUN_ID/replay
python scripts/eval_cli.py suite=spatial_multi preset=memory/smpd mode=test outdir=runs/$RUN_ID/test
```

**Pass criteria**
- Baseline success rate < 30% with sizable optimality gaps.
- SMPD memory shows improving success and shrinking ∆steps across episodes; macro reuse KPI improves post-replay.


---

## TASKS_05_M5_Guardrails_Leakage_Reporting.md

# T05 — Guardrails, leakage probes, and richer reporting (Milestone M5)

**Goal**  
Fail fast when suites are solvable in-context, ensure test prompts don't contain facts, and surface memory attribution clearly.

---

## Required Changes

1. **Prompt validator (new script)**
   - `scripts/validate_prompts.py` (if not built in T03) validates test JSONL has no leaked facts or schema tokens from teach.

2. **Baseline ceiling guard**
   - In `hippo_eval/eval/harness.py` (or `eval/audit.py`), add a pre-check that runs a few baseline samples on memory-required suites; if EM > 0.2, abort with an actionable error (allow override `--allow-baseline-high`).

3. **Leakage probes**
   - Add option to inject contradictions across items (e.g., same entity with different attributes) during build for a dedicated probe split.
   - In evaluation, compute a **context-aware retrieval success**: the model should select the correct context or abstain, per suite rule.

4. **Reporting panels**
   - Extend `hippo_eval/reporting/report.py` and `tables.py` to include:
     - Leakage probe pass/fail.
     - Justification coverage (% answered with ≥1 supporting trace from the right context).
     - Gate quality (where labels exist): precision/recall for salience writes.

---

## Tests & Smoke

- Run prompt validation on all memory suites.
- Trigger the baseline guard with a purposely “too-easy” test to see the failure message.
- Produce a report and visually confirm the new panels. 

**Pass criteria**
- Violations cause a clear **FAILED** status with diagnostics.
- Reports render new panels without layout regressions.


---

## TASKS_06_M6_Protocol_Migration_and_Dual_Path.md

# T06 — Protocol migration with dual-path compatibility (Milestone M6)

**Goal**  
Default the repo to the new memory-first pipeline while keeping legacy paths temporarily available.

---

## Steps

1. **EVAL_PROTOCOL.md**
   - Promote the new suites to the default flow. Add a boxed “Legacy Path (Deprecated)” appendix.

2. **scripts/build_datasets.py**
   - Build both legacy and new artifacts during M6. Emit `datasets/<suite>_mem/...` alongside legacy for side-by-side runs.

3. **CI / Smoke scripts**
   - Add a job that runs:
     - `semantic_mem`, `episodic_cross_mem`, `spatial_multi` (new path)
     - legacy semantic/episodic/spatial (old path)
   - Mark the legacy job as **allowed to fail** temporarily, but report its status.

**Pass criteria**
- Two consecutive green runs of the new path at `n=50, seed=1337`.
- Legacy path is documented as deprecated and remains runnable for comparison.


---

## TASKS_07_M7_Cleanup_and_Deletion.md

# T07 — Cleanup: remove deprecated generators, presets, tests, docs (Milestone M7)

**Goal**  
Eliminate bloat and ensure a single source of truth remains after the new pipeline proves stable.

---

## Deletions (examples; adjust to actual tree)

- Legacy dataset generators that embed facts in prompts:
  - `hippo_eval/tasks/semantic/generator_legacy.py` (if present)
  - `hippo_eval/tasks/episodic_cross/generator_legacy.py` (if present)
  - `hippo_eval/tasks/spatial/generator.py` (the one-shot puzzle version), once `generator_multi.py` is adopted.
- Legacy presets/configs referencing in-prompt facts.
- `datasets/legacy/` artifacts (remove from repo; keep in release tags if needed).
- Tests that only cover legacy behavior.
- Doc references in `EVAL_PLAN.md`, `DESIGN.md`, `EVAL_PROTOCOL.md` to old suites.

## Safety Net

- Before deletion, tag the repo (e.g., `v0_eval_legacy`).
- Ensure two green runs on the new path.
- Confirm wheels/binaries don’t grow.

**Pass criteria**
- No references to legacy code remain.
- CI green after deletion PR.
