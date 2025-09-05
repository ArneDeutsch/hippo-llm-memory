
# Stepwise Plan — Memory-First Evaluation Pipeline Rework
**Project:** LLM Hippocampus (HEI‑NW · SGC‑RSS · SMPD)  
**Version:** v2025‑09‑05  
**Owner:** Eval/Infra Working Group  
**Goal:** Address the review findings by evolving the evaluation pipeline **incrementally** so that (1) memory is required for target suites, (2) leakage is eliminated/measurable, (3) SMPD tests true map learning, and (4) deprecated suites are removed without breaking the workflow.

---

## 0) Scope & Principles

- **Do not “start over.”** Every step yields a runnable state with smoke‑tests (n=50, seed=1337) and clear acceptance checks.
- **Teach/Test separation.** Facts/trajectories are injected **only** in teach; queries appear **only** in test.
- **Isolation & attribution.** Prevent or precisely measure cross‑item leakage; require retrieval justifications from the right context.
- **Guardrails.** Fail fast if a “memory‑required” suite is solvable in‑context.
- **Deletion at the end.** When the new path is green (twice), delete superseded code/data to avoid bloat.

---

## 1) Milestones (M0 → M7)

### ✅ M0 — Branch, freeze & smoke
**Outcome:** Safe workspace for changes; current system reproducible.

- Create branch: `rework/memory-split`.
- Tag current state: `v0_eval_legacy`.
- Run a smoke to freeze baselines:  
  ```bash
  RUN_ID=20250905_smoke BASE="n=50 seed=1337"
  python scripts/eval_cli.py suite=semantic preset=baseline ${BASE} outdir=runs/$RUN_ID/semantic_baseline
  python scripts/eval_cli.py suite=episodic_cross preset=baseline ${BASE} outdir=runs/$RUN_ID/episodic_cross_baseline
  python scripts/eval_cli.py suite=spatial preset=baseline ${BASE} outdir=runs/$RUN_ID/spatial_baseline
  ```
- Archive: `reports/$RUN_ID` and `runs/$RUN_ID`.

**Acceptance:** Baselines reproduce previous numbers within ±1pp EM/F1.

---

### 🧭 M1 — Documentation‑first refactor
**Outcome:** The repository’s ground truth explains **how** we will evaluate memory.

**Edit:**
- `EVAL_PLAN.md`
  - Define **paired datasets**: `*_teach.jsonl` (facts/trajectories only) and `*_test.jsonl` (queries only).
  - Define **modes** and defaults per suite: `teach` (retrieval **off**), `replay` (optional), `test` (retrieval **on**).
  - Define **isolation levels**: `per_item`, `per_episode`, `none`.
  - Add **guardrails**: “no facts in test prompts”; “baseline EM ≤ 0.2” for memory‑required suites; fail if violated.
- `DESIGN.md`
  - Thread a `context_key` (episode_id/time) through write/retrieve APIs and telemetry.
  - Describe **justification logging** (trace IDs, context match rate) and **leakage probes**.
- `EVAL_PROTOCOL.md`
  - Stepwise recipe for each suite: **build → teach → (replay) → test → report**.
  - Smoke commands for `n=50, seed=1337` after each milestone.

**Acceptance:** Docs build contains the above sections; commands copy‑pastable; CI spell/links lint passes.

---

### 🧰 M2 — Harness controls & isolation
**Outcome:** The core runner can enforce teach/test discipline and isolation.

**Code changes:**
- `hippo_eval/eval/harness.py` → introduce:
  - `--no-retrieval-during-teach` (default **true** for memory suites).
  - `--isolate={per_item|per_episode|none}`; implement store **fork/clear** between units.
  - `context_key` plumbed into store writes/retrieval; reject mismatched traces when isolation is active.
- `hippo_mem/stores/*`:
  - Add optional `context_key` column/field; retrieval filter.
- `scripts/eval_cli.py`:
  - Surface the new flags; default appropriately by suite.
- Telemetry:
  - Counters: `retrieval.requests`, `writes.count`, `store.size_before/after`, `justification.context_match_rate`.

**Tests:**
- Unit tests for isolation (mock store): per‑item clear/fork works; mismatched trace blocked.
- Teach pass asserts `retrieval.requests == 0` when the flag is on.

**Acceptance (smoke):**
- Teach logs show `retrieval.requests==0` for memory suites.
- Test logs show `retrieval.requests>0` and `context_match_rate≥0.9` on synthetic setups.

---

### 🧪 M3 — Memory‑required semantic & episodic suites
**Outcome:** Suites that **cannot** be solved from the prompt alone.

**Datasets & CLI:**
- `hippo_eval/tasks/semantic/generator.py`
  - Add `require_memory=True` mode producing:
    - `semantic_teach.jsonl`: triples/tuples only.
    - `semantic_test.jsonl`: questions only (no facts).
- `hippo_eval/tasks/episodic_cross/generator.py`
  - Replace “FLUSH”-in‑prompt pattern with **true cross‑session** split:
    - teach: single episodic fact,
    - test: query referencing prior entity; **no supporting facts**.
- `hippo_eval/datasets/cli.py`
  - Expose `require_memory` and generate paired files.
- Guardrails:
  - Validator fails if any test prompt contains schema keywords that match teach facts.
  - Pre‑test baseline check aborts if EM>0.2.

**Harness wiring:**
- New presets: `memory/sgc_rss_mem`, `memory/hei_nw_cross` use M2 flags by default.

**Acceptance (smoke):**
- Baseline EM on `semantic_mem` and `episodic_cross_mem` ≤ 0.2.
- Memory variants show **≥ +0.5 EM uplift** at n=50, seed=1337.
- Reports include justification snippets for ≥90% of correct answers.

---

### 🗺️ M4 — SMPD evaluation rebuilt for map learning
**Outcome:** Spatial suite requires **iterative map building** and **macro reuse**.

**Datasets & environment:**
- `hippo_eval/tasks/spatial/env.py` (new): fixed grid/topology sampler with optimal path oracle.
- `hippo_eval/tasks/spatial/generator_multi.py` (new):
  - **Teach episodes:** progressive exploration on the same grid (visitation coverage grows).
  - **Replay:** successful trajectories available for distillation.
  - **Test episodes:** held‑out start→goal inside explored region; no map hints in prompt.

**Metrics & reporting:**
- Success rate, optimality gap (∆steps vs. oracle), mean plan length, and **learning curves** across episodes.
- Macro reuse KPI: improvement in plan length/latency before vs. after replay‑to‑policy.

**Acceptance (smoke):**
- Baseline succeeds < 30% with larger optimality gaps.
- SMPD memory shows rising success curve and decreasing ∆steps across episodes.

---

### 📊 M5 — Guardrails, leakage probes & richer reports
**Outcome:** Fail fast on design regressions; make memory use observable.

**Add:**
- `scripts/validate_prompts.py`: assert no facts leaked into test prompts.
- `hippo_eval/reporting/`:
  - **Leakage panel:** contradictions injected across items; ensure context‑aware retrieval picks the right context.
  - **Justification coverage:** % answers with ≥1 supporting trace from the correct context.
  - **Gate quality:** (where labeled) precision/recall for neuromodulated writes.

**Acceptance:** Any violation flips run status to **FAILED** with actionable diagnostics; HTML/MD reports show new panels.

---

### 🔁 M6 — Protocol migration & dual‑path compatibility
**Outcome:** Users can run both legacy and new paths during transition.

- `EVAL_PROTOCOL.md`:
  - Default to the **new** suites; keep a “legacy path” appendix.
- `scripts/build_datasets.py`:
  - Build both legacy and new artifacts during M6 only.
- CI: two green end‑to‑end runs on `semantic_mem`, `episodic_cross_mem`, and `spatial_multi`.

**Acceptance:** Both paths run; new path meets M3/M4 thresholds; legacy marked **DEPRECATED** in docs.

---

### 🧹 M7 — Deletion of deprecated suites & cleanup
**Outcome:** No bloat; single source of truth.

- Remove legacy dataset generators:
  - `tasks/semantic/generator_legacy.py`, `tasks/episodic_cross/generator_legacy.py`, `tasks/spatial/generator.py` (one‑shot shortest‑path).
- Remove legacy presets and configs referencing in‑prompt facts.
- Purge old artifacts under `datasets/legacy/` and `examples/legacy/`.
- Update tests to drop legacy paths.
- Final doc sweep: remove references in `EVAL_PLAN.md`, `DESIGN.md`, `EVAL_PROTOCOL.md`.

**Acceptance:** Repo has no legacy code/data; CI green; binary size & wheel size stable or reduced.

---

## 2) Implementation Task Matrix (for assignment / Codex prompts)

| Area | Tasks | Files (indicative) |
|---|---|---|
| Harness | Flags, isolation, context_key, telemetry | `hippo_eval/eval/harness.py`, `hippo_mem/stores/*`, `scripts/eval_cli.py` |
| Datasets | Paired teach/test generators & CLI | `hippo_eval/tasks/semantic/*`, `hippo_eval/tasks/episodic_cross/*`, `hippo_eval/datasets/cli.py` |
| Spatial | Multi‑episode env + generator + metrics | `hippo_eval/tasks/spatial/env.py`, `generator_multi.py`, `metrics.py` |
| Guardrails | Prompt validator; baseline ceiling check | `scripts/validate_prompts.py`, `hippo_eval/eval/guards.py` |
| Reporting | Leakage, justification, gate panels | `hippo_eval/reporting/*` |
| Docs | Plan/protocol/design updates | `EVAL_PLAN.md`, `EVAL_PROTOCOL.md`, `DESIGN.md` |
| Cleanup | Delete legacy & tests | tree under `tasks/*`, `presets/*`, `datasets/legacy/*` |

> Tip: implement in the order **Harness → Datasets → Spatial → Guardrails → Reporting → Cleanup** to maintain runnable states.

---

## 3) Step‑by‑Step Commands (per milestone)

**M2 harness smoke (semantic_mem placeholder):**
```bash
python scripts/eval_cli.py suite=semantic preset=memory/sgc_rss_mem \
  mode=teach --no-retrieval-during-teach=true --isolate=per_item \
  outdir=runs/m2_semantic_teach store_dir=stores/m2_semantic session_id=m2_sem

python scripts/eval_cli.py suite=semantic preset=memory/sgc_rss_mem \
  mode=test --isolate=per_item \
  outdir=runs/m2_semantic_test store_dir=stores/m2_semantic session_id=m2_sem
```

**M3 dataset validation:**
```bash
python scripts/build_datasets.py suite=semantic --require_memory=true --out datasets/semantic_mem/
python scripts/validate_prompts.py datasets/semantic_mem/semantic_test.jsonl
python scripts/eval_cli.py suite=semantic_mem preset=baseline n=50 seed=1337 outdir=runs/m3_semantic_baseline
```

**M4 SMPD learning curve:**
```bash
python scripts/eval_cli.py suite=spatial_multi preset=memory/smpd mode=teach outdir=runs/m4_spatial_teach
python scripts/eval_cli.py suite=spatial_multi preset=memory/smpd mode=replay outdir=runs/m4_spatial_replay
python scripts/eval_cli.py suite=spatial_multi preset=memory/smpd mode=test outdir=runs/m4_spatial_test
```

---

## 4) Acceptance Gates (must meet to proceed)

1. **Semantic_mem & Episodic_cross_mem**
   - Baseline EM ≤ 0.20; Memory uplift ≥ +0.50 EM at n=50, seed=1337.
   - Teach: `retrieval.requests==0`; Test: `retrieval.requests>0`.
   - Justification context match ≥ 90%.
2. **SMPD (spatial_multi)**
   - Success rate ↑ across episodes; optimality gap ↓; macro reuse KPI improves after replay‑to‑policy.
3. **Guardrails**
   - No facts in test prompts; leakage probe passes under `per_item` isolation.
4. **Docs & Cleanup**
   - Docs reflect new pipeline; legacy code/data removed; CI green twice consecutively.

---

## 5) Risks & Mitigations

- **Risk:** Over‑tight guardrails cause false failures.  
  **Mitigation:** Provide override flags (`--allow-baseline-high`), but red‑flag reports.
- **Risk:** Store fork/clear is slow at `per_item`.  
  **Mitigation:** Add a memory‑mapped fork path; default `per_episode` for heavy SMPD tasks.
- **Risk:** Dataset drift across episodes.  
  **Mitigation:** Fix random seeds per RUN_ID; persist topology configs alongside artifacts.

---

## 6) Deliverables Checklist

- [ ] Updated `EVAL_PLAN.md`, `DESIGN.md`, `EVAL_PROTOCOL.md`.
- [ ] Harness flags & isolation + unit tests.
- [ ] `semantic_mem` and `episodic_cross_mem` paired datasets.
- [ ] `spatial_multi` (teach/replay/test) suite & metrics.
- [ ] Guardrails + validator script.
- [ ] Reporting panels for leakage/justification/gate quality.
- [ ] Two consecutive green runs meeting thresholds.
- [ ] Deletion PR removing legacy generators, presets, tests, and docs.

---

## 7) Appendix — Minimal Interfaces (illustrative)

**Harness call (pseudo):**
```python
run_suite(
  suite_name: str,
  mode: Literal["teach","replay","test"],
  store: Store,
  no_retrieval_during_teach: bool = True,
  isolate: Literal["per_item","per_episode","none"] = "per_item",
  context_key_fn: Callable[[Example], str] = default_episode_id,
)
```

**Store retrieve/write (added):**
```python
store.write(trace, context_key=episode_id, salience=score)
store.retrieve(query, k=K, context_key=episode_id)  # filter by context when provided
```

**Telemetry fields (JSONL):**
```json
{
  "qid": "semantic_mem/00042",
  "mode": "test",
  "retrieval": {"requests": 3, "hits": 3, "context_match_rate": 1.0},
  "writes": {"count": 2, "gated_in": 2, "gated_out": 1},
  "store": {"size_before": 1204, "size_after": 1207}
}
```

---

**End of Plan**

