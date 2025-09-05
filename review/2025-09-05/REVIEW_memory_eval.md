# Deep Review: Validation Approach, Evidence, Implications, Proposed Changes **and Deprecations**

**Project:** LLM hippocampus-inspired memory (HEI‑NW, SGC‑RSS, SMPD)  
**Ground truth:** `research/experiment-synthesis.md`  
**Evaluation:** `EVAL_PLAN.md`, `EVAL_PROTOCOL.md`  
**Artifacts reviewed:** `runs/run_20250904/`, `reports/run_20250904/` (SIZES=50, SEED=1337)


> **Path hints (adjust to your repo):**
> - Datasets: `data/{semantic,semantic_hard,episodic*,spatial*}`
> - Generators: `hippo_eval/tasks/generators.py`, `hippo_eval/tasks/spatial/generator.py`
> - Harness & eval: `hippo_eval/eval/harness.py`, `hippo_eval/harness/*`, `scripts/eval_cli.py`
> - Stores: `hippo_mem/{episodic,relational,spatial}/*`, `hippo_eval/stores/*`
> - Reporting: `hippo_eval/reporting/*`, `reports/*`
> - Configs: `configs/datasets/*`, `configs/presets/*`
> - Docs: `EVAL_PLAN.md`, `EVAL_PROTOCOL.md`, `DESIGN.md`, `MILESTONE_9_PLAN.md`
> - Artifacts (your run): `runs/run_20250904/`, `reports/run_20250904/`


---

## 0) Scope & Audience

This review explains **why current results cannot prove or disprove** the value of your memory algorithms, and details **exact changes** to fix the evaluation, including **what to remove** from the repo to avoid bloat and mixed signals.

Audience: engineers implementing the pipeline, reviewers signing off on methodology, and anyone reading the reports.

---

## 1) Ground Truth: What the algorithms should do

### HEI‑NW (Episodic indexing with replay)
- **Goal:** recall *specific past episodes* encountered earlier. Durable across time; robust to distractors.
- **Mechanics (intended):** write episodic traces (embeddings/sparse sketches) during “teach”; later queries retrieve and surface relevant traces.

### SGC‑RSS (Relational/semantic consolidation with selective recall)
- **Goal:** build a compact **graph** of entities and relations from facts learned earlier; answer multi‑hop queries; handle contradictions via evidence/weighting.
- **Mechanics (intended):** extract (head, relation, tail) edges during “teach”, maintain per‑session KG; retrieve minimal subgraph during “test”.

### SMPD (Spatial map & path planning)
- **Goal:** **accumulate** a map from partial observations across episodes; **plan** a path at test (exploit).
- **Mechanics (intended):** ingest transitions u→v in “teach”; at “test”, run a planner (Dijkstra/A*) on the accumulated graph.

> **Key commonality:** Memory is **written** during teach and **read** during test, where test is **closed‑book** (no answer‑critical facts in the test prompt).

---

## 2) Current Approach: Why it fails to measure memory

### 2.1 All facts in the test prompt → baseline saturation
- Most datasets concatenate **all needed facts + the question** into a **single prompt**. The base model solves it **in‑context**.
- “Hard” variants mainly add distractors; failures (if any) reflect **reasoning difficulty**, not missing information.

**Symptom in reports:** near‑ceiling EM/F1 for baselines; memory variants show **no uplift** because memory isn’t needed.

### 2.2 “FLUSH” is non‑functional
- Prompts like:  
  `Frank walked to the Park. --- FLUSH --- Carol traveled to the Office ... Where did Frank go?`
- Still **one prompt** → no actual reset. The harness does not start a new conversation. So past info remains available in context.

### 2.3 Writers are placeholders; no per‑scenario namespace
- “Teach” writes often store **dummy vectors** or unrelated triples; they do **not** encode the facts from teach.
- Reused names (Alice/Bob/Carol) across scenarios + **no `session_id` filter** → risk of cross‑story interference.

### 2.4 SMPD is mis‑tested
- Spatial suite is a **one‑shot path puzzle**; there’s no **multi‑episode** map accumulation.
- The harness never ingests **transitions**, so the planner (if any) has nothing to use.

### 2.5 Telemetry doesn’t demonstrate memory contribution
- Missing or incomplete: retrieval hit‑rate, retrieved token budget, memory latency, interference checks. Reports can’t show *how* memory helped.

---

## 3) Evidence (code & artifacts)

> These examples are representative; line numbers may differ after refactors.

- **Datasets:** `data/semantic/*`, `data/episodic_cross/*`, `data/spatial/*` include **single‑prompt** items with all facts present. “--- FLUSH ---” appears inside prompts but does not cause a reset.
- **Harness:** single message per item; no conversation reset between teach and test; `mode=test` disables writes (good) but test still **contains all facts** → memory unused.
- **Reports:** baseline EM ≈ 0.94–1.00 on semantic/episodic; SMPD ≈ 0 on spatial (no map ingestion).

**Root‑cause table**

| Symptom | Root Cause | Impact |
|---|---|---|
| Baseline ≈ 1.0, memory no better | Test prompt already contains facts | Memory cannot show uplift |
| “FLUSH” ineffective | No multi‑turn conversation; token is inert | Episodic evaluation invalid |
| Cross‑story interference risk | No `session_id` enforced in stores | Contaminated retrieval (silent) |
| SMPD zero success | No transitions ingested; one‑shot puzzles | Spatial memory untested |
| Reports not informative | No hit‑rate/latency/pack size | Can’t attribute effects to memory |

---

## 4) What “good” looks like (target behavior)

### 4.1 Closed‑book, sessionized evaluation
- **Teach (writes allowed):** present facts or observations only.
- **Reset:** fresh conversation; no carry‑over tokens.
- **Test (reads only):** query without restating facts; retrieval pulls only from memory **scoped by `session_id`**.

**Example scenario (semantic):**
```json
{{
  "suite": "semantic_closed_book",
  "session_id": "sem_001",
  "teach": [
    "Carol bought an apple at StoreB.",
    "StoreB is in Berlin.",
    "Some reports claim StoreB is in London."
  ],
  "test": {{
    "query": "In which city did Carol buy the apple?",
    "answer": "Berlin"
  }}
}}
```

### 4.2 Content‑aware writers
- **Episodic:** store embeddings/sparse keys and raw snippets from teach sentences.
- **Relational:** extract triples and upsert to a per‑session KG; track evidence.
- **Spatial:** parse transitions `(x,y)→(u,v)` and build a per‑session graph.

### 4.3 Telemetry
- **Hit‑rate:** did memory contribute?  
- **Pack size:** how many tokens retrieved?  
- **Latency:** retrieval + generation times.  
- **Interference test:** when `session_id` is withheld on a subset, accuracy should drop (by design).

---

## 5) Proposed Changes (detailed)

### 5.1 Data model (ClosedBookScenario)
```python
class ClosedBookScenario(BaseModel):
    suite: Literal["semantic_closed_book","episodic_closed_book","spatial_explore"]
    session_id: str
    teach: List[str]            # facts or observations only
    test: Dict[str, Any]        # {{ "query": str, "answer": str, "metadata": dict }}
```
- Files under `data/<suite>/<n>_<seed>.jsonl`.

### 5.2 Harness: ScenarioRunner
- `run_teach(scenario, algo, session_id)` → writes allowed.
- `reset_conversation()` → start fresh dialog.
- `run_test(scenario, algo, session_id, retrieval=True, long_context=False)` → reads only.

**Pseudocode**
```python
for sc in load_scenarios(path):
    with Session(sc.session_id) as sess:
        for msg in sc.teach:
            algo.write(msg, session_id=sc.session_id)   # teach phase
        reset_conversation()
        answer = algo.answer(sc.test["query"],
                             session_id=sc.session_id,
                             retrieval=True, write=False)
        score(answer, sc.test["answer"])
```

### 5.3 Stores and writers
- **Common:** every write/read carries `session_id`. Default retrieval filters by `session_id`.
- **Episodic (HEI‑NW):**
  - Encode teach sentences → `vec` → sparse sketch via k‑WTA (top‑k indices).
  - Store: `{{session_id, key, text, ts}}`.
  - Retrieve top‑k by cosine or overlap; pack ≤ 256 tokens.
- **Relational (SGC‑RSS):**
  - Rule‑based extraction for synthetic text → `(h, r, t)`.
  - Store per‑session KG; store evidence counts and timestamps.
  - Retrieve k‑hop subgraph relevant to query entities; serialize compactly.
- **Spatial (SMPD):**
  - Parse `OBS: (x,y)->(u,v)`; `graph.add_edge((x,y),(u,v), cost=1)`.
  - Test: run Dijkstra/A*; return action string (e.g., `RRUUR`).

### 5.4 Telemetry & metrics
- **Closed‑book uplift:** `memory_em − closed_book_baseline_em`.
- **Hit‑rate:** fraction of items where at least one retrieved unit overlaps gold provenance.
- **Pack size:** tokens introduced from memory.
- **Latency:** retrieval ms, generation ms.
- **SMPD:** success rate, optimality gap, planning latency, re‑trial speedup.

**Log schema (per item)**
```json
{{
  "session_id": "sem_001",
  "suite": "semantic_closed_book",
  "mode": "test",
  "retrieval": {{
    "hit": true,
    "k": 3,
    "tokens": 182,
    "latency_ms": 17
  }},
  "gen": {{"latency_ms": 210}},
  "scores": {{"em": 1, "f1": 1.0}}
}}
```

---

## 6) **Deprecations & Removals** (explicit)

We will **remove** legacy data and code paths that encode all facts in single prompts or one‑shot spatial puzzles. We keep reproducibility via a **git tag**.

### 6.1 Datasets (remove from main)
- `data/semantic/`, `data/semantic_hard/`
- `data/episodic*/` (including `episodic_cross*`, `episodic_multi*`, `episodic_capacity*`)
- `data/spatial/`, `data/spatial_hard/`

### 6.2 Generators & tooling (remove/replace)
- Remove: `hippo_eval/tasks/generators.py`, `hippo_eval/tasks/spatial/generator.py`, `scripts/datasets_cli.py`, `scripts/audit_datasets.py`
- Replace with: `scripts/gen_closed_book.py`, `scripts/gen_spatial_explore.py`

### 6.3 Configs & harness code (prune)
- Remove suite references from `configs/datasets/*.yaml`, presets, and harness branches that assume single‑prompt items.

### 6.4 Tests (rewrite)
- Remove tests tied to legacy suites; add tests for schema, session scoping, planner, and uplift metrics.

### 6.5 Docs
- Update `EVAL_PLAN.md`, `EVAL_PROTOCOL.md`, `DESIGN.md`, `MILESTONE_9_PLAN.md`.
- Add `DEPRECATIONS.md` with the above list and the archival tag (e.g., `legacy-datasets-v1`).

### 6.6 CI guard
- Add a grep step failing on any reintroduction of forbidden paths (patterns included in the tasks).

---

## 7) Migration & Rollout

**M0:** Archive & remove legacy assets; CI guard.  
**M1:** Introduce closed‑book schema & generators.  
**M2:** ScenarioRunner (teach→reset→test) + `session_id` propagation.  
**M3:** Content‑aware writers (HEI‑NW, SGC‑RSS, SMPD).  
**M4:** Telemetry uplift panels.  
**M5:** Docs & protocols.  
**M6:** QA smoke (n=10) then scale.  
**M7:** Extended reporting & stress tests.

---

## 8) Conclusions

- Your doubts are **confirmed**: the current pipeline measures **in‑context reasoning**, not memory.  
- The redesign is straightforward: **closed‑book**, **sessionized**, **content‑aware**.  
- With explicit deprecations and CI guards, we avoid bloat and keep the repo focused on what matters.
