# High‑Level Plan (Detailed): Pipeline Overhaul with Deprecations & Migration


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

## 1) Goals & Non‑Goals

**Goals**
- Measure **memory uplift** under **closed‑book** conditions.
- Exercise HEI‑NW / SGC‑RSS / SMPD as in `experiment-synthesis.md`.
- Provide **trustworthy telemetry** to attribute gains to memory.

**Non‑Goals**
- Perfect IE for arbitrary prose (rule‑based extraction is fine for synthetic).
- Maintaining legacy datasets in `main` (they will be archived).

---

## 2) Architecture Overview

### 2.1 Data → Harness → Stores → Retriever → LLM

```
+-------------------+     +-------------------+     +------------------+
| ClosedBookScenario| --> | ScenarioRunner    | --> | Stores (write)   |
| (teach/test)      |     |  teach / reset    |     |  episodic/KG/map |
+-------------------+     |  test (read-only) |     +------------------+
                          |  session_id scope |              |
                          +-------------------+              v
                                        +--------------------------+
                                        | Retriever (per algorithm)|
                                        | pack <= 256-512 tokens   |
                                        +-------------+------------+
                                                      v
                                              +---------------+
                                              |  LLM (answer) |
                                              +---------------+
```

### 2.2 Data Model
```python
class ClosedBookScenario(BaseModel):
    suite: Literal["semantic_closed_book","episodic_closed_book","spatial_explore"]
    session_id: str
    teach: List[str]
    test: Dict[str, Any]  # {{ "query": str, "answer": str, "metadata": dict }}
```

### 2.3 Store Interfaces (sketch)
```python
class EpisodicStore:
    def write(self, text: str, session_id: str, ts: int): ...
    def retrieve(self, query: str, session_id: str, k: int=3) -> List[str]: ...

class RelationalStore:
    def upsert(self, h: str, r: str, t: str, session_id: str, evidence:int=1): ...
    def subgraph(self, query_ents: List[str], session_id: str, hops:int=2): ...

class SpatialStore:
    def observe(self, u, v, cost: int, session_id: str): ...
    def plan(self, start, goal, session_id: str) -> List[str]: ...
```

### 2.4 Retrieval Packing
- For episodic/relational: pack up to **256–512 tokens** of retrieved context appended to the test prompt.
- For spatial: return **action string**; no retrieval packing needed.

---

## 3) Algorithm‑Specific Design

### 3.1 HEI‑NW (Episodic)
- **Write:** encode teach sentences (e.g., model embeddings) → normalize → k‑WTA sparse indices. Store `(session_id, key, text, ts)`.
- **Retrieve:** cosine/overlap to top‑k; pack sentences with minimal preamble.
- **Edge cases:** tie‑break by recency; deduplicate near‑identical keys.

### 3.2 SGC‑RSS (Relational)
- **IE rules (synthetic):** simple patterns like `"{{A}}" bought `"{{X}}"` at `"{{B}}"` -> `(A, bought_at, B)`.
- **Contradictions:** multiple edges `(StoreB, located_in, Berlin/London)` tracked with `evidence` counts or timestamps; retrieval can surface both with stance.
- **Retrieve:** entities in query → k‑hop neighborhood; serialize compact graph into ~200 tokens (triples list or mini‑adjacency).

### 3.3 SMPD (Spatial)
- **Teach:** parse `OBS: (x,y)->(u,v)`; add undirected or directed edges per environment.
- **Test:** plan with Dijkstra/A* on accumulated graph; output `UDLR` (or `NSEW`) action string.
- **Multi‑episode:** multiple `teach` observations per `session_id`, possibly across files; planner should work with partial maps.

---

## 4) Telemetry & Metrics (spec)

### 4.1 Per‑item JSON
```json
{{
  "session_id":"...",
  "suite":"...",
  "mode":"test",
  "retrieval":{{"hit":true,"k":3,"tokens":180,"latency_ms":17}},
  "gen":{{"latency_ms":210}},
  "scores":{{"em":1,"f1":1.0}},
  "store":{{"version":2,"backend":"sqlite|jsonl|faiss"}}
}}
```

### 4.2 Aggregates
- **Closed‑book baseline EM/F1**
- **Memory EM/F1**
- **Uplift**
- **Hit‑rate**
- **Avg pack tokens**
- **Avg latency (retrieval/gen)**
- **SMPD:** success rate, optimality gap, planning latency, re‑trial speedup

### 4.3 Reports
- New tables with the above metrics; per‑suite and overall.
- Plots (optional) showing EM vs pack tokens (diminishing returns).

---

## 5) Deprecations & Migration (step‑by‑step)

1. **Archive:** `git tag legacy-datasets-v1 && git push origin legacy-datasets-v1`  
2. **Remove legacy datasets:** `git rm -r data/semantic data/semantic_hard ... data/spatial_hard`  
3. **Remove generators/tooling:** `git rm hippo_eval/tasks/generators.py hippo_eval/tasks/spatial/generator.py scripts/datasets_cli.py scripts/audit_datasets.py`  
4. **Prune configs & harness branches** referencing legacy suites.  
5. **Add CI grep‑guard** to fail if forbidden patterns reappear.  
6. **Introduce closed‑book generators and ScenarioRunner.**  
7. **Implement writers and telemetry.**  
8. **Docs:** add `DEPRECATIONS.md`, update plans/protocols.

Forbidden path patterns are listed in the tasks milestone (M0.6).

---

## 6) Timeline & Milestones

- **M0 (week 1):** Cleanup & CI guard.  
- **M1 (week 1–2):** Closed‑book data model & generators (n=10 smoke).  
- **M2 (week 2):** ScenarioRunner & sessionization.  
- **M3 (week 2–3):** Writers/retrievers for all algorithms.  
- **M4 (week 3):** Telemetry & reports.  
- **M5 (week 3):** Docs & protocol updates.  
- **M6 (week 3–4):** QA smoke, then scale to n=50/100.  
- **M7 (week 4+):** Stress, ablations, and full reports.

---

## 7) Acceptance Gates

- **Gate A:** Closed‑book baseline < in‑context baseline by ≥ 20pp (or a threshold you pick), proving closed‑book is *actually* closed.  
- **Gate B:** Memory uplift > 0 on at least one suite (n=10).  
- **Gate C:** Hit‑rate > 0 and pack tokens ≤ budget.  
- **Gate D:** SMPD success > 0 with correct optimality on tiny mazes.  
- **Gate E:** CI green with grep‑guard enabled.

---

## 8) Risks & Mitigations

- **IE fragility:** templates are synthetic → rules are stable; unit test with golden fixtures.  
- **Token budgets:** cap pack to 256–512 tokens; measure tradeoffs.  
- **Interference:** default to `session_id` scoping; include explicit interference tests.  
- **Regressions from removals:** archival tag + CI guard.

---

## 9) Deliverables

- New data generators & harness flow.  
- Algorithm writers & retrievers.  
- Telemetry and updated reports.  
- `DEPRECATIONS.md` + CI guard.  
- Passing smoke and scaled runs.
