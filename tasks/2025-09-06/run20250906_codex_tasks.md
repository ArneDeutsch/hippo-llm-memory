# Codex Implementation Tasks — Run 20250906 Review Fixes

**Repo:** `hippo-llm-memory-main/hippo-llm-memory-main`  
**Run reviewed:** `runs/run20250906` (N=50, SEED=1337)  
**Goal:** Fix the issues uncovered in the review so the three hippocampus-inspired memory algorithms become **observable, testable, and useful** in the evaluation pipeline.

> Each task below is self-contained. It includes: context, target files (or how to locate them), concrete changes, acceptance criteria, and verification commands. Execute tasks in order (P0 → P1 → P2).

---

## P0 — Make memories non-degenerate and visible in prompts

### Task P0.1 — **Enable embeddings by default in SGC‑RSS (semantic KG)**
**Context**
- In `runs/run20250906/stores/sgc_rss/.../kg.jsonl`, nodes/edges exist but `embedding` fields are null/empty and retrieval hit@k=0.  
- We must compute and persist node/edge embeddings when not provided.

**Locate**
- Search for the semantic KG module (one of):  
  - `hippo/**/semantic/*kg*.py`, `hippo/**/kg/*.py`, or `hippo_mem/**/semantic/**`.  
  - Functions likely named `upsert`, `add_node`, `add_edge`, `persist`, `retrieve`, or similar.

**Change**
1. In the **upsert** path for nodes and edges, if `*_embedding` is `None`/empty, compute:
   ```python
   from hippo_mem.retrieval.embed import embed_text

   def ensure_vec(text: str, dim: int = 16):
       v = embed_text(text, dim=dim)
       assert v is not None and len(v) == dim and any(abs(x) > 1e-12 for x in v)
       return v
   ```
   - Node: `name` → `embedding`
   - Edge: `relation` or stringified triple → `embedding`
2. Persist these vectors in `kg.jsonl` (`embedding: List[float]`).
3. Update retrieval to treat missing vectors as an **error** (fail-fast).

**Acceptance Criteria**
- When inserting 2 facts (`bought`, `is in`), both nodes and edges have **non-empty, non-zero** embeddings.
- Retrieval with `k>=1` returns a **2-hop** subgraph containing both facts.
- New unit test passes (see “Tests” below).

**Tests**
- Create `tests/semantic/test_embeddings_default.py`:
  ```python
  def test_kg_auto_embeddings(kg):
      a = kg.upsert_node("Carol")
      b = kg.upsert_node("StoreB")
      c = kg.upsert_node("Berlin")
      e1 = kg.upsert_edge(a, "bought_at", b)
      e2 = kg.upsert_edge(b, "located_in", c)

      for x in (a,b,c,e1,e2):
          assert getattr(x, "embedding", None)
          assert sum(abs(v) for v in x.embedding) > 0

      sub = kg.retrieve(query="Carol bought apple in which city?", k=4, hops=2)
      names = {n.name for n in sub.nodes}
      rels  = {e.relation for e in sub.edges}
      assert {"Carol","StoreB","Berlin"} <= names
      assert {"bought_at","located_in"} <= rels
  ```
- Wire a minimal fixture or factory for `kg`.

**Verify**
```
pytest -q tests/semantic/test_embeddings_default.py
```

---

### Task P0.2 — **Provide default schemas for tuple extraction in SGC‑RSS**
**Context**
- Synthetic data uses relations like `bought`, `bought_at`, `is in`, `located_in`. Nothing is promoted into the KG due to schema/threshold mismatch.

**Locate**
- Tuple/IE pipeline: search `schema`, `extract`, `relation`, `tuple`, `triplet`.
- A config file may exist: `configs/semantic/*.yaml` or Python dict defaults.

**Change**
1. Add a default schema covering: `bought`, `bought_at`, `is`, `in`, `located_in`, `at`.
2. Set thresholds to `0.5–0.6` (align with `tuples.score_confidence` if present).
3. Ensure **promotion** to KG runs in teach mode.

**Acceptance Criteria**
- On a 10-sample synthetic batch, the KG totals `nodes >= 100` and `edges >= 100`.  
- Gate attempts > 0 (see P2.1), retrieval hit@k **> 0**.

**Verify**
```
python scripts/eval_cli.py suite=semantic preset=memory/sgc_rss n=10 seed=1337 mode=teach persist=true --strict-telemetry
python scripts/validate_store.py --suite semantic --expect-nodes 100 --expect-edges 100
```

---

### Task P0.3 — **Fix episodic key export (HEI‑NW)**
**Context**
- `episodic.jsonl` shows many zero-norm keys; retrieval hit-rate is low and context match is 0.

**Locate**
- HEI‑NW store writer; search `episodic`, `DGKey`, `kWTA`, `to_dense`, `save`:
  - e.g., `hippo/**/episodic/*.py`

**Change**
1. Ensure keys persisted are **dense** (`to_dense()`).
2. Enforce `k > 0` (k-WTA). If misconfigured, clamp to a sane default and **log a warning**.
3. Add assertion at write-time: `||key|| > 0` for ≥90% of traces; otherwise raise.

**Acceptance Criteria**
- New write path yields `nonzero_ratio >= 0.9` on a 50-sample teach run.
- Retrieval hit@k improves (non-zero).
- A unit test validates densification and norm > 0.

**Tests**
- `tests/episodic/test_dense_keys.py`:
  ```python
  def test_dense_keys_written(episodic_store_factory):
      store = episodic_store_factory()
      store.write_trace(event="Carol visited Library", key_params={"k":8})
      rec = next(store.iter_jsonl())
      assert "key" in rec and sum(abs(x) for x in rec["key"]) > 0
  ```

**Verify**
```
pytest -q tests/episodic/test_dense_keys.py
```

---

### Task P0.4 — **Audit actual prompt packing and injected recalls**
**Context**
- The LLM ignores retrieved memory. We need to confirm recalls are present and *where* in the prompt.

**Locate**
- Prompt assembly for each suite; search `pack_prompt`, `assemble_prompt`, `context`, `retrieval`.

**Change**
1. Extend `audit_sample.jsonl` entries to include:
   - `injected_context`: the exact text spans inserted,
   - `positions`: token start/end (or character offsets),
   - `source`: identifiers of retrieved items.
2. If `injected_context` is missing for a sample where retrieval attempted, **fail** the run (exit non-zero in strict mode).

**Acceptance Criteria**
- `runs/.../audit_sample.jsonl` contains the new fields for ≥90% of rows.
- Strict telemetry aborts if missing.

**Verify**
```
python scripts/eval_cli.py ... --strict-telemetry
jq -c '.injected_context,.positions,.source' runs/run20250906/**/audit_sample.jsonl | head
```

---

## P1 — Constrained decoding & answerability ceilings

### Task P1.1 — **Short‑answer enforcer (episodic + semantic)**
**Context**
- Model outputs are long/off-format; EM/F1 remain 0 even when memory might be correct.

**Locate**
- Post-processing of generations: search `normalize_prediction`, `postprocess`, `metric_em`.

**Change**
1. Add a normalizer enforcing:
   - Allowed chars: `A-Za-z0-9 .,'-` (configurable),
   - Max length: 32 chars (configurable per suite).
2. Log `normalized_pred` alongside raw prediction.
3. Compute EM/F1 on normalized strings.

**Acceptance Criteria**
- New column `normalized_pred` present in `metrics.csv`.
- EM/F1 computed on normalized and reported as `pre_em`/`pre_f1`.

**Verify**
```
pytest -q tests/eval/test_normalizer.py
```

---

### Task P1.2 — **Spatial output validator + oracle path (SMPD)**
**Context**
- Outputs violate the `UDLR` policy; retrieval hit@k=0; need a mechanistic oracle.

**Locate**
- SMPD evaluator/generator; search `spatial`, `grid`, `path`, `UDLR`.

**Change**
1. Add a validator: `^[UDLR]{1,64}$`. If invalid, set `normalized_pred = ""` and flag `format_violation=true` in row logs.
2. Implement **oracle shortest path** (BFS or A* with unit costs) over the stored grid/graph to compute `oracle_path`.
3. Log:
   - `oracle_path`,
   - `oracle_success` (bool, if target is reachable),
   - `pred_matches_oracle` (bool).
4. Report **oracle_success_rate** as an additional metric.

**Acceptance Criteria**
- On synthetic grids with a defined path, `oracle_success_rate` ≈ 1.0.
- `valid_action_rate` reported and > 0.8 on normalized outputs (after training/teach).

**Verify**
```
pytest -q tests/spatial/test_oracle_bfs.py
```

---

### Task P1.3 — **Oracle readers for episodic & semantic**
**Context**
- We need an **upper bound** that ignores LLM generation and reads memory directly.

**Locate**
- Evaluation pipeline; introduce `oracle_mode` or separate evaluator.

**Change**
1. For episodic/semantic samples, if the retrieved context contains the gold span (string match or id match), the oracle returns the gold answer.
2. Emit `oracle_em` and `oracle_f1` in metrics and `metrics.json`.

**Acceptance Criteria**
- On runs where memory is present, `oracle_em` ≫ 0 even if model EM=0.
- CI asserts `oracle_em >= 0.8` on synthetic mini-runs.

**Verify**
```
pytest -q tests/eval/test_oracle_readers.py
```

---

## P2 — Gate calibration, retrieval sweeps, and fail‑fast checks

### Task P2.1 — **Gating regression tests & calibration**
**Context**
- HEI‑NW gate accepts ~100%; SGC‑RSS accepts 0%. We want middle ground in teach.

**Locate**
- Gating modules; search `gate`, `gating`, `should_write`, `accept`.

**Change**
1. Add synthetic fixtures with known positives/negatives and confidence scores.
2. Target acceptance ~50–80% in teach mode (configurable); assert via tests.
3. Log `gate.attempts` and `gate.accepted` per suite; never zero in teach.

**Acceptance Criteria**
- `gate.attempts > 0` and `0 < gate.accepted < gate.attempts` on smoke runs.

**Verify**
```
pytest -q tests/gating/test_gate_calibration.py
```

---

### Task P2.2 — **Retrieval @k sweeps + monotonicity check**
**Context**
- hit@k=0 indicates index/query issues.

**Locate**
- Retrieval config; search `k`, `topk`, `neighbors`, `faiss`.

**Change**
1. Add a sweep `k in {1,4,8,16}` and compute `hit_rate_at_k` for each.
2. Assert monotonic non-decreasing hit rates.

**Acceptance Criteria**
- In telemetry, `ret.hit_rate.k1 <= k4 <= k8 <= k16`.
- Non-zero hits on at least one k.

**Verify**
```
pytest -q tests/retrieval/test_k_sweep.py
```

---

### Task P2.3 — **Fail‑fast telemetry guards (store size, embeddings, injected context)**
**Context**
- We want early exits when memory is empty or broken.

**Locate**
- `scripts/validate_store.py` and pipeline guards.

**Change**
1. Add checks per suite:
   - **Size**: episodic traces ≥ N*0.8; semantic nodes/edges ≥ thresholds; spatial nodes/edges ≥ thresholds.
   - **Embedding integrity**: non-zero ratio ≥ 0.9 (episodic/semantic).
   - **Injected context**: audit rows must show `injected_context` when retrieval attempted.
2. On violation with `--strict-telemetry`, **exit non-zero**.

**Acceptance Criteria**
- Smoke runs either pass all guards or fail clearly with actionable messages.

**Verify**
```
python scripts/validate_store.py --strict-telemetry
```

---

## CI & Smoke

### Task CI.1 — **Update smoke tests and preflight**
**Context**
- Current smoke fails due to `gate.attempts == 0 in dry-run`.

**Locate**
- `scripts/ci_smoke_eval.sh`, `preflight` checks (possibly `scripts/preflight.py` or in `validate_store.py`).

**Change**
1. Record **gate attempts** even in dry-run/teach, or **relax** the check in smoke (only warn).
2. Ensure smoke runs with small `n=8` produce:
   - non-empty stores,
   - non-zero `gate.attempts`,
   - `audit_sample.jsonl` with injected context fields.

**Acceptance Criteria**
- `scripts/ci_smoke_eval.sh` exits 0 on green and prints a short summary table.

**Verify**
```
bash scripts/ci_smoke_eval.sh
```

---

## Documentation

### Task DOC.1 — **Add slim EVAL_PROTOCOL.md (no fluff)**
**Context**
- Keep only the minimal, parameterized protocol.

**Change**
- Replace content with:
  ```md
  # EVAL_PROTOCOL (slim)
  ## Params
  export RUN_ID=run$(date +%Y%m%d)
  export SIZES=(50)     # or: 50 100
  export SEEDS=(1337)   # or: 1 2 3
  export MODEL=gpt-4o-mini
  ## Run
  scripts/eval_cli.py suite=episodic preset=memory/hei_nw n="$SIZES" seed="$SEEDS" run_id="$RUN_ID" mode=teach persist=true --strict-telemetry
  scripts/validate_store.py --strict-telemetry
  scripts/eval_cli.py suite=semantic preset=memory/sgc_rss n="$SIZES" seed="$SEEDS" run_id="$RUN_ID" mode=teach persist=true --strict-telemetry
  scripts/validate_store.py --strict-telemetry
  scripts/eval_cli.py suite=spatial preset=memory/smpd n="$SIZES" seed="$SEEDS" run_id="$RUN_ID" mode=teach persist=true --strict-telemetry
  scripts/validate_store.py --strict-telemetry
  ```
- Reference new guards/metrics in `DESIGN.md` briefly.

**Acceptance Criteria**
- File compiles as Markdown; commands run end-to-end.

---

## Rollout & Guardrails

### Task ROL.1 — **Feature flags**
- Add env flags:
  - `HIPPO_ORACLE=1` to enable oracle metrics,
  - `HIPPO_STRICT=1` to enable fail-fast guards,
  - `HIPPO_ENFORCE_SHORT_ANSWER=1` and `HIPPO_ENFORCE_UDLR=1`.

### Task ROL.2 — **Backward compatibility**
- If a legacy path lacks embeddings, log a warning, compute on the fly, and persist.

---

## Global Acceptance (end-to-end)
- On a fresh run with `N=50`, `SEED=1337`:
  - KG nodes/edges ≥ thresholds; embeddings present (non-zero ≥ 0.9).
  - Episodic keys non-zero ≥ 0.9; retrieval hit@k > 0 on at least one k.
  - Spatial validator active; `oracle_success_rate ≈ 1.0` on synthetic grids.
  - `audit_sample.jsonl` includes `injected_context`, `positions`, `source`.
  - `oracle_em`/`oracle_f1` reported and ≫ 0.
  - Smoke CI passes; strict telemetry produces green or actionable red.

## Quick Command Cheatsheet
```
pytest -q
bash scripts/ci_smoke_eval.sh
python scripts/eval_cli.py suite=semantic preset=memory/sgc_rss n=10 seed=1337 mode=teach persist=true --strict-telemetry
python scripts/validate_store.py --strict-telemetry
```

---
(End)