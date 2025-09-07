# Self‑Audit Review — Why GA Wasn’t Reached (2025‑09‑07)

**Repo:** `hippo-llm-memory-main`  
**Ground truth:** `research/experiment-synthesis.md`  
**User review:** `review/2025-09-06/run20250906_review.md`  
**Codex tasks:** `tasks/2025-09-06/run20250906_codex_tasks.md`

---

## Executive summary

Global Acceptance (GA) failed for two structural reasons in the evaluation harness, not (primarily) in the core memory modules:

1) **Telemetry code is mutating stores with placeholders.**  
   During `teach`, the harness writes *fake* records solely to bump gate counters:
   - **Episodic:** it writes a **zero vector** to the store (`store.write(q=0, ...)`), which drives the *non‑zero key ratio* down to ~0.5.  
   - **Semantic:** it unconditionally **upserts a dummy edge** (“a —rel→ b”), which explains the “**2 nodes / 1 edge**” KG even when real ingestion is absent.

2) **There is no real *teach* ingestion path implemented for episodic or semantic.**  
   The harness never turns dataset facts into store writes. As a result, stores remain empty (except for the placeholders above), retrieval has nothing to return, and all downstream metrics collapse.

These two issues fully explain Codex’s results: episodic fails the ≥90 % non‑zero key check; semantic fails the ≥100 nodes/edges check; spatial “passes” only because the placeholder `graph.observe(ctx)` creates a chain of nodes/edges.

---

## What we missed vs. the ground truth

The ground‑truth doc requires that **teach** runs *populate the stores* while retrieval is disabled, and **test** runs read from those persisted stores. Concretely:

- **HEI‑NW:** write sparse/dense episodic traces with k‑WTA, then consolidate.  
- **SGC‑RSS:** extract tuples → schema fast‑track → KG writes (with embeddings), plus buffer flush into KG.  
- **SMPD:** ingest trajectories into a place graph and optionally replay.

In our harness:
- No tuple extraction / schema routing is called during teach.  
- No episodic key encoding from text is done.  
- Telemetry code performs writes with dummy content.  
- Embeddings backfill isn’t invoked by default, so even when edges appear, embeddings stay empty.

---

## Concrete root causes (with file/function anchors)

1) **Placeholder writes during teach** (must be side‑effect‑free):
   - File: `hippo_eval/eval/harness.py`, function: `_evaluate(...)`  
     - Episodic block writes `q = np.zeros(store.dim)` → `store.write(q, ...)` when `mode == "teach"` (telemetry section).  
     - Relational block writes `kg.upsert("a","rel","b",...)` under the same condition.  
     - Spatial block calls `graph.observe(context)`.
   - Effect: corrupts acceptance metrics (zero‑norm episodic keys; minimal KG size; artificial spatial growth).

2) **No ingestion for teach**:
   - `episodic_cross_mem`: items have `{ "fact": str, "context_key": str }` (see `hippo_eval/tasks/generators.py`). No code converts `fact/context_key` to a key vector + value → **no real episodic writes**.
   - `semantic_mem`: items have `facts: [{ "text": ... , "schema_fit": bool, ...}]`. No call to `extract_tuples()` / `SchemaIndex.fast_track()` → **no KG growth**.
   - `spatial_multi`: episodes include moves; current code doesn’t ingest trajectories; only the telemetry `observe()` fires.

3) **Schemas missing by default** (semantic):
   - `configs/memory/relational.yaml` leaves `schemas:` commented out. With an empty schema index, `SchemaIndex.fast_track()` rarely inserts; the gate often routes to episodic.
   - We need either default schemas matching the generator (`buy`, `sold_at`, `is_in`) or a **schemaless fallback** (“match‑all if schema set is empty”).

4) **Embeddings backfill not guaranteed**:
   - Even after KG writes, node/edge embeddings remain absent unless `_backfill_embeddings()` is called (see `KnowledgeGraph` implementation). This hurts retrieval quality.

5) **Strict telemetry conflates “gate exercise” with state changes**:
   - GA wants *attempt counts* and *store health checks*, but our gate‑exercise code changes stores. It should only update counters, not write any records.

---

## Acceptance criteria gaps

- **Episodic:** The ≥90 % non‑zero key check is correct, but it’s currently testing **our placeholder writes**, not the real writes. We need real keys from text → embeddings → k‑WTA → persist.
- **Semantic:** The ≥100 nodes/edges guard is reasonable for `n=50` (the semantic generator yields multiple facts per item). But our code never ingests facts, so the guard will keep failing.
- **Oracle readers:** The harness has an env‑flagged oracle (`HIPPO_ORACLE`) but GA talks about verifying memory function *independent of LM*. We should expose this as a first‑class `--oracle` switch and record `oracle_em/f1` alongside normal metrics.
- **EVAL_PROTOCOL.md:** Slimmed, but it still references old paths and does not state the pre/post replay expectations per suite nor the store layout assertions tailored to each algorithm.

---

## What exactly is missing to “complete” the new validation strategy

1) **Remove all telemetry side effects** in `_evaluate()` (no writes during gate probing). Keep counters only.
2) **Implement teach‑time ingestion** for each suite:
   - **episodic_cross_mem:** build a stable non‑zero key from text:
     - Tokenize fact → mean‑pool hidden state or (CPU‑friendly) hashed bag‑of‑words to `store.dim`, L2‑normalise, then apply k‑WTA.  
     - `store.write(key, value=TraceValue(provenance=..., ...), context_key)` behind `WriteGate` (if enabled).
   - **semantic_mem:** for each `fact["text"]`:
     - `extract_tuples(text, threshold=0.2)` → `SchemaIndex.fast_track(t, kg)`; if not inserted, leave in `episodic_buffer`.  
     - After the split: `schema_index.flush_buffer(kg)` and call `kg._backfill_embeddings()` (or a public wrapper) before `kg.save(...)`.
   - **spatial_multi:** ingest trajectories from teach episodes (sequence of observations) using `graph.observe(ctx)` for each step, not the telemetry placeholder.
3) **Default schemas or schemaless mode**:
   - Add to `configs/eval/memory/sgc_rss.yaml`:
     ```yaml
     memory:
       relational:
         schemas:
           - { name: buy, relation: buy }
           - { name: sold_at, relation: sold_at }
           - { name: is_in, relation: is_in }
     ```
   - Additionally, modify `SchemaIndex.fast_track` to **treat an empty schema set as “pass”** (score=1.0) so ingestion doesn’t stall when schemas are not provided.
4) **Enable embeddings by default** for KG:
   - On each `upsert`, compute or defer and **guarantee** backfill during `save()`. Add a config switch `embeddings.auto_backfill=true` and default it to true.
5) **Make the oracle path first‑class**:
   - Add CLI flag `--oracle` → records `oracle_em/f1` in metrics (no env var dependency). Update `rollup` to include oracle metrics as an upper bound. 
6) **Preflight & smoke alignment**:
   - Preflight should only check **baselines** and **store layout presence**, not mutate stores or require gate success in a dry run.  
   - Smoke tests should assert: (a) store growth > 0 for teach; (b) episodic non‑zero ratio ≥ 0.9; (c) semantic KG nodes/edges ≥ threshold; (d) spatial JSONL present with >0 nodes/edges; (e) retrieval requests > 0 in test.
7) **Update EVAL_PROTOCOL.md (minimal, parametric)**:
   - Loop over `SIZES`/`SEEDS`; for each suite: `teach → (replay) → test`; run `scripts/validate_store.py` with suite‑specific checks; run `hippo_eval.reporting.rollup` at the end.

---

## Suggested code changes (surgical)

1) **`hippo_eval/eval/harness.py::_evaluate` — remove placeholder writes**
   - Delete the `store.write(q, ...)`, `kg.upsert("a",...)`, and `graph.observe(...)` in the *telemetry* block. Only increment counters.

2) **`hippo_eval/eval/harness.py` — add teach ingestion**
   - Add helper functions:
     - `_episodic_key_from_text(text: str, dim: int) -> np.ndarray` (hash‑BOW, L2‑norm, k‑WTA).
     - `_ingest_episodic(item, modules, gate, strict)`.
     - `_ingest_semantic(item, modules, gate, strict)`.
     - `_ingest_spatial(item, modules, gate, strict)`.
   - In the main loop, when `mode=="teach"` and module present, call the appropriate ingestor **instead of** the telemetry writes.

3) **`hippo_mem/relational/schema.py` — schemaless fallback**
   - If `self.schemas` is empty, set `best_score = 1.0` before thresholding.

4) **`hippo_mem/relational/kg.py` — guaranteed embedding backfill**
   - Add `save(..., ensure_embeddings=True)` to call `_backfill_embeddings()` if any node/edge lacks embeddings.

5) **`configs/eval/memory/sgc_rss.yaml` — seed schemas**
   - Add `memory.relational.schemas` block matching the generator’s relations.

6) **`scripts/validate_store.py` — stricter semantic checks**
   - Verify `nodes>=100 && edges>=100` for `kind="kg"` and `--strict-telemetry` (configurable thresholds). Verify `embedding_coverage>=0.9` when enabled.

7) **`scripts/eval_model.py` — oracle flag passthrough**
   - Add `--oracle` to env or config and set `HIPPO_ORACLE` equivalently (or remove env usage entirely and plumb config).

8) **Tests**  
   - Add unit tests for: (a) no side effects in telemetry; (b) teach ingestion populates stores; (c) schemaless fallback; (d) embedding backfill on save; (e) oracle metrics emitted.

---

## Verification checklist (single‑GPU friendly)

```bash
# 1) Episodic teach should populate and pass non‑zero check
python scripts/eval_cli.py suite=episodic_cross_mem preset=memory/hei_nw   n=50 seed=1337 run_id=run20250907 model=models/tiny-gpt2 mode=teach   store_dir=runs/run20250907/stores session_id=hei_run20250907 persist=true --strict-telemetry
python scripts/validate_store.py --run_id run20250907 --algo hei_nw --kind episodic --strict-telemetry

# 2) Semantic teach should populate KG with embeddings
python scripts/eval_cli.py suite=semantic_mem preset=memory/sgc_rss   n=50 seed=1337 run_id=run20250907 model=models/tiny-gpt2 mode=teach   store_dir=runs/run20250907/stores session_id=sgc_run20250907 persist=true --strict-telemetry
python scripts/validate_store.py --run_id run20250907 --algo sgc_rss --kind kg   --expect-nodes 100 --expect-edges 100 --expect-embedding-coverage 0.9 --strict-telemetry

# 3) Spatial teach should write a non‑empty spatial.jsonl
python scripts/eval_cli.py suite=spatial_multi preset=memory/smpd   n=50 seed=1337 run_id=run20250907 model=models/tiny-gpt2 mode=teach   store_dir=runs/run20250907/stores session_id=smpd_run20250907 persist=true --strict-telemetry
python scripts/validate_store.py --run_id run20250907 --algo smpd --kind spatial --strict-telemetry

# 4) Run test passes with retrieval > 0 and oracle metrics available
python scripts/eval_cli.py suite=semantic_mem preset=memory/sgc_rss   n=50 seed=1337 run_id=run20250907 model=models/tiny-gpt2 mode=test --oracle   store_dir=runs/run20250907/stores session_id=sgc_run20250907 --strict-telemetry
```

---

## Prioritised TODOs (what Codex tasks missed)

- **P0 (blocker):** Remove telemetry side effects; implement true teach ingestion for episodic/semantic/spatial.  
- **P0:** Seed schemas or add schemaless fallback; ensure KG embedding backfill.  
- **P1:** Make oracle a first‑class flag; tighten validators; adjust smoke tests to check store growth rather than dry‑run gate success.  
- **P2:** Document the minimal EVAL_PROTOCOL with suite‑specific checks and thresholds; regenerate runs.

With these fixes, GA should be reachable: episodic will pass the ≥90 % non‑zero key check (real keys), semantic KG will easily exceed the ≥100 nodes/edges threshold (generator yields multiple facts per item), and spatial will persist a valid map built from real trajectories.
