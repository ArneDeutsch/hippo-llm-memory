# Codex Task Pack — Complete the New Validation Strategy (2025‑09‑07)

**Repo:** `hippo-llm-memory-main`  
**Context sources:**  
- Ground truth: `research/experiment-synthesis.md`  
- Latest review: `review/2025-09-06/run20250906_review.md`  
- Self‑audit: `run20250907_self_audit_review.md` (in project artifacts)  
- Current harness: `hippo_eval/eval/harness.py`  
- Runner: `hippo_eval/harness/runner.py`  
- Episodic store/gating: `hippo_mem/episodic/{store.py,gating.py,persistence.py}`  
- Relational KG/schema: `hippo_mem/relational/{kg.py,schema.py,tuples.py,backend.py}`  
- Spatial map: `hippo_mem/spatial/{map.py,gating.py}`  
- Validate script: `scripts/validate_store.py` and `hippo_mem/utils/stores.py`  
- Smoke/preflight tests: `tests/eval/{test_preflight*.py,test_smoke_memory_flow.py}`

> **Goal:** Remove dummy side effects, implement real `teach` ingestion for all memory suites, strengthen validators and tests, and update docs so GA criteria become achievable and meaningful.

---

## P0 — Remove telemetry side effects in the harness

**Why**: Dummy writes during telemetry corrupt acceptance metrics (zero‑norm episodic keys, 2‑node KG, artificial spatial growth).

**Files**: `hippo_eval/eval/harness.py`

**Edits**:
1. In the main evaluation loop, locate the gate‑probing blocks:
   - Episodic: search for `q = np.zeros(store.dim` and subsequent `store.write(q, ...)` inside the *gate/telemetry* section.
   - Relational: search for `tup = ("a", "rel", "b", ...` and subsequent `kg.upsert(` within the same *gate/telemetry* section.
   - Spatial: search for `decision = gate.decide(None, "ctx", graph)` and subsequent `graph.observe(`.

   **Remove the writes** (`store.write`, `kg.upsert`, `graph.observe`) from these telemetry blocks. Keep `GateCounters` increments only.

2. Ensure the dry‑run preflight path (`preflight_check` → `evaluate(..., dry_run=True)`) never mutates stores.

**Acceptance**:
- `pytest -q -k preflight` passes.
- Running a dry‑run (`python scripts/eval_model.py suite=episodic_cross_mem mode=teach n=1 dry_run=true persist=false ...`) does **not** create or change any store files (verify via mtime or absence).

**Guard test** (add to `tests/eval/test_preflight_gate.py`):
- Monkeypatch a spy on `EpisodicStore.write`, `KnowledgeGraph.upsert`, and spatial `PlaceGraph.observe` during `dry_run=True` and assert they were **not** called.


---

## P0 — Implement real *teach* ingestion for all suites

### A) Episodic (HEI‑NW)

**Files**: `hippo_eval/eval/harness.py`, `hippo_mem/episodic/gating.py`, `hippo_mem/retrieval/embed.py`

**Tasks**:
1. Add a helper in the harness (or a small `hippo_eval/eval/featurize.py`) to convert text → dense vector using `embed_text(text, dim=store.dim)`, L2‑normalize, then sparsify via `k_wta(query, k=store.k)` to produce a `DGKey`.
2. In `mode=="teach"`, for items from `episodic_cross_mem`, compute `key` from the **prompt** (or `context_key` field if present) and `store.write(key, TraceValue(...), context_key=...)` **only when gate.decide(...) returns `"insert"`**.
3. Remove any zero‑vector fallbacks. Ensure provenance is set (e.g., `"teach"` or dataset filename).

**Acceptance**:
- After `mode=teach` on `n=50, seed=1337`, `runs/<RID>/stores/hei_nw/<SID>/episodic.jsonl` exists and **≥90%** of keys have non‑zero norm.
- `scripts/validate_store.py --run_id <RID> --algo hei_nw --kind episodic --strict-telemetry` passes new checks (see P0 validators).

**Unit tests**:
- New `tests/algo/test_episodic_teach_ingest.py`: run a tiny teach (n=5) into a temp store; assert non‑zero ratio ≥ 0.9 and that values carry provenance/context.


### B) Semantic (SGC‑RSS)

**Files**: `hippo_eval/eval/harness.py`, `hippo_mem/relational/{tuples.py,schema.py,kg.py}`

**Tasks**:
1. During `mode=="teach"` for semantic suites, for each example:
   - Extract tuples with `extract_tuples(text, threshold=0.2)` from `tuples.py`. Use example text from the item (e.g., `item["prompt"]`); if the dataset uses a different key (e.g., `facts`), iterate those strings.
   - Route via `SchemaIndex.fast_track(t, kg)`; non‑accepted tuples remain in `schema_index.episodic_buffer`.
2. After processing the batch/split, call `schema_index.flush(kg)` to promote eligible buffered tuples.
3. Ensure the KG save step is invoked with a **guaranteed embeddings backfill** (see P0 change in KG below).

**Acceptance**:
- After `mode=teach` on `n=50`, the KG has **≥100 nodes** and **≥100 edges** (with embeddings coverage ≥0.9 when enabled).

**Unit tests**:
- New `tests/algo/test_relational_teach_ingest.py`: ingest a few sentences; assert KG node/edge counts increase and at least one tuple is fast‑tracked.


### C) Spatial (SMPD)

**Files**: `hippo_eval/eval/harness.py`, `hippo_mem/spatial/{map.py,gating.py}`

**Tasks**:
1. Replace telemetry‑only `graph.observe("ctx")` with **trajectory ingestion** during `mode=="teach"` by iterating per‑example steps/observations (fields as produced by the dataset; if absent, derive a simple context sequence and call `observe` per step).
2. Keep gating decisions, but writes only on `"insert"`.

**Acceptance**:
- `runs/<RID>/stores/smpd/<SID>/spatial.jsonl` exists and contains >0 nodes/edges **derived from the data**, not a single synthetic observation.


---

## P0 — Strengthen validators and CLI ergonomics

**Files**: `scripts/validate_store.py`, `hippo_eval/eval/store_utils.py` (if present), `hippo_mem/utils/stores.py`

**Tasks**:
1. Extend `scripts/validate_store.py` with optional thresholds:
   - `--expect-nodes INT`, `--expect-edges INT` for `kind="kg"`
   - `--expect-embedding-coverage FLOAT` for KG (node+edge embeddings)
   - `--expect-nonzero-ratio FLOAT` for episodic
2. Implement reading logic:
   - Episodic: parse `episodic.jsonl` or SQLite via `TracePersistence.iter_all()`; compute fraction of keys with `||k|| > 0`.
   - KG: read via `KnowledgeGraph.backend` (`SQLiteBackend`) or `kg.jsonl` to derive counts; compute embedding coverage when available.
3. Return non‑zero exit with a clear error if expectations are not met.

**Acceptance**:
- The failing cases from Codex’s GA check now get **actionable** errors (e.g., “non‑zero key ratio 0.49 < 0.90”).


---

## P0 — Schema defaults or schemaless fallback

**Files**: `configs/memory/relational.yaml`, `hippo_mem/relational/schema.py`

**Tasks**:
1. **Config seed** (preferred for fixtures): add default schemas
   ```yaml
   # configs/memory/relational.yaml
   schemas:
     - { name: "likes", relation: "likes" }
     - { name: "buy", relation: "buy" }
     - { name: "sold_at", relation: "sold_at" }
     - { name: "is_in", relation: "is_in" }
   ```
2. **Fallback**: in `SchemaIndex.fast_track`, if `self.schemas` is empty, treat as schemaless by setting `best_score = 1.0` before threshold checks.

**Acceptance**:
- Tuple routing proceeds even when schemas are not provided in the preset.


---

## P0 — Guarantee KG embedding backfill

**Files**: `hippo_mem/relational/kg.py`

**Tasks**:
1. Add a config gate (default **on**) to ensure embeddings on save:
   - Accept a parameter `ensure_embeddings: bool = True` (or read `memory.relational.embeddings.auto_backfill: true` from config where available).
   - In `save(...)`, if `ensure_embeddings`, call `_backfill_embeddings()` before writing files.
2. Expose a public method if needed (avoid using a private `_backfill_embeddings` from outside).

**Acceptance**:
- After teach, running `validate_store.py --expect-embedding-coverage 0.9` passes on KG.


---

## P1 — Oracle metrics as a first‑class flag

**Files**: `scripts/eval_model.py`, `hippo_eval/eval/harness.py`, tests under `tests/eval/test_oracle_readers.py`

**Tasks**:
1. Add CLI `--oracle` boolean flag to `scripts/eval_model.py`. Thread it through to the harness by setting an explicit config field (e.g., `cfg.compute.oracle=true`) instead of relying on `HIPPO_ORACLE`.
2. In `harness.evaluate/_evaluate`, prefer the config flag; maintain env var as a fallback for backward compatibility.
3. Ensure metrics include `oracle_em` and `oracle_f1` when enabled.

**Acceptance**:
- `pytest -q -k oracle_readers` passes without needing to set `HIPPO_ORACLE` env var (but should still work when set).


---

## P1 — Preflight & smoke alignment (no state change; positive data checks)

**Files**: `hippo_eval/eval/harness.py` (function `preflight_check`), `tests/eval/test_smoke_memory_flow.py`

**Tasks**:
1. **Preflight**: keep baseline presence and store existence checks, but **do not require successful gate inserts** in dry‑run. Only assert that gate **attempts > 0** (already present) and that, in `mode=test` with `store_dir/session_id`, the expected store file exists and is **non‑empty**.
2. **Smoke test**: extend `test_smoke_memory_flow` to assert:
   - Episodic non‑zero ratio ≥ 0.9 using the validator.
   - Semantic KG nodes ≥ 100 and edges ≥ 100 using the validator (invoke the script via subprocess).
   - Spatial store has >0 nodes/edges.

**Acceptance**:
- `pytest -q` passes locally; CI job `scripts/ci_smoke_eval.sh` succeeds using the new checks.


---

## P1 — Refactor harness for clarity (small, targeted)

**Files**: `hippo_eval/eval/harness.py`

**Tasks**:
1. Extract small helpers:
   - `_episodic_key_from_text(text, dim, k) -> DGKey`
   - `_ingest_episodic(item, modules, gate, strict)`
   - `_ingest_semantic(item, modules, gate, strict)`
   - `_ingest_spatial(item, modules, gate, strict)`
2. Add docstrings explaining side‑effect boundaries (no writes in telemetry; writes only within `mode=="teach"`/`"replay"` decision paths).

**Acceptance**:
- Lints clean; cyclomatic complexity reduced around the main loop; new helpers are unit‑tested where practical.


---

## P2 — Minimal EVAL_PROTOCOL.md & docs refresh

**Files**: `EVAL_PROTOCOL.md`, `EVAL_PLAN.md`, `DESIGN.md`

**Tasks**:
1. Rewrite `EVAL_PROTOCOL.md` as a minimal, parametric runbook (SIZES, SEEDS, RUN_ID). No deprecated calls; list per‑suite sequence: `teach → (optional replay) → test → validate`.
2. Add explicit validator invocations with sane defaults:
   ```bash
   python scripts/validate_store.py --run_id $RUN_ID --algo hei_nw --kind episodic --expect-nonzero-ratio 0.9
   python scripts/validate_store.py --run_id $RUN_ID --algo sgc_rss --kind kg --expect-nodes 100 --expect-edges 100 --expect-embedding-coverage 0.9
   python scripts/validate_store.py --run_id $RUN_ID --algo smpd --kind spatial
   ```
3. Note the `--oracle` flag as an optional upper‑bound metric.


---

## Verification matrix (single‑GPU friendly)

```bash
# Episodic teach → validate
python scripts/eval_cli.py suite=episodic_cross_mem preset=memory/hei_nw n=50 seed=1337   run_id=run20250907 model=models/tiny-gpt2 mode=teach persist=true   store_dir=runs/run20250907/stores session_id=hei_run20250907 --strict-telemetry

python scripts/validate_store.py --run_id run20250907 --algo hei_nw --kind episodic   --expect-nonzero-ratio 0.9 --strict-telemetry

# Semantic teach → validate
python scripts/eval_cli.py suite=semantic_mem preset=memory/sgc_rss n=50 seed=1337   run_id=run20250907 model=models/tiny-gpt2 mode=teach persist=true   store_dir=runs/run20250907/stores session_id=sgc_run20250907 --strict-telemetry

python scripts/validate_store.py --run_id run20250907 --algo sgc_rss --kind kg   --expect-nodes 100 --expect-edges 100 --expect-embedding-coverage 0.9 --strict-telemetry

# Spatial teach → validate
python scripts/eval_cli.py suite=spatial_multi preset=memory/smpd n=50 seed=1337   run_id=run20250907 model=models/tiny-gpt2 mode=teach persist=true   store_dir=runs/run20250907/stores session_id=smpd_run20250907 --strict-telemetry

python scripts/validate_store.py --run_id run20250907 --algo smpd --kind spatial --strict-telemetry
```

---

## Notes for Codex

- Prefer small, composable commits: (1) remove side effects, (2) implement teach ingestion per suite, (3) validators, (4) schemas/backfill, (5) oracle flag, (6) tests & docs.
- Keep telemetry pure. When adding counters or probes, **never** write to stores.
- Align names with existing modules (`WriteGate`, `SchemaIndex.fast_track`, `PlaceGraph.observe`).

