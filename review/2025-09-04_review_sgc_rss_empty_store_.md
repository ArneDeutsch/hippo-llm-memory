# SGC‑RSS “empty store” failure — deep dive & verdict

**Date:** 2025-09-04_1728  
**Repo:** `hippo-llm-memory-main.zip` (unzipped and inspected)  
**You hit:** `ValueError: empty store: runs/run_20250904/stores/sgc_rss/sgc_run_20250904/kg.jsonl` at **EVAL_PROTOCOL.md §4.2**

---

## TL;DR

- The **relational store stays empty** because the **teach path writes *no tuples*** into the KG.  
- Root cause: After the refactor, the teach path relies on `SchemaIndex.fast_track(...)` for a **placeholder tuple** but **no schemas are seeded**, so nothing is inserted.  
- This is **not a missing major implementation** (the KG, gating, retrieval, save/load are there). It’s a **wiring/regression** in the eval harness: dummy writes depend on an unconfigured schema.  
- Minimal fix: in `teach` for relational memory, when the gate says *insert*, **write directly via `kg.upsert(...)`** (bypass schema dependency). Optionally seed a default schema or pass config into `KnowledgeGraph` too.  
- After the fix, `kg.jsonl` will contain node/edge records and `validate_store.py` will pass.

---

## Reproduction & evidence

You ran (excerpt):
```bash
python scripts/eval_cli.py ... preset=memory/sgc_rss ... mode=teach persist=true ...
python scripts/validate_store.py --algo sgc_rss --kind kg
```
and got:
```
ValueError: empty store: runs/run_20250904/stores/sgc_rss/sgc_run_20250904/kg.jsonl
```

### What the code does today

1) **Teach loop (relational)** — `hippo_eval/eval/harness.py` (search for the relational block in `_evaluate(..., mode="teach")`):
- It constructs a **dummy tuple** `("a","rel","b","ctx", None, 1.0, 0)`.
- The gate decides; if `insert` and `mode=="teach"` it calls:
   ```py
   kg.schema_index.fast_track(tup, kg)
   # comment: avoid double counting (kg.ingest invokes the gate again)
   ```
- **But** `kg.schema_index` has **zero schemas** by default, so `fast_track` never inserts.
  - See `hippo_mem/relational/schema.py`: `score(...)` returns 1.0 only if `relation == schema.relation`. With **no schemas**, best score is 0.0 < threshold ⇒ **buffered**, not inserted.

2) **Where schemas come from**  
   - Preset `configs/eval/memory/sgc_rss.yaml` only toggles the gate. It doesn’t define schemas.  
   - `_init_modules(...)` in `hippo_eval/bench/__init__.py` builds a bare `KnowledgeGraph()` **without** passing config or seeding schemas.  
   - Result: **no relational writes** during `teach` ⇒ `kg.jsonl` saved but **empty** (validator looks for at least one non‑blank JSON line).

3) **Why this is a regression**  
   - The refactor split bench vs. eval harness and introduced a **schema‑dependent dummy write** (via `fast_track`).  
   - Previously, dummy data paths (bench) could seed content with `allow_dummy_stores=True`. In the eval harness that flag is **off by default** and no schema is seeded.  
   - Net effect: SGC‑RSS section in the protocol **silently produces an empty store**.

---

## Is implementation missing?

- **No critical gaps** in the relational components:
  - **KG**: `hippo_mem/relational/kg.py` supports `upsert`, `ingest`, retrieval, save/load.
  - **Gating**: `hippo_mem/relational/gating.py` is complete.
  - **Retrieval & adapter**: present and tested.
  - **Tuple extraction** exists (`hippo_mem/relational/tuples.py`) but is not yet **wired into the harness**; the harness currently uses a **placeholder tuple** in teach mode.

- The **evaluation harness wiring** is the issue (teach path depends on schemas that aren’t configured). The EVAL_PROTOCOL notes you shared (“pending integration”) were a conservative workaround, but we **can** make the section work now with a small code fix.

---

## Concrete fixes (ranked by impact/size)

1) **[Minimal, robust]** In relational **teach**: when gate accepts, **call `kg.upsert(...)`** directly with default types (`"entity","entity"`). This guarantees at least one edge (and two nodes) and keeps gate counters intact.  
   *Pros:* one‑line logic change; avoids schema dependency and double‑gating.  
   *Cons:* bypasses `SchemaIndex` for the placeholder tuple (fine for a dummy).

2) **[Nice to have]** Teach harness: if `kg.schema_index.schemas` is empty, **seed a trivial schema** (`("rel","rel")`) **before** calling `fast_track(...)`.  
   *Pros:* keeps the fast‑track semantics.  
   *Cons:* still extra coupling to placeholder relation name.

3) **[Structural]** `_init_modules(...)`: **pass relational config** (from `memory.relational`) into `KnowledgeGraph(config=...)` and **optionally seed schemas** from `memory.relational.schemas` if provided.  
   *Pros:* respects `configs/memory/relational.yaml` (thresholds, gnn_updates, prune).  
   *Cons:* small refactor in a shared helper; requires a couple of tests.

4) **Docs & tooling**  
   - Remove “pending integration” caveat for §4.2 in `EVAL_PROTOCOL.md`.  
   - `validate_store.py`: expand the **actionable hint** (“no schemas?”) when a relational store is empty after a teach run.  
   - Add a CLI test that `preset=memory/sgc_rss` + teach creates a **non‑empty** `kg.jsonl`.

---

## What to expect after the fix

- Running your §4.2 loop will **populate** `runs/<RID>/stores/sgc_rss/<SID>/kg.jsonl` with **nodes+edges** and a `store_meta.json` whose `source` is `"teach"` when the gate attempted inserts.  
- `scripts/validate_store.py --algo sgc_rss --kind kg` will pass.  
- Replay/test will also proceed (even if retrieval hits are low on small synthetic data).

---

## Sanity checks to run locally

```bash
# 1) Teach (same as §4.2)
python scripts/eval_cli.py suite=semantic preset=memory/sgc_rss n=50 seed=1337   run_id=run_20250904 model=$MODEL mode=teach persist=true   store_dir=runs/run_20250904/stores session_id=sgc_run_20250904   outdir=runs/run_20250904/memory/sgc_rss/semantic/50_1337 --strict-telemetry

# 2) Validate non-empty KG + meta
python scripts/validate_store.py --algo sgc_rss --kind kg

# 3) Replay (should not error)
python scripts/eval_cli.py ... mode=replay replay.cycles=3 --strict-telemetry

# 4) Test
python scripts/eval_cli.py ... mode=test --strict-telemetry
```

If you still see an empty store after applying the code change, print the first few lines:
```bash
sed -n '1,10p' runs/run_20250904/stores/sgc_rss/sgc_run_20250904/kg.jsonl
cat runs/run_20250904/stores/sgc_rss/sgc_run_20250904/store_meta.json
```

---

## Footnotes / related observations

- **Spatial (SMPD)** teach currently writes an observation node but **`PlaceGraph.save` emits an empty `spatial.jsonl` when `replay_samples == 0`**. That’s by design (episodic & spatial stores serialize data only after replay). The empty‑file validator is strict for relational only (`kg.jsonl` is written at teach).  
- `_init_modules(...)` not passing config means the KG’s default `schema_threshold` stays at `0.8` even if the YAML says `0.7`. Not the cause of your crash, but worth fixing.  
- The repo already has strong unit coverage for relational components; this is primarily a **harness/CI semantics** correction.
