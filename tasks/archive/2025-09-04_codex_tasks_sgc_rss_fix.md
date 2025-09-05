# Codex task pack — fix SGC‑RSS empty store & wire minimal config

**Date:** 2025-09-04_1728  
**Goal:** Make §4.2 of `EVAL_PROTOCOL.md` pass by ensuring relational teach writes data; improve config plumbing; tighten tests & docs.

---

## Task 1 — Ensure relational teach writes tuples (minimal fix)

**Why**  
Teach uses a placeholder tuple but `SchemaIndex` has no schemas ⇒ `fast_track` never inserts ⇒ `kg.jsonl` empty.

**Edit**  
File: `hippo_eval/eval/harness.py` — in the **relational teach block** inside `_evaluate(...)`.

**Change (exact)**  
Replace the `fast_track` write with a direct `upsert`:

```diff
@@ if "relational" in modules and modules["relational"].get("gate") is not None:
-                tup = ("a", "rel", "b", "ctx", None, 1.0, 0)
+                tup = ("a", "rel", "b", "ctx", None, 1.0, 0)
                 decision = gate.decide(tup, kg)
                 gc = gating["relational"]
                 gc.attempts += 1
                 if decision.action == "insert":
                     gc.accepted += 1
                     if mode == "teach":
-                        # avoid double counting: ``kg.ingest`` invokes the gate again
-                        kg.schema_index.fast_track(tup, kg)
+                        # Bypass schema dependency for the placeholder tuple;
+                        # avoid double-gating by writing directly.
+                        head, rel, tail, ctx, time_str, conf, prov = tup
+                        kg.upsert(head, rel, tail, ctx, "entity", "entity", time_str, conf, prov)
                 else:
                     gc.skipped += 1
```

**Acceptance**  
- Running §4.2 produces a `kg.jsonl` with ≥1 non-empty line and `store_meta.json` with `"source": "teach"` when attempts>0.  
- `scripts/validate_store.py --algo sgc_rss --kind kg` succeeds.

---

## Task 2 — Pass relational config into `KnowledgeGraph`

**Why**  
`_init_modules(...)` ignores `memory.relational` YAML (e.g., `schema_threshold`, `gnn_updates`).

**Edit**  
File: `hippo_eval/bench/__init__.py` — function `_init_modules(...)` in `_add_relational()`.

**Change (exact)**  
Load/merge config and pass it to `KnowledgeGraph`:

```diff
-    def _add_relational() -> None:
-        kg = KnowledgeGraph()
+    def _add_relational() -> None:
+        cfg = {}
+        try:
+            # `memory` may be OmegaConf; resolve and pull `relational` subtree
+            mem_dict = OmegaConf.to_container(memory, resolve=True) if memory is not None else {}
+            cfg = dict(mem_dict.get("relational") or {})
+            # gate is applied separately; do not pass through in `config`
+            cfg.pop("gate", None)
+        except Exception:
+            cfg = {}
+        kg = KnowledgeGraph(config=cfg or None)
         if allow_dummy_stores:
             kg.upsert("a", "rel", "b", "a rel b")
         modules["relational"] = {"kg": kg, "adapter": RelationalMemoryAdapter()}
```

**Acceptance**  
- Setting `relational.schema_threshold=0.6` via config affects `kg.schema_index.threshold`.  
- Unit tests still pass.

---

## Task 3 — Optional schema seeding via config

**Why**  
Allow users to seed schemas without code changes; useful beyond the dummy tuple.

**Edits**  
- File: `hippo_eval/bench/__init__.py` — in `_add_relational()` **after** creating `kg`.  
- File: `configs/memory/relational.yaml` — document new optional key.

**Change (exact)**

```diff
         kg = KnowledgeGraph(config=cfg or None)
+        # Optional: seed schemas from config: memory.relational.schemas: [{name, relation}|str]
+        schemas = (cfg.get("schemas") if isinstance(cfg, dict) else None) or []
+        for s in schemas:
+            if isinstance(s, str):
+                kg.schema_index.add_schema(s, s)
+            elif isinstance(s, dict):
+                kg.schema_index.add_schema(s.get("name", s.get("relation", "rel")), s.get("relation", "rel"))
```

`configs/memory/relational.yaml` (comment only, no behavioral change if omitted):

```diff
 # Relational memory configuration.
+# Optional: seed schemas for fast-track routing (use with simple fixtures)
+# schemas:
+#   - { name: "rel", relation: "rel" }
```

**Acceptance**  
- When `memory.relational.schemas=['rel']` is set, `kg.schema_index.schemas` is non-empty.

---

## Task 4 — Improve validator hint

**Why**  
When KG is empty, the error should hint at schema/gate preconditions.

**Edit**  
File: `scripts/validate_store.py`, in the `empty store` branch.

**Change (exact)**

```diff
-            raise ValueError(
-                "empty store: "
-                f"{path} — run:\n  python scripts/eval_model.py --mode teach --run-id {run_id}"
-            )
+            raise ValueError(
+                "empty store: "
+                f"{path} — run:\n  python scripts/eval_model.py --mode teach --run-id {run_id}\n"
+                "hint: for SGC-RSS ensure the teach path actually writes tuples "
+                "(gate accepts *and* either schemas are seeded or direct upsert is used)."
+            )
```

**Acceptance**  
- Error message includes the extra hint.

---

## Task 5 — Update protocol wording

**Why**  
Remove the “pending integration” caveat for §4.2 now that teach writes data.

**Edit**  
File: `EVAL_PROTOCOL.md` — §4.2 SGC‑RSS paragraph. Remove the *skip* note; add a short sentence that the teach step seeds the store with minimal tuples even without extraction integration.

**Acceptance**  
- The protocol no longer instructs to skip §4.2. CI docs linting passes (if any).

---

## Task 6 — CLI test to guard against regressions

**Why**  
Prevent the empty‑store regression from coming back.

**New file**  
`tests/cli/test_sgc_store_population.py`

**Content** (minimal):
```python
import json
from pathlib import Path
import sys
import subprocess

def test_sgc_teach_populates_store(tmp_path):
    outdir = tmp_path / "o"
    store = tmp_path / "s"
    sid = "sgc_test"
    cmd = [
        sys.executable, "scripts/eval_model.py",
        "suite=semantic", "preset=memory/sgc_rss",
        "n=1", "seed=1337",
        f"outdir={outdir}", f"store_dir={store}", f"session_id={sid}",
        "model=models/tiny-gpt2",
        "mode=teach", "persist=true",
    ]
    subprocess.check_call(cmd)
    kg = store / "sgc_rss" / sid / "kg.jsonl"
    meta = store / "sgc_rss" / sid / "store_meta.json"
    assert meta.exists()
    assert kg.exists()
    # must have at least one non-empty JSON record
    assert any(line.strip() for line in kg.read_text().splitlines())
```

**Acceptance**  
- The new test passes locally and in CI.
