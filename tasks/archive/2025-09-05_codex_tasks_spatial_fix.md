# Codex Tasks — Fix SMPD empty store & strengthen validation

Apply in order. Each task includes concrete file edits and acceptance checks.

---

## T1 — Remove blanking of `spatial.jsonl` when no replay

**Why:** Teach‑only runs currently write an empty `spatial.jsonl`, tripping validation.

**Edit:** `hippo_mem/spatial/map.py`

- Replace the early‑return block with unconditional JSONL writing (like KG):

```diff
 def save(
     self,
     directory: str,
     session_id: str,
     fmt: str = "jsonl",
     replay_samples: int = 0,
     gate_attempts: int = 0,
 ) -> None:
-    """Save map under ``directory/session_id``."""
+    \"\"\"Save map under ``directory/session_id``.\"\"\"
     if fmt not in {"jsonl", "parquet"}:
         raise ValueError(f"Unsupported format: {fmt}")

     path = Path(directory) / session_id
     path.mkdir(parents=True, exist_ok=True)
     self._write_meta(path, replay_samples, gate_attempts)

-    file = path / "spatial.jsonl"
-    if replay_samples <= 0:
-        io.atomic_write_file(file, lambda tmp: open(tmp, "w", encoding="utf-8").write(""))
-        return
-
-    if fmt == "jsonl":
-        self._save_jsonl(path)
-    else:  # fmt == "parquet"
-        self._save_parquet(path)
+    if fmt == "jsonl":
+        # Always materialize JSONL (meta + nodes/edges), even for teach‑only.
+        self._save_jsonl(path)
+    else:  # fmt == "parquet"
+        self._save_parquet(path)
```

**Acceptance:**  
- Run §4.3 *teach* step (n small is fine) and verify that `spatial.jsonl` contains ≥1 non‑blank line.

---

## T2 — Broaden the store‑validation hint (include SMPD)

**Why:** The CLI error points to SGC‑RSS only.

**Edit:** `scripts/validate_store.py`

- Change the hint string to mention both relational and spatial:

```diff
- hint: for SGC-RSS ensure the teach path actually writes tuples (gate accepts *and* either schemas are seeded or direct upsert is used).
+ hint: teach path must persist data.
+  - SGC-RSS: ensure tuples are written (gate accepts and schemas are seeded or direct upsert is used).
+  - SMPD: ensure the spatial map writes nodes/edges (no blank JSONL on teach-only).
```

**Acceptance:**  
- Triggering validation on an empty file shows the expanded hint.

---

## T3 — Add a regression test for spatial teach persistence

**Why:** Prevent reintroducing the “empty on teach” behavior.

**New file:** `tests/algo/test_spatial_teach_persists.py`

```python
from omegaconf import OmegaConf
from pathlib import Path
from hippo_eval.eval.harness import evaluate
from hydra.utils import to_absolute_path

def test_spatial_teach_writes_jsonl(tmp_path: Path):
    cfg = OmegaConf.create({
        "suite": "spatial",
        "preset": "memory/smpd",
        "n": 2,
        "seed": 0,
        "mode": "teach",
        "persist": True,
        "store_dir": str(tmp_path/"stores"),
        "session_id": "smpd_test",
        "outdir": str(tmp_path/"out"),
        "strict_telemetry": True,
    })
    evaluate(cfg, tmp_path)  # in-process
    f = tmp_path/"stores"/"smpd"/"smpd_test"/"spatial.jsonl"
    assert f.exists()
    # Must contain at least one non-blank line (meta or node)
    assert any(line.strip() for line in f.read_text(encoding="utf-8").splitlines())
```

**Markers:** none (default fast).

**Acceptance:**  
- `pytest -q -k spatial_teach_writes_jsonl` passes in < 1s.

---

## T4 — (Optional) More realistic context in spatial teach

**Why:** The current teach path always observes `"ctx"`, so no edges form across multiple steps.

**Edit:** `hippo_eval/eval/harness.py` (spatial teach branch)

- Replace `graph.observe("ctx")` with a stable but varying token (e.g., `f"ctx_{i}"` or derived from `item.qid`) so consecutive observations yield edges:

```diff
- graph.observe("ctx")
+ context = f"ctx_{i}" if hasattr(item, "qid") else f"ctx_{i}"
+ graph.observe(context)
```

**Acceptance:**  
- After a tiny teach run (`n=3`), `spatial.jsonl` contains at least one `"type": "edge"` record.

---

## T5 — Document the §4.3 expectation in `EVAL_PROTOCOL.md`

**Why:** Mirror the §4.2 note so users know what “populated” means for spatial.

**Edit:** `EVAL_PROTOCOL.md` under **4.3) SMPD (spatial)**

- Add a note:

> The teach step must persist a **non-empty** `spatial.jsonl` under `stores/smpd/<SID>/`.  
> This file should at least contain the meta record and one node after a gate‑accepted observation. Replay is **not** required for the file to be non‑empty.

**Acceptance:**  
- `grep` shows the new paragraph; the protocol reads cleanly.

---

## T6 — Add a CLI validation test for SMPD

**Why:** Symmetry with the relational guard; keeps the CLI path healthy.

**New file:** `tests/cli/test_smpd_store_population.py`

```python
import subprocess, sys, os
from pathlib import Path

def test_smpd_store_population(tmp_path: Path):
    run_id = "run_ci"
    stores = tmp_path/"stores"
    sid = "smpd_run_ci"
    # teach
    subprocess.run([
        sys.executable, "scripts/eval_cli.py",
        "suite=spatial", "preset=memory/smpd", "n=2", "seed=0",
        f"run_id={run_id}", "mode=teach", "persist=true",
        f"store_dir={stores}", f"session_id={sid}",
        "strict_telemetry=true"
    ], check=True)
    # validate
    subprocess.run([
        sys.executable, "scripts/validate_store.py",
        "--algo", "smpd", "--kind", "spatial",
        "--stores", str(stores), "--sid", sid, "--run-id", run_id
    ], check=True)
```
**Markers:** falls under `tests/cli/` ⇒ auto‑marked `integration` (skipped on PR; covered nightly).

**Acceptance:**  
- `pytest -q --runintegration -k smpd_store_population` passes quickly.
