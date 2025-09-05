# Review — Step 4.3 **SMPD (spatial)** fails with “empty store”

**Date:** 2025-09-05  
**Context:** After your latest ZIP update, §4.2 (SGC‑RSS) now passes. §4.3 (SMPD) fails at the *teach → validate_store* boundary with:

```
empty store: runs/<RUN_ID>/stores/smpd/<SID>/spatial.jsonl
```

## Root cause

In `hippo_mem/spatial/map.py`, the `PlaceGraph.save(...)` early‑returns with an **explicitly empty** `spatial.jsonl` whenever `replay_samples <= 0`:

```python
# map.py (current)
file = path / "spatial.jsonl"
if replay_samples <= 0:
    io.atomic_write_file(file, lambda tmp: open(tmp, "w", encoding="utf-8").write(""))
    return
```

In the §4.3 *teach* step we **don’t run replay** yet, so `replay_samples == 0`. Even though the harness does call `graph.observe("ctx")` after the spatial gate accepts, the subsequent `.save(...)` still truncates `spatial.jsonl` to an empty file. `scripts/validate_store.py` then raises the “empty store” error because it only checks for *any non‑blank line* in the file.

This differs from the relational path (`hippo_mem/relational/kg.py`), where `kg.save(...)` always writes data records to `kg.jsonl` regardless of replay, and uses `store_meta.json`’s `"source": "teach"` to signal teach‑only population.

## Evidence from the tree

- `hippo_eval/eval/harness.py` (spatial teach branch):
  - After a spatial gate **insert** decision in *teach* mode, it calls `graph.observe("ctx")`. ✔️
  - On persist, it calls `modules["spatial"]["map"].save(..., replay_samples=replay_samples, gate_attempts=spat_attempts)`. ✔️
- `hippo_mem/spatial/map.py`:
  - `save(...)` **empties and returns** if `replay_samples <= 0`. ❌
  - `_write_meta(...)` correctly sets `"source": "teach"` when `gate_attempts > 0`. ✔️
  - `_save_jsonl(...)` writes a **meta** record and **node/edge** records — but is **never reached** without replay. ❌
- `scripts/validate_store.py`:
  - Treats a file with **no non‑blank lines** as empty and errors out. (Same behavior that caught SGC‑RSS before it was fixed.) ✔️

## Minimal change that fixes §4.3

Make `PlaceGraph.save(...)` mirror the relational KG behavior:

- **Always** write `spatial.jsonl` (at least the meta line; if gate accepted, there will also be at least one node line) — do **not** blank the file when `replay_samples == 0`.
- Keep `store_meta.json` `"source"` logic as is (`teach` vs `replay` vs `stub`).

This change preserves replay semantics while making the §4.3 *teach* validation pass.

## Optional robustness improvements

- In the spatial teach path, vary the observed context (e.g., include item id) instead of the hardcoded `"ctx"` so successive observations form edges; not required for the fix, but improves realism.
- Update `scripts/validate_store.py` hint message to include SMPD alongside SGC‑RSS.
- Add a regression test: teach‑only spatial run with `gate_attempts > 0` must produce a non‑empty `spatial.jsonl` (≥ 1 line).

## Expected outcome

- `python scripts/eval_cli.py ... mode=teach persist=true` (SMPD) will produce a **non‑empty** `spatial.jsonl` with at least a meta record (and typically a node).
- `python scripts/validate_store.py --algo smpd --kind spatial` passes.
- Subsequent `replay` and `test` steps continue to work, now that the precondition holds.
