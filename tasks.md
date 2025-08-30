# P0 — Must Fix

## T1 — EVAL\_PROTOCOL.md §8: fix `$SESS` typo and make §8 self-contained

**Why:** §8 currently uses `$SESS` (unset) and doesn’t re-establish env vars, which led to an empty `--model` and a store path check against `""`.

**Edits (docs):**

* In **EVAL\_PROTOCOL.md**, §8 prelude must define env (mirror §0) and consistently use `$SID`.
* Replace any `$SESS` with `$SID`.
* Ensure the guard checks reference `"$STORES/hei_nw/$SID/episodic.jsonl"`.

**Replace §8 prelude block with:**

```bash
# §8 prelude (self-contained)
: "${RUN_ID:=${DATE:-$(date -u +%Y%m%d_%H%M)}}"
DATE="$RUN_ID"
RUNS="runs/$RUN_ID"
REPORTS="reports/$RUN_ID"
STORES="$RUNS/stores"
ADAPTERS="adapters/$RUN_ID"
: "${MODEL:=Qwen/Qwen2.5-1.5B-Instruct}"  # default for production runs

mkdir -p "$RUNS" "$REPORTS" "$STORES" "$ADAPTERS"

SID="hei_${RUN_ID}"

# Require a persisted HEI-NW store
test -f "$STORES/hei_nw/$SID/episodic.jsonl" || {
  echo "No persisted HEI-NW store for $SID. Run §4.1 (teach+replay with persist=true) first."
  exit 1
}
```

**Acceptance criteria**

* No mention of `$SESS` remains in EVAL\_PROTOCOL.md.
* Running §8 in a fresh shell (only `RUN_ID` exported) works and fails fast with a clear message if the store is missing.

---

## T2 — Arg validation: forbid empty `--model` outside tests

**Why:** An empty `--model` currently resolves to the repo root, causing HF to try to load the CWD as a model.

**Files:**

* `scripts/test_consolidation.py`
* `hippo_mem/eval/harness.py` (or centralize in a helper)

**Changes (minimal + safe):**

1. In `scripts/test_consolidation.py` after parsing args:

```python
import os, sys

def _resolve_model(arg_model: str) -> str:
    m = (arg_model or os.environ.get("MODEL") or "").strip()
    if not m:
        raise SystemExit(
            "Error: --model is empty and $MODEL is not set.\n"
            "Set --model (e.g., Qwen/Qwen2.5-1.5B-Instruct) or export MODEL."
        )
    return m

args.model = _resolve_model(getattr(args, "model", None))
```

2. In `hippo_mem/eval/harness.py` before `AutoTokenizer.from_pretrained(...)`:

```python
model_id = (str(cfg.model) or "").strip()
if not model_id:
    raise ValueError("cfg.model is empty. Pass --model or set $MODEL.")

# Additional sanity: if model_id resolves to an existing dir, ensure it has config.json
from pathlib import Path
p = Path(model_id)
if p.exists() and p.is_dir():
    if not (p / "config.json").exists():
        raise ValueError(
            f"Model path '{p}' exists but is not a Hugging Face model dir (missing config.json). "
            "Did you accidentally pass the repository root? Set --model correctly."
        )
```

**Acceptance criteria**

* Invoking any affected script with `--model ""` or without `$MODEL` exits with a clear, actionable error.
* Invoking with a directory lacking `config.json` raises the improved explanation.

---

## T3 — Store existence checks: centralize and use consistently

**Why:** Avoid duplicated brittle `test -f ...` logic and inconsistent variable use.

**Files:**

* New: `hippo_mem/utils/stores.py`
* Update imports in: `scripts/replay_consolidate.py`, `scripts/eval_model.py` (replay/test modes), `scripts/test_consolidation.py`

**Add helper:**

```python
# hippo_mem/utils/stores.py
from pathlib import Path

def assert_store_exists(store_dir: str, session_id: str, kind: str = "episodic") -> Path:
    p = Path(store_dir) / "hei_nw" / session_id / f"{kind}.jsonl"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing persisted store: {p}\n"
            "Run §4.1 (teach + replay with persist=true) to create it."
        )
    return p
```

**Use it** in the scripts right before replay/consolidation phases. Keep the Bash guard in the protocol, but rely on Python checks as the source of truth.

**Acceptance criteria**

* Running replay/consolidation without a store fails immediately with a precise `FileNotFoundError` pointing to the expected path.

---

# P1 — Quality & DX

## T4 — Introduce shared shell prelude to DRY env setup

**Why:** Multiple protocol sections redefine the same env block; single source avoids drift.

**Files:**

* New: `scripts/env_prelude.sh`
* Docs: EVAL\_PROTOCOL.md references `source scripts/env_prelude.sh`

**Content:**

```bash
# scripts/env_prelude.sh
: "${RUN_ID:=${DATE:-$(date -u +%Y%m%d_%H%M)}}"
DATE="$RUN_ID"
RUNS="runs/$RUN_ID"
REPORTS="reports/$RUN_ID"
STORES="$RUNS/stores"
ADAPTERS="adapters/$RUN_ID"
: "${MODEL:=Qwen/Qwen2.5-1.5B-Instruct}"
mkdir -p "$RUNS" "$REPORTS" "$STORES" "$ADAPTERS"
```

**Docs change (top of each section that needs it):**

```bash
source scripts/env_prelude.sh
```

**Acceptance criteria**

* All protocol sections use the shared prelude; no duplicate divergent copies remain.

---

## T5 — Make tiny-gpt2 strictly opt-in for tests

**Why:** We only want tiny-gpt2 for local CI/smoke tests, never as a silent fallback.

**Files:**

* `scripts/test_consolidation.py` (and any other script that had a hidden fallback)
* `tests/` (new smoke test)

**Changes:**

* Remove any fallback to tiny-gpt2 in runtime code.
* Add a **test-only** flag in CLI if desired:

  * `--allow-tiny-test-model` that, when set, allows `--model tiny-gpt2` or sets it automatically in test scripts.
* Create a tiny smoke test in `tests/test_smoke_consolidation.py` guarded by `pytest.mark.slow or env` to run with `MODEL=tiny-gpt2`.

**Acceptance criteria**

* Production runs error if `--model` is unset.
* smoke that can run offline with tiny-gpt2 when explicitly enabled

---

## T6 — Protocol enhancements for quick runs (n=50, seed=1337)

**Why:** Many users trial-run with SIZES=(50) SEEDS=(1337). Protocol should call this out and show the expected persisted path.

**Docs edits (EVAL\_PROTOCOL.md):**

* Add a note in §4.1 and §8:

  * “When running with `SIZES=(50)` and `SEEDS=(1337)`, the expected store is `runs/$RUN_ID/stores/hei_nw/hei_$RUN_ID/episodic.jsonl`.”
* Add a one-liner check:

  ```bash
  ls -l "runs/$RUN_ID/stores/hei_nw/hei_${RUN_ID}/episodic.jsonl"
  ```

**Acceptance criteria**

* New note appears and accurately reflects the path users see.

---

## T7 — CLI consistency: validate `--store_dir` and `--session_id`

**Why:** Provide uniform, helpful error messages across all scripts that interact with stores.

**Files:**

* `scripts/replay_consolidate.py`
* `scripts/eval_model.py` (replay/test modes)

**Changes:**

* Early validation:

  ```python
  if not args.store_dir or not args.session_id:
      raise SystemExit("Error: --store_dir and --session_id are required for this mode.")
  from hippo_mem.utils.stores import assert_store_exists
  assert_store_exists(args.store_dir, args.session_id, kind="episodic")
  ```

**Acceptance criteria**

* Missing parameters produce uniform CLI errors before any heavy work starts.

---

# P2 — Nice to Have

## T8 — Makefile helper to verify stores for current RUN\_ID

**Why:** Fast sanity check before §8.

**Files:**

* `Makefile`

**Add target:**

```make
check-stores:
\t@./scripts/ls_store.sh
```

**New script `scripts/ls_store.sh`:**

```bash
#!/usr/bin/env bash
source scripts/env_prelude.sh
SID="hei_${RUN_ID}"
P="$STORES/hei_nw/$SID/episodic.jsonl"
if [[ -f "$P" ]]; then
  echo "OK: $P"
else
  echo "MISSING: $P" >&2
  exit 2
fi
```

**Acceptance criteria**

* `make check-stores` prints `OK: …` for a prepared run, otherwise exits 2 with `MISSING: …`.

---

# Search & Cleanups

## T9 — Purge `$SESS` usage across repo

**Why:** Prevent recurrence of the typo.

**Action for Codex:**

* Run a search and replace (review diffs):

  * Search: `\bSESS\b`
  * Replace with: `SID`
* Verify only shell/docs references are updated (do not touch unrelated code identifiers).

**Acceptance criteria**

* No `$SESS` remains in the repository.

---

# Verification Plan (post-merge)

1. Fresh shell: `unset MODEL; export RUN_ID=20250829_1307`
2. `source scripts/env_prelude.sh` (MODEL defaults to Qwen)
3. `make check-stores` — should report OK if §4.1 was run with n=50/seed=1337.
4. Run §8 commands exactly as documented:

   * `python scripts/test_consolidation.py --phase pre --suite episodic --n 50 --seed 1337 --model "$MODEL" --outdir "$RUNS/consolidation/pre"`
   * `python scripts/replay_consolidate.py --store_dir "$STORES/hei_nw" --session_id "hei_${RUN_ID}" --model "$MODEL" --config configs/consolidation/lora_small.yaml --outdir "$RUNS/consolidation/lora"`
   * `python scripts/test_consolidation.py --phase post --suite episodic --n 50 --seed 1337 --model "$MODEL" --adapter "$RUNS/consolidation/lora" --pre_dir "$RUNS/consolidation/pre" --outdir "$RUNS/consolidation/post"`
5. Negative test: `unset MODEL` and omit `--model` → scripts must exit with clear “model empty” message.
