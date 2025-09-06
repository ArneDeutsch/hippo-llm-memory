## Task 1 — (Non-breaking) Add preset **aliases** for legacy names

**Why:** CI and docs currently reference `memory/hei_nw_cross` and `memory/sgc_rss_mem`. Aliases keep those names working even if we also update CI.

**Files**

* `configs/eval/memory/hei_nw.yaml` (canonical episodic preset)
* `configs/eval/memory/sgc_rss.yaml` (canonical relational preset)
* **Create:**

  * `configs/eval/memory/hei_nw_cross.yaml`
  * `configs/eval/memory/sgc_rss_mem.yaml`

**Do**

1. **Copy** the canonical YAMLs to the alias filenames, then adjust only comments/title lines if present.

```bash
cp configs/eval/memory/hei_nw.yaml configs/eval/memory/hei_nw_cross.yaml
cp configs/eval/memory/sgc_rss.yaml configs/eval/memory/sgc_rss_mem.yaml
```

2. If the canonical files have names or descriptions inside, update those strings to mention “Alias of …” (no behavioral changes).

**Accept**

* Both new YAMLs exist and are byte-for-byte copies (modulo comments) of their canonical presets.
* `hydra` can load both `memory/hei_nw_cross` and `memory/sgc_rss_mem` without error:

```bash
python - <<'PY'
from omegaconf import OmegaConf
from pathlib import Path
for p in ["configs/eval/memory/hei_nw_cross.yaml","configs/eval/memory/sgc_rss_mem.yaml"]:
    assert Path(p).exists(), p
print("OK")
PY
```

---

## Task 2 — Preflight: resolve store files by **kind** (alias-aware)

**Why:** Aliases like `_cross`/`_mem` currently make preflight look for non-existent `*.jsonl` files (e.g., `hei_nw_cross.jsonl`). Stores persist as `episodic.jsonl`, `kg.jsonl`, `spatial.jsonl`.

**File**

* `hippo_eval/eval/harness.py` (function `preflight_check`)

**Do**

1. Ensure `re` is imported (add if missing near other imports).
2. Replace the filename mapping with alias-aware logic.

**Patch (apply verbatim; adjust line numbers if needed)**

```diff
*** a/hippo_eval/eval/harness.py
--- b/hippo_eval/eval/harness.py
@@
-import json
+import json
+import re
@@ def preflight_check(cfg, outdir: Path) -> None:
-        store_files = {
-            "hei_nw": "episodic.jsonl",
-            "sgc_rss": "kg.jsonl",
-            "smpd": "spatial.jsonl",
-        }
-        store_file = meta_path.parent / store_files.get(algo, f"{algo}.jsonl")
+        # Normalize algorithm aliases and resolve store kind to canonical filenames
+        base = re.sub(r"(_mem|_cross)$", "", algo)
+        kind = {"sgc_rss": "kg", "smpd": "spatial"}.get(base, "episodic")
+        store_file = meta_path.parent / f"{kind}.jsonl"
         if not store_file.exists() or store_file.stat().st_size == 0:
             fail("missing_or_empty_store_file", {
                 "expected_file": str(store_file),
                 "algo": algo,
                 "session_id": session_id,
                 "store_dir": str(store_dir),
             })
```

**Accept**

* `python -m pyflakes hippo_eval/eval/harness.py` shows no new import errors.
* Running a local teach→test on any of the three memory presets expects `episodic.jsonl`/`kg.jsonl`/`spatial.jsonl` (not `<algo>.jsonl`).

---

## Task 3 — Gates: count **attempts** on null-input paths

**Why:** The preflight dry-run for `n=1` may hit null-input paths; not counting those as attempts produces false negatives.

**Targets (search):** look under `hippo_mem/` for gate classes.

* `hippo_mem/**/episodic*_gate*.py`
* `hippo_mem/**/relational*_gate*.py` or `kg*_gate*.py`
* `hippo_mem/**/spatial*_gate*.py`

**Do**

1. Locate `decide(...)` or equivalent methods that update gate stats (often via `stats.attempts` / `stats.null_input`).
2. In the branch handling empty inputs (e.g., `if not inputs:` or `if not memory_ctx:`), **increment `attempts` before returning**, e.g.:

```python
stats.attempts += 1
stats.null_input += 1
return decision
```

3. Ensure analogous changes in episodic, relational (kg), and spatial gates.

**Codemod helper (optional)**
If the pattern is consistent, run this safe replacement and review diffs:

```bash
rg -n "def decide\\(" -g 'hippo_mem/**/*.py'
# Open each match; for branches like `if not .*:` that return early,
# add `stats.attempts += 1` before the return and ensure `stats` exists.
```

**Accept**

* Grep shows at least one `stats.attempts += 1` in each of the three gate families.
* Dry-run preflight no longer fails with `gate.attempts == 0` on the smoke matrix.

---

## Task 4 — Align **smoke script** to canonical presets and scope failure grep

**Why:** Names drifted; also the final grep is redundant and can catch transient files.

**File**

* `scripts/ci_smoke_eval.sh`

**Do**

1. Replace legacy preset names, keep suites the same.
2. Remove or scope the final `failed_preflight.json` grep.

**Patch (edit the env and grep parts; adapt to your file content)**

```diff
@@
- PRESET="memory/hei_nw_cross"
+ PRESET="memory/hei_nw"
@@
- PRESET="memory/sgc_rss_mem"
+ PRESET="memory/sgc_rss"
@@
- # Hard fail if any failed_preflight.json exists under the run root
- if find "runs/$RUN_ID" -name failed_preflight.json -print -quit | grep -q .; then
-   echo "Preflight failed"
-   exit 1
- fi
+ # (Optional) If you keep this check, scope it to the final test output directory only
+ # find "runs/$RUN_ID/<suite>/*/failed_preflight.json" -print -quit | grep -q . && exit 1
```

**Accept**

* Script runs without referencing `*_cross`/`*_mem` presets.
* If the grep is kept, it is scoped to a specific suite output folder.

---

## Task 5 — (Optional, recommended) Align CI workflow preset names

**Why:** Reduces reliance on aliases; keeps config tidy.

**File (typical):** `.github/workflows/ci.yaml` (or similar)

**Do**

* Replace `memory/hei_nw_cross` → `memory/hei_nw`
* Replace `memory/sgc_rss_mem` → `memory/sgc_rss`

**Patch (illustrative; adjust job names/keys to your file)**

```diff
- PRESET: memory/hei_nw_cross
+ PRESET: memory/hei_nw
@@
- PRESET: memory/sgc_rss_mem
+ PRESET: memory/sgc_rss
```

**Accept**

* Workflow compiles (pass GitHub Actions syntax check if available).
* Jobs show canonical presets in their logs.

---

## Task 6 — (Optional) Add a focused preflight sanity script

**Why:** Quick local reproduction and guardrail in CI.

**File (new):** `scripts/preflight_sanity_check.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
RUN_ID="sanity_${RANDOM}"
STORES="runs/${RUN_ID}/stores"
MODEL="models/tiny-gpt2"
SIZES=(50)
SEEDS=(1337)

suites=("episodic_cross_mem" "semantic_mem" "spatial_multi")
presets=("memory/hei_nw" "memory/sgc_rss" "memory/smpd")

for i in "${!suites[@]}"; do
  suite="${suites[$i]}"
  preset="${presets[$i]}"
  out="runs/${RUN_ID}/${preset##*/}/${suite}"
  python scripts/eval_model.py suite="$suite" preset="$preset" run_id="$RUN_ID" n="${SIZES[0]}" seed="${SEEDS[0]}" \
    mode=teach persist=true store_dir="$STORES" session_id="san_${RUN_ID}" compute.pre_metrics=true strict_telemetry=true model="$MODEL"
  test -s "${out}/failed_preflight.json" && { echo "Unexpected preflight fail"; exit 1; } || true
done
echo "Preflight sanity OK"
```

Make it executable:

```bash
chmod +x scripts/preflight_sanity_check.sh
```

**Accept**

* Running the script completes with “Preflight sanity OK”.

---

## Task 7 — Run the smoke locally

**Do**

```bash
bash scripts/ci_smoke_eval.sh
```

**Accept**

* No `failed_preflight.json` is produced.
* Stores exist and are non-empty: `episodic.jsonl`, `kg.jsonl`, `spatial.jsonl` in the expected session folders.
* CI matrix completes locally on tiny model `models/tiny-gpt2`.
