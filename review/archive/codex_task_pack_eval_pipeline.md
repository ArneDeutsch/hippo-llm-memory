# Codex Task Pack — Strengthen Evaluation Plan & Pipeline

This pack contains **copy‑paste prompts** for Codex. Each task follows the repository’s issue template (Goal / Files to touch / Acceptance tests / Run / Context) and references **actual files and modules present in the ZIP**. Apply tasks in order.

---

## T1 — Enhance `EVAL_PLAN.md` with answer‑format policy, normalization rules, and memory‑dependent suites

**Goal:**  
Clarify how we score answers (strict EM vs. normalized EM vs. F1), enforce a *short‑span output policy*, and add memory‑dependent episodic variants so the plan matches the current harness and upcoming code changes.

**Files to touch:**  
- `EVAL_PLAN.md`

**Edits (verbatim insertions):**

1) **After the first `# 5) Metrics` section, insert a new subsection titled `## 5.4 Answer format & normalization policy` with this content:**

> ### 5.4 Answer format & normalization policy  
> All suites using span extraction MUST follow a short‑answer policy.  
> **Model instruction:** “Answer with the exact shortest span; no punctuation; no extra words.”  
> **Metrics:** We report three scores side‑by‑side:  
> - **EM (raw):** `pred.strip() == gold` (exact string match).  
> - **EM (normalized):** lower‑case, strip punctuation and articles (`a|an|the`) from both sides before comparison. Normalizer defined in `hippo_mem/eval/score.py`.  
> - **Token‑F1:** whitespace token F1.  
> Diagnostics we also log: `pred_len`, `gold_len`, `overlong` (pred_len > gold_len), and `format_violation` (any terminal punctuation or contains a period).

2) **Under `# 3) Suites & generators`, append a new subsection `### 3.1 Episodic variants (memory‑dependent)` with this content:**

> We add three episodic variants to force memory usage beyond trivial one‑shot extraction:  
> - **`episodic_multi`** — multi‑turn episodes with distractors and last‑mention‑wins corrections.  
> - **`episodic_cross`** — cross‑episode recall after session flush; facts only available via store replay.  
> - **`episodic_capacity`** — episodes longer than the decoding context budget; retrieval required.  
> Generators live in `hippo_mem/eval/datasets.py` and are addressable via the CLI (`scripts/build_datasets.py`).

3) **Under `# 4) Run matrix`, add bullets to include the three episodic variants for both `baselines/span_short` and `memory/*` presets and explicitly list `n ∈ {50, 200, 1000}` and `seeds ∈ {1337, 2025, 4242}`.**

4) **Under `# 9) Commands (examples)`, append examples using `baselines/span_short` and `memory/hei_nw` that show identical decoding settings (`use_chat_template: true`, `max_new_tokens: 8`).**

5) **Under `# 11) Success criteria (v0 targets)`, add:** “Baselines **must not be saturated**: target EM(raw) < 60% on H2‑like episodic; we expect **+8–12pp EM(norm)** from memory on `episodic_multi` at n=200.”

**Acceptance tests:**  
- Plain text check: the new subsections and bullets are present with the exact headings above.  
- References to modules (`hippo_mem/eval/score.py`, `hippo_mem/eval/datasets.py`) exist after completing T2–T5.

**Run:** `make test` (doc‑only, no failures expected).

**Context:** `EVAL_PLAN.md`, `configs/eval/baselines/span_short.yaml` already uses the short‑span decoding profile; we harmonize memory presets next.

---

## T2 — Align memory presets with short‑span decoding (apples‑to‑apples)

**Goal:**  
Ensure memory presets use the same decoding constraints as `baselines/span_short.yaml` so EM is comparable.

**Files to touch:**  
- `configs/eval/memory/hei_nw.yaml`  
- `configs/eval/memory/sgc_rss.yaml`  
- `configs/eval/memory/smpd.yaml`

**Changes:**  
Add the following fields to each of the three memory presets (values exactly as in `baselines/span_short.yaml`):

```yaml
use_chat_template: true
system_prompt: "Answer with the exact shortest span from the prompt. No explanations."
max_new_tokens: 8
```

**Acceptance tests:**  
- Add `tests/test_memory_presets_span.py` asserting the three files contain these fields and that baseline toggles (retrieval enabled as appropriate; gating toggled by each memory preset) remain unchanged.  
- Existing `tests/test_baseline_presets.py` must still pass.

**Run:** `make test`.

**Context:** `configs/eval/baselines/span_short.yaml` already defines the desired decoding settings.

---

## T3 — Centralize scoring & add normalized EM

**Goal:**  
Create a scoring helper exposing `em_raw`, `em_norm`, and `f1`, and integrate it into the harness. Emit both EMs in `metrics.json` and `metrics.csv`.

**Files to touch:**  
- `hippo_mem/eval/score.py` (new)  
- `hippo_mem/eval/harness.py` (modify)  
- `scripts/report.py` (augment to prefer `em_norm` if present)

**Implementation details:**

1) **New module** `hippo_mem/eval/score.py` with:
```python
import re, string
from collections import Counter

_ARTICLES = {"a","an","the"}
_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")

def normalize(s: str) -> str:
    s = s.strip().lower()
    s = _PUNCT_RE.sub("", s)
    toks = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(toks)

def em_raw(pred: str, gold: str) -> int:
    return int(pred.strip() == gold)

def em_norm(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))

def f1(pred: str, gold: str) -> float:
    pt, gt = pred.split(), gold.split()
    if not pt or not gt: return 0.0
    common = Counter(pt) & Counter(gt)
    overlap = sum(common.values())
    if overlap == 0: return 0.0
    precision = overlap / len(pt)
    recall = overlap / len(gt)
    return 2 * precision * recall / (precision + recall)
```

2) **Wire into** `hippo_mem/eval/harness.py` in the evaluation loop: compute `em_raw`, `em_norm`, `f1`, plus diagnostics: `pred_len`, `gold_len`, `overlong`, `format_violation` (regex: period at end or contains newline). Add these per‑row to `metrics.csv` and aggregate in `metrics.json` under `metrics[<suite>]` as `em_raw`, `em_norm`, `f1`, and counts for `overlong` and `format_violation`.

3) **Update `scripts/report.py`**: if `em_norm` exists, display it as “EM (norm)” and keep EM(raw) in a separate column; otherwise fall back to legacy `em`.

**Acceptance tests:**  
- New `tests/test_scoring_normalization.py` validates `normalize`, `em_norm`, and that `harness.evaluate` writes both EMs into `metrics.json`.  
- Existing tests still pass.

**Run:** `make test`.

**Context:** Harness currently uses `pred.strip() == answer` and a local `_f1`; we migrate to the new helper without changing legacy behavior for raw EM.

---

## T4 — Add near‑miss & format diagnostics to CSV

**Goal:**  
Make it easy to see why “looks correct” answers fail EM(raw).

**Files to touch:**  
- `hippo_mem/eval/harness.py`

**Changes:**  
Per row in `metrics.csv` add columns: `pred_len`, `gold_len`, `overlong` (bool), `format_violation` (bool). Aggregate counts in `metrics.json` as `diagnostics` under the suite.

**Acceptance tests:**  
- Extend `tests/test_report_shapes.py` (new) to assert these columns are present for an episodic run.  
- `metrics.csv` schema retains previous columns as‑is.

**Run:** `make test`.

**Context:** The uploaded H2 sample showed frequent punctuation and article mismatches.

---

## T5 — Implement episodic memory‑dependent generators and suite names

**Goal:**  
Add three generators and wire them into CLI so we can create datasets that demand memory usage.

**Files to touch:**  
- `hippo_mem/eval/datasets.py` (add functions and mapping)  
- `scripts/build_datasets.py` (CLI already routes to `datasets.py`)  
- `Makefile` (extend `datasets` target)  
- `EVAL_PLAN.md` (already updated in T1)

**Implementation details:**  
- Add functions: `generate_episodic_multi(size:int, seed:int, distractors:int=8, corrections:bool=True)`, `generate_episodic_cross(size:int, seed:int)`, `generate_episodic_capacity(size:int, seed:int, context_budget:int=256)` that produce the same JSONL shape (`prompt`, `answer`, plus any extra metadata).  
- Extend `SUITE_TO_GENERATOR` to include keys: `"episodic_multi"`, `"episodic_cross"`, `"episodic_capacity"`.  
- Add CLI `--suite` support for the three names.  
- In `Makefile` `datasets` target, add those suite names to the loop.

**Acceptance tests:**  
- `tests/test_datasets_variants.py` (new) calls the three generators and asserts deterministic length and that prompts contain distractors/flush markers (for cross‑episode).  
- `make datasets` completes without errors.

**Run:** `make test` and `make datasets`.

**Context:** The current `generate_episodic` creates one‑shot items; these variants produce scenarios where retrieval matters.

---

## T6 — Ensure memory runs use identical decoding and retrieval is exercised

**Goal:**  
Harmonize decoding across presets and keep the CI guardrail that requires retrieval on memory runs.

**Files to touch:**  
- (No new code beyond T2.)  
- Optionally expand `tests/test_ci_guardrails.py` with a case that a memory preset lacking retrieval usage raises the existing `RuntimeError` (“retrieval.requests == 0 for memory run”).

**Acceptance tests:**  
- `tests/test_ci_guardrails.py` new case passes.  
- All prior tests pass.

**Run:** `make test`.

**Context:** Guardrail already exists in `hippo_mem/eval/harness._enforce_guardrails`.

---

## T7 — Update `scripts/run_baselines_bench.py` matrix to include new episodic variants

**Goal:**  
Allow one command to generate baseline and memory results for the new episodic variants.

**Files to touch:**  
- `hippo_mem/eval/baselines.py` (constant lists)  
- `scripts/run_baselines_bench.py` (reads those constants)

**Changes:**  
- In `hippo_mem/eval/baselines.py`, extend `SUITES` to:  
  `["episodic","semantic","spatial","episodic_multi","episodic_cross","episodic_capacity"]`.

**Acceptance tests:**  
- `tests/test_run_baselines_bench.py` updated/added to assert the matrix covers the six suites.
- Running `python scripts/run_baselines_bench.py --date 20990101 --presets baselines/span_short` prints commands for the new suites.

**Run:** `make test`.

**Context:** Keeps the runner in sync with the expanded plan.

---

## T8 — Reporting: show EM(raw) vs EM(norm) side‑by‑side

**Goal:**  
Make reports reflect normalized scoring and aid diagnosis.

**Files to touch:**  
- `scripts/report.py`

**Changes:**  
- When building the Markdown tables, include both EM(raw) and EM(norm) if present; add diagnostic columns for `overlong` and `format_violation` ratios.

**Acceptance tests:**  
- `tests/test_report_output.py` (new) runs the collector on a synthetic run directory (tiny JSON fixtures) and asserts both EMs appear in the generated Markdown text.

**Run:** `make test`.

**Context:** Aligns reporting with T3/T4.

---

## T9 — Examples in `EVAL_PLAN.md` and smoke script

**Goal:**  
Provide copy‑paste commands that reflect the new setup.

**Files to touch:**  
- `EVAL_PLAN.md` (commands already updated in T1)  
- `scripts/smoke_eval.sh` (append a line using `baselines/span_short` and one using `memory/hei_nw`)

**Acceptance tests:**  
- `bash scripts/smoke_eval.sh` executes without error in a cold environment (tokenizer/model download may be mocked in CI).

**Run:** `make test` (script presence only).

**Context:** Ensures practitioners run apples‑to‑apples configs.

---

## T10 — (Optional) Config gate to select EM(raw) vs EM(norm) as “primary”

**Goal:**  
Make it explicit which EM we consider primary for gates/regressions.

**Files to touch:**  
- `configs/eval/default.yaml` (add `primary_em: "norm"` default)  
- `hippo_mem/eval/harness.py` (if primary is `"norm"`, write `metrics[<suite>]["em"] = em_norm` for backward compat in plots)

**Acceptance tests:**  
- `tests/test_primary_em_switch.py` sets primary to `"raw"` and `"norm"` via Hydra overrides and asserts the chosen value flows to `metrics.json` as `em` while both detailed keys remain present.

**Run:** `make test`.

**Context:** Non‑breaking and keeps downstream tooling simple.

---

### Notes

- Keep *all existing behavior* intact unless explicitly changed.  
- Use small, well‑named helpers; follow `CODING_STANDARDS.md`.  
- Add unit tests next to existing ones; do not modify unrelated tests.