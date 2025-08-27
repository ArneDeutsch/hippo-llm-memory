# Milestones 9 & 9.5 — Human Tasks Review and Updates
_Date: 2025‑08‑27_

This document reviews the **current human tasks** in `MILESTONE_9_PLAN.md` and `MILESTONE_9_5_PLAN.md` against the **ground truth** (`research/experiment-synthesis.md`) and the **evaluation harness** as implemented in the repository (`EVAL_PLAN.md`, `hippo_mem/eval/harness.py`, `configs/eval/*`). It lists the exact changes needed to ensure that executing the tasks will **actually validate** HEI‑NW, SGC‑RSS and SMPD, and gives revised, copy‑pasteable commands.

---

## 1) TL;DR verdict

- ✅ The overall structure is sound and most commands line up with the implemented harness.
- ⚠️ **Four blocking mismatches** would prevent clean validation if you run the plans as‑is:
  1. **Config path drift**: the plans mention `configs/presets/...`; the code uses `configs/eval/...`.
  2. **Baseline completeness**: the baseline grid (H2) omits `baselines/span_short`, which is required for **EM parity**.
  3. **Matrix semantics**: “memory grid” (H3) runs **all suites for each memory preset**; this wastes time and mixes intents. You should target **HEI‑NW→episodic**, **SGC‑RSS→semantic**, **SMPD→spatial** (plus episodic memory‑dependent variants for HEI‑NW).
  4. **Metric expectations**: `metrics.json` **does not contain a “delta” block** in the eval harness. Deltas are produced by `scripts/test_consolidation.py` (Milestone 9.5). Acceptance checks in M9 must use **pre/post fields** or compute deltas from `metrics.csv` flags instead of expecting a “Δ section”.

- ✅ All referenced scripts exist and are wired: `scripts/eval_model.py`, `scripts/eval_bench.py` (plumbing), `scripts/build_datasets.py`, `scripts/summarize_runs.py`, `scripts/replay_consolidate.py` (LoRA), `scripts/test_consolidation.py` (pre/post deltas). The Makefile has `datasets` and `smoke` targets.

---

## 2) What to change (minimal, no new code required)

1) **Fix preset paths in docs** to `configs/eval/...` everywhere.

2) **Baseline grid must include** `baselines/span_short` (for apples‑to‑apples EM). Keep `core`, `rag`, `longctx`.

3) **Memory grid must be per‑suite**, not all‑suites‑per‑preset:
   - **HEI‑NW** → run on `episodic`, **plus** the memory‑dependent variants: `episodic_multi`, `episodic_cross`, `episodic_capacity`.
   - **SGC‑RSS** → run on `semantic` only.
   - **SMPD** → run on `spatial` only.

4) **Acceptance criteria in M9**:
   - Replace “`metrics.json` has **Δ** section” with: “`metrics.json` has `pre_*` and `post_*` keys; compute Δ as `post_* − pre_*` (or use `flags` in `metrics.csv`).”
   - Keep refusal‑rate ≤ 0.5 and non‑zero retrieval/gating checks for memory runs (these are enforced by guardrails).

5) **Add a short “H0 Build datasets” step** using `make datasets`, so your size/seed matrix is guaranteed to exist before H2/H3.

6) **Minor**: In H1b acceptance, don’t require `system_prompt`/`max_new_tokens` in `meta.json` (they are not recorded). Check `model.chat_template_used` plus span‑format diagnostics instead.

---

## 3) Revised Human Tasks (copy‑paste)

### H0 — Build datasets (once)
```bash
make datasets
```

### H1 — Dev smoke after WP1–WP2
```bash
DATE=$(date +%Y%m%d_%H%M)
python scripts/eval_model.py suite=episodic preset=baselines/core n=50 seed=1337   model=Qwen/Qwen2.5-1.5B-Instruct outdir=runs/$DATE/baselines/core
```
**Verify:** `pred` is not echoing; `EM(raw)>0` or `F1≥0.20`; `audit_sample.jsonl` exists; `time_ms_per_100` is consistent with tokens/time.

### H1b — Span‑short sanity (EM parity)
```bash
DATE=$(date +%Y%m%d_%H%M)
python scripts/eval_model.py suite=episodic preset=baselines/span_short n=50 seed=1337   model=Qwen/Qwen2.5-1.5B-Instruct outdir=runs/$DATE/baselines/span_short
```
**Verify:** `meta.json.model.chat_template_used=true`; refusal‑rate ≤ 0.5; format‑violation and overlong ratios are small.

### H2 — Baseline grid (now includes span_short)
```bash
DATE=$(date +%Y%m%d_%H%M)
python scripts/eval_model.py +run_matrix=true date=$DATE   presets="[baselines/core,baselines/span_short,baselines/rag,baselines/longctx]"   tasks="[episodic,semantic,spatial]" n_values="[50,200,1000]"   seeds="[1337,2025,4242]"   model=Qwen/Qwen2.5-1.5B-Instruct outdir=runs/$DATE
```

### H3 — Memory grid (targeted per suite)
```bash
DATE=$(date +%Y%m%d_%H%M)

# HEI‑NW on episodic + memory‑dependent variants
python scripts/eval_model.py +run_matrix=true date=$DATE   presets="[memory/hei_nw]"   tasks="[episodic,episodic_multi,episodic_cross,episodic_capacity]"   n_values="[50,200,1000]" seeds="[1337,2025,4242]"   model=Qwen/Qwen2.5-1.5B-Instruct outdir=runs/$DATE

# SGC‑RSS on semantic only
python scripts/eval_model.py +run_matrix=true date=$DATE   presets="[memory/sgc_rss]" tasks="[semantic]"   n_values="[50,200,1000]" seeds="[1337,2025,4242]"   model=Qwen/Qwen2.5-1.5B-Instruct outdir=runs/$DATE

# SMPD on spatial only
python scripts/eval_model.py +run_matrix=true date=$DATE   presets="[memory/smpd]" tasks="[spatial]"   n_values="[50,200,1000]" seeds="[1337,2025,4242]"   model=Qwen/Qwen2.5-1.5B-Instruct outdir=runs/$DATE
```

### H4 — Summaries
```bash
python scripts/summarize_runs.py runs/$DATE --out runs/$DATE/summaries
```
**Verify:** Summary CSV/JSON exist; compute deltas from `metrics.json` `pre_*` vs `post_*` or via `metrics.csv` row flags.

---

## 4) Milestone 9.5 (cross‑session & consolidation) — unchanged, but with two clarifications

The commands already reference the correct tools:

- **Teach & persist** via `scripts/eval_model.py mode=teach persist=true store_dir=... session_id=...`.
- **LoRA training** via `scripts/replay_consolidate.py --config configs/consolidation/lora_small.yaml`.
- **Post test (memory OFF + Δ in `metrics.json`)** via `scripts/test_consolidation.py`.

**Clarifications to keep runs robust:**

- In **H2/H3/H4/H5** of 9.5, prefer the **same model** you used for M9 baselines to make deltas interpretable.
- For **H5**, use `--pre_dir` to point to the H4 pre-consolidation output so `test_consolidation.py` can compute `delta` and enforce the `+0.20` EM uplift gate on `episodic@50`.

**Example (H5) with explicit pre_dir:**
```bash
# Pre (memory OFF)
python scripts/test_consolidation.py --phase pre   --suite episodic --n 50 --seed 1337   --model Qwen/Qwen2.5-1.5B-Instruct   --outdir runs/$DATE/consolidation/pre

# Post (LoRA merged on the fly; requires adapter path from training)
python scripts/test_consolidation.py --phase post   --suite episodic --n 50 --seed 1337   --model Qwen/Qwen2.5-1.5B-Instruct   --adapter runs/$DATE/consolidation/lora   --pre_dir runs/$DATE/consolidation/pre   --outdir runs/$DATE/consolidation/post
```

---

## 5) Acceptance checklist (updated)

- **Harness correctness** (M9): no echo; chat template ON where expected; `metrics.json.metrics.<suite>` contains `pre_*`/`post_*` keys for memory runs; `metrics.csv` rows flagged with `pre_replay`/`post_replay`.
- **Baselines complete**: `core`, `span_short`, `rag`, `longctx` across episodic/semantic/spatial, `n ∈ {50,200,1000}`, `seeds ∈ {1337,2025,4242}`.
- **Memory runs complete**: targeted per suite; episodic variants included for HEI‑NW.
- **Guardrails**: refusal ≤ 0.5 on span suites; retrieval/gating non‑zero in memory runs (guardrails will fail otherwise).
- **Artifacts**: `meta.json`, `metrics.json`, `metrics.csv`, `audit_sample.jsonl` under `runs/<date>/<preset>/<suite>/...`; summaries under `runs/<date>/summaries/`.
- **9.5 consolidation**: `test_consolidation.py` writes `delta` block; **EM uplift ≥ +0.20** on `episodic@50` (seed=1337).

---

## 6) Codex tasks to update the milestone docs

> Copy‑paste into Codex. These modify only the two milestone docs.

### Task C‑M9‑HUMAN‑REFRESH
**Files:** `MILESTONE_9_PLAN.md`  
**Change:** Fix config paths to `configs/eval/...`; insert **H0 datasets**; update **H2** to include `baselines/span_short`; make **H3** targeted per suite and include episodic variants; adjust acceptance to use `pre_*`/`post_*` fields (no Δ block). Replace all `task=` with `suite=` for consistency.

**Acceptance:** Running the new H2/H3 commands on a small seed/size succeeds and writes outputs under `runs/<DATE>/...` matching `scripts/summarize_runs.py` expectations.

### Task C‑M9_5‑HUMAN‑CLARIFY
**Files:** `MILESTONE_9_5_PLAN.md`  
**Change:** In H5, add explicit `--pre_dir` in the Post example and a note that deltas come from `scripts/test_consolidation.py`. Ensure earlier text never claims eval harness writes a dedicated “Δ section”.

**Acceptance:** `scripts/test_consolidation.py --phase post ...` produces a `metrics.json` containing a `delta` object and fails if EM uplift < +0.20 on `episodic@50` seed=1337.

---

## 7) Why this is enough

These edits align the milestone runbooks with the **actual harness behavior** and the **evaluation plan** so that executing the steps produces **auditable** evidence of: (a) baseline parity on span decoding, (b) targeted effects of each algorithm on its intended suite, and (c) systems consolidation gains with memory disabled post‑LoRA.

Short version: treat **DATE as a “cohort” identifier**. For Milestone 9 you should use **one single DATE for the entire batch (H0→H4)** so baselines and memory runs land under the **same** `runs/<DATE>/…` tree and your summaries/reports cover them together.

## 8) How to handle DATE

Why this is the *intended* way in this repo:

* The harness writes outputs under `runs/<date>/<preset>/<suite>/<size>_<seed>/…` (see `hippo_mem/eval/harness.py`). `scripts/report.py` and `scripts/summarize_runs.py` both **scan a single `runs/<date>` root** to build cohort-level summaries and the top-level report in `reports/<date>/…`. If baselines and memory runs use different dates, the default report won’t show them side-by-side.
* The EVAL plan validates each algorithm *relative to baselines* (HEI-NW→episodic, SGC-RSS→semantic, SMPD→spatial). That comparison is clearest when the runs are under the **same date** so the report scripts collect both sets in one pass.
* M9’s acceptance gates include baseline completeness **and** memory-preset telemetry; they’re designed to be checked together.

So:

* **Milestone 9 (baselines + memory grid):** pick `DATE=…` **once** at the start and reuse it for all steps H0–H4. Then run `scripts/summarize_runs.py runs/$DATE …` or `scripts/report.py --date $DATE` to get a single, coherent report.

* **Milestone 9.5 (consolidation):** you have two reasonable options:

  * **Same DATE** if it’s the *continuation* of the same experiment (makes `--pre_dir runs/$DATE/consolidation/pre` trivial and puts the 9.5 report alongside the M9 results under `reports/<DATE>`).
  * **New DATE** only if you intentionally start a *new* experiment (different model/config/data). In that case keep **pre/post of 9.5 under the same new DATE**, and pass `--pre_dir` accordingly.

* **When would “memory grid alone” with its own DATE be OK?** Only if you explicitly want to *isolate* a memory-preset tweak and **you are not producing a combined report**. If you still need baseline comparison in the same report, re-run (or copy in) the baselines under that same DATE.

Rule of thumb: **one DATE per auditable experiment** (fixed commit, model list, dataset build, seed set). If any of those change, start a new DATE; otherwise keep them together.
