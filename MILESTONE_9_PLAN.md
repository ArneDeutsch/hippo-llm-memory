# Milestone 9 — Baselines, Memory Validation & Data Generation
_Generated: 2025-08-25 08:34_

## 0) Goal (what we ship)
Establish reliable **baselines** and **memory-enabled** evaluations for the three hippocampus-inspired algorithms, producing final, reproducible datasets & metrics. The plan fixes known issues (prompt echo, incorrect timing) and ensures **chat-template correctness**, **expanded run matrix**, and **clear pre/post** deltas for memory runs.

**Algorithms under test (ground truth from `research/experiment-synthesis.md`):**
- **HEI‑NW** (Hippocampal Episodic Indexing — Neural Weighting)
- **SGC‑RSS** (Sparse Gated Consolidation — Rehearsal/Replay)
- **SMPD** (Semantic Memory Pattern Distillation)

## 1) Gate (exit criteria)
We declare Milestone 9 *done* only if **all** the following are true:

- **Harness correctness**
  - No prompt echo: decoded `pred` is **only the generated continuation** (input is sliced off).
  - **Chat templates** are applied for chat-tuned models (e.g., Qwen‑Instruct) and disabled for base models.
  - `metrics.json.compute.time_ms_per_100` is correctly computed as `100 * total_time_ms / total_tokens`.
  - `metrics.json.compute` includes `input_tokens` and `generated_tokens`.

- **Baselines complete**
  - Presets: `baselines/core`, `baselines/rag`, `baselines/longctx` executed across tasks `episodic`, `semantic`, `spatial` with `n ∈ {50,200,1000}` and seeds `{1337,2025,4242}`.
  - Episodic@50 dev-smoke run (seed=1337) achieves **EM > 0** or **F1 ≥ 0.20** (sanity threshold).

- **Memory runs complete**
  - Presets: `memory/hei_nw`, `memory/sgc_rss`, `memory/smpd` executed with `replay.cycles ∈ {1,3}`, `gating_enabled=true`, retrieval on.
  - `metrics.json` contains **pre** and **post** metrics and a populated **Δ** (delta) section.
  - Retrieval (`requests`, `hits`, `r@k`) and gating stats are **non‑zero** for memory runs.

- **Docs & reproducibility**
  - `README.md`, `MILESTONE_9_PLAN.md`, `PROJECT_PLAN.md`, `EVAL_PLAN.md` updated to reflect model selection, presets, run matrix, and acceptance checks.
  - CI/Smoke: a script or target runs `episodic@50` seed=1337 and fails fast if the sanity threshold is not met.

- **Artifacts**
  - For each run: `meta.json`, `metrics.json`, `metrics.csv`, and a `config_snapshot.yaml` are written under `runs/<DATE>/<preset>/<task>/<model>/`.
  - A consolidated CSV/JSON summary per preset is generated in `runs/<DATE>/summaries/`.

---

## 2) Work Packages Overview
- **WP1** — Fix eval harness (decoding + chat templates)
- **WP2** — Metrics correctness & richer compute section
- **WP3** — Model registry & generation config
- **WP4** — Presets & run matrix expansion (baselines + memory)
- **WP5** — QA (smoke tests, unit tests) & CI fast‑fail
- **WP6** — Full execution & artifact consolidation
- **WP7** — Documentation updates

Each WP below contains **Codex tasks** (ready-to-use prompts) and **Human tasks**.

---

## WP1 — Fix eval harness (decoding + chat templates)

### Codex Task C1 — Slice decoding to remove prompt echo
**Files:** `scripts/eval_model.py`  
**Change:** Decode only generated continuation.  
**Acceptance:** In `metrics.csv`, `pred` no longer starts with the full `prompt`.

**Prompt for Codex:**
> Update `scripts/eval_model.py` so that decoding excludes the input tokens. After calling `model.generate(**inputs, ...)`, slice the output tensor with `out[:, inputs["input_ids"].shape[-1]:]` before decoding. Use `pad_token_id=tokenizer.pad_token_id` and `eos_token_id=tokenizer.eos_token_id` in `generate(...)`. Trim whitespace from final `pred`.

### Codex Task C2 — Add chat-aware encoding with fallback
**Files:** `scripts/eval_model.py`, new helper `src/eval/encode.py`  
**Change:** Implement `encode_prompt(tokenizer, prompt, device)` that uses `tokenizer.apply_chat_template` if available; otherwise falls back to plain `.encode`. Default system message: “You are a helpful assistant.”  
**Acceptance:** For Qwen‑Instruct models, `tokenizer.chat_template` path is taken and completions are well-formed.

**Prompt for Codex:**
> Create `src/eval/encode.py` exposing `encode_prompt(tokenizer, prompt, device)` which uses `apply_chat_template` when available with:
> ```python
> messages=[{"role":"system","content":"You are a helpful assistant."}, {"role":"user","content":prompt}]
> input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
> ```
> Else: `input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]`. Import and use this in `scripts/eval_model.py`.

---

## WP2 — Metrics correctness & richer compute

### Codex Task C3 — Fix `time_ms_per_100` and add token counters
**Files:** `scripts/eval_model.py` (metrics assembly)  
**Change:** Compute `time_ms_per_100 = (100 * total_time_ms) / max(1, total_tokens)`. Add `compute.input_tokens`, `compute.generated_tokens`, and `compute.total_tokens`.  
**Acceptance:** Values match a manual recomputation from `metrics.csv` latencies and tokenizer counts.

**Prompt for Codex:**
> In compute metrics, add fields: `input_tokens`, `generated_tokens`, `total_tokens`. Compute `time_ms_per_100` using total wall time divided by total token count times 100. Ensure `latency_ms_mean` remains per-item average.

### Codex Task C4 — Add small audit sample
**Files:** `scripts/eval_model.py`  
**Change:** Log first 10 items (prompt, answer, pred) to `audit_sample.jsonl` per run.  
**Acceptance:** File exists and is human-readable JSONL.

**Prompt for Codex:**
> Write the first 10 (or all if <10) records as JSONL to `<run_dir>/audit_sample.jsonl` with keys: `id`, `prompt`, `answer`, `pred` (truncated to 2k chars).

---

## WP3 — Model registry & generation config

### Codex Task C5 — Introduce `configs/models.yaml` and a loader
**Files:** `configs/models.yaml`, `src/eval/models.py`  
**Change:** A small registry keyed by model id with flags: `use_chat_template`, optional `system_prompt`, `eos_token_id`, `pad_token_id`, and default `max_new_tokens`. Provide sane defaults; override per model.  
**Acceptance:** Qwen2.5‑Instruct is marked `use_chat_template=true`.

**Prompt for Codex:**
> Add `configs/models.yaml` with entries like:
> ```yaml
> Qwen/Qwen2.5-1.5B-Instruct:
>   use_chat_template: true
>   system_prompt: "You are a helpful assistant."
>   max_new_tokens: 256
> ```
> Create `src/eval/models.py` to load this file and expose `ModelConfig get(model_id)` returning a dataclass. Use it inside `eval_model.py` to configure encoding and generation.

### Codex Task C6 — CLI flag to override chat template
**Files:** `scripts/eval_model.py` (arg parsing)  
**Change:** Flags `+force_chat=true/false` and `+force_no_chat=true/false` (mutually exclusive) to override registry behavior.  
**Acceptance:** Flags take precedence in runs and are logged to `meta.json`.

**Prompt for Codex:**
> Add optional overrides `force_chat` and `force_no_chat` (default None). If set, they override `use_chat_template`. Persist the resolved value in `meta.json` under `model.chat_template_used`.

---

## WP4 — Presets & run matrix expansion

### Codex Task C7 — Add baseline presets
**Files:** `configs/presets/baselines/core.yaml`, `configs/presets/baselines/rag.yaml`, `configs/presets/baselines/longctx.yaml`  
**Change:** Ensure three baseline presets exist with appropriate toggles:
- `core`: memory off, replay=0
- `rag`: retrieval on, no memory consolidation
- `longctx`: concatenate supporting context into prompt (no retrieval at inference)
**Acceptance:** Running each preset switches the flags in `meta.json` accordingly.

**Prompt for Codex:**
> Create/verify the three baseline preset YAMLs with explicit flags: `gating_enabled`, `replay.cycles`, `retrieval.enabled`, `long_context.enabled`, etc. Include comments describing each preset’s intent.

### Codex Task C8 — Add memory presets
**Files:** `configs/presets/memory/hei_nw.yaml`, `sgc_rss.yaml`, `smpd.yaml`  
**Change:** Enable replay (`cycles: 1` default), `gating_enabled: true`, and retrieval on.  
**Acceptance:** `metrics.json` includes **pre** and **post** sections and a **delta** block; retrieval/gating metrics are populated.

**Prompt for Codex:**
> Add three memory presets enabling replay and gating. Ensure the evaluation loop runs a **pre** pass, applies replay per preset, then runs a **post** pass, and finally computes deltas. Persist counters: retrieval.requests, retrieval.hits, r@5, r@10; gates.open_rate, gates.mean_score.

### Codex Task C9 — Matrix sweeping & summaries
**Files:** `scripts/run_matrix.py` or augment `eval_model.py` sweep; `scripts/summarize_runs.py`  
**Change:** Support `+run_matrix=true` to sweep over tasks `{episodic,semantic,spatial}`, `n`, `seeds`, `presets`, and produce per‑preset summary CSV/JSON under `runs/<DATE>/summaries/`.  
**Acceptance:** One command launches the full grid; summary files are produced.

**Prompt for Codex:**
> Add a matrix driver that iterates over configured lists (`tasks`, `n_values`, `seeds`, `presets`) and dispatches `eval_model.py` subprocesses with proper `outdir`. After completion, aggregate `metrics.json` files into `summaries/<preset>_summary.csv` with columns: task, model, n, seed, EM, F1, tokens, time_ms_per_100.

---

## WP5 — QA & CI

### Codex Task C10 — Unit tests
**Files:** `tests/test_encoding.py`, `tests/test_metrics.py`  
**Change:** Tests for (a) chat-template encoding vs. plain encoding; (b) decode slicing correctness; (c) `time_ms_per_100` calculation.  
**Acceptance:** Tests pass locally.

**Prompt for Codex:**
> Add unit tests that mock tokenizer/model to verify (1) `encode_prompt` uses chat template when available, (2) generated slice excludes input ids, (3) metrics math is correct for a toy batch.

### Codex Task C11 — Smoke target with fast-fail
**Files:** `scripts/smoke_eval.sh`, `Makefile` or `tox.ini`  
**Change:** A target that runs `episodic` with `n=50`, `seed=1337`, `preset=baselines/core` and fails if EM==0 and F1<0.20.  
**Acceptance:** CI fails on regressions.

**Prompt for Codex:**
> Create a shell script or Make target `smoke` that runs a small eval and checks the thresholds; exit non‑zero if unmet. Integrate into CI config if present.

---

## WP6 — Full execution & artifact consolidation

### Human Task H1 — Dev smoke after WP1–WP2
Run (replace `$DATE`):
```bash
DATE=$(date +%Y%m%d_%H%M)
python scripts/eval_model.py preset=baselines/core task=episodic n=50 seed=1337 \
  model=Qwen/Qwen2.5-1.5B-Instruct outdir=runs/$DATE/baselines/core/Qwen2.5-1.5B
```
**Verify:** `pred` is not echoing; `EM>0` or `F1≥0.20`; `audit_sample.jsonl` exists; `time_ms_per_100` matches manual recompute.

### Human Task H2 — Baseline grid
```bash
DATE=$(date +%Y%m%d_%H%M)
python scripts/eval_model.py +run_matrix=true date=$DATE \
  presets="[baselines/core,baselines/rag,baselines/longctx]" \
  tasks="[episodic,semantic,spatial]" n_values="[50,200,1000]" \
  seeds="[1337,2025,4242]" \
  model=Qwen/Qwen2.5-1.5B-Instruct outdir=runs/$DATE
```

### Human Task H3 — Memory grid
```bash
python scripts/eval_model.py +run_matrix=true date=$DATE \
  presets="[memory/hei_nw,memory/sgc_rss,memory/smpd]" \
  tasks="[episodic,semantic,spatial]" n_values="[50,200,1000]" \
  seeds="[1337,2025,4242]" \
  model=Qwen/Qwen2.5-1.5B-Instruct outdir=runs/$DATE
```
**Verify:** `metrics.json` has `pre`, `post`, `delta`; retrieval/gating metrics are non‑zero.

### Human Task H4 — Summaries
After any matrix run:
```bash
python scripts/summarize_runs.py runs/$DATE --out runs/$DATE/summaries
```
**Verify:** Summary CSV/JSON exist; spot-check deltas for memory runs.

---

## WP7 — Documentation updates

### Codex Task C12 — Update docs
**Files:** `README.md`, `MILESTONE_9_PLAN.md`, `PROJECT_PLAN.md`, `EVAL_PLAN.md`  
**Change:** Reflect new presets, model registry, seeds policy (dev=1, final=3), sanity thresholds, and how to run matrices & summaries.  
**Acceptance:** Documents include copy‑pasteable commands (from H2–H4) and link to CI smoke target.

**Prompt for Codex:**
> Update the four docs with: (1) explanation of presets, (2) model registry and chat templates, (3) seeds policy, (4) matrix commands, (5) acceptance/gate criteria, (6) artifact layout. Keep examples aligned with Hydra/CLI style used here.

---

## Final checklist before closing M9
- [ ] WP1 implemented and verified (no echo; chat template applied)
- [ ] WP2 metrics corrected with token counters
- [ ] WP3 model registry wired and overridable
- [ ] WP4 presets present (core/rag/longctx + hei_nw/sgc_rss/smpd)
- [ ] WP5 tests & smoke target green
- [ ] WP6 baselines + memory grids executed; summaries generated
- [ ] WP7 docs updated; gate met

---

## Notes on seeds
- Use **one seed (1337)** for dev/smoke to save ~66% time.
- Use **three seeds (1337, 2025, 4242)** for reported results and milestone gates to stabilize metrics via macro‑averaging.
