# Consolidation Readiness Review — Code vs. Plan

**Scope:** Verify that Codex tasks in `MILESTONE_9_5_PLAN.md` are implemented so humans can start, and check alignment with `research/experiment-synthesis.md` + `review/consolidation-design-clarification.md`.

## Summary

- Stores, replay, LoRA trainer, evaluation harness, ablations, telemetry, and CI guardrails are present and wired. Overall readiness: **✅**

- The only blocker for human runs is **CLI syntax** in the docs using `--mode`/`--persist` etc. — Hydra expects `key=value`. Fixing the command examples unblocks execution.


## Implementation status (C12–C28)

**C12** ✅

  - hippo_mem/episodic/store.py (save/load JSONL|Parquet)

  - hippo_mem/relational/kg.py (save/load JSONL|Parquet)

  - hippo_mem/spatial/map.py (save/load)



**C13** ✅

  - hippo_mem/common/io.py (atomic_write_jsonl/json; per‑path locks)



**C14** ✅

  - scripts/eval_model.py (Hydra entrypoint)

  - hippo_mem/eval/harness.py (mode=teach|test, persist, store_dir, session_id)



**C15** ✅

  - hippo_mem/eval/harness.py (teach path saves stores when persist=true)



**C16** ✅

  - hippo_mem/eval/harness.py (test path loads stores, memory_off flag)



**C17** ✅

  - hippo_mem/consolidation/replay_dataset.py (ReplayDataset)



**C18** ✅

  - hippo_mem/consolidation/replay_dataset.py (teacher signals)

  - tests/test_replay_teacher.py



**C19** ✅

  - hippo_mem/consolidation/replay_dataset.py (policy, cycles, noise_level, max_items)

  - tests/test_replay_sampling.py

  - tests/test_replay_sampler_knobs.py



**C24** ✅

  - scripts/test_consolidation.py (pre/post, delta reporting)



**C25** ✅

  - configs/eval/baselines/span_short.yaml

  - configs/eval/baselines/longctx.yaml

  - configs/eval/baselines/rag.yaml

  - tests/test_baseline_presets.py



**C26** ✅

  - hippo_mem/common/telemetry.py (retrieval stats)

  - hippo_mem/eval/harness.py (metrics.json includes retrieval/gates/replay/store/refusal_rate)



**C27** ✅

  - tests/test_ci_guardrails.py (retrieval_requests guard; uplift; refusal_rate)



**C28** ✅

  - README.md, PROJECT_PLAN.md, EVAL_PLAN.md, MILESTONE_9_PLAN.md (updated)



## Alignment with the design

- Research frames consolidation as offline replay to **weights** (cortex). Code provides `ReplayDataset`, LoRA trainer, and `test_consolidation.py` to measure **post‑LoRA uplift with memory OFF**.

- Hippocampal stores are non‑parametric with gating & retrieval. Code has episodic (FAISS + neuromodulated writes), relational (KG + schema/index), and spatial (place/route graph) modules with maintenance hooks.

- Persistence is treated as plumbing (JSONL/Parquet) with schema tags; atomic/thread‑safe I/O is implemented. This matches the clarified design note.

- Baselines include span‑short, long‑context and simple‑RAG presets to isolate consolidation effects (tests assert fields).


## Issues found (with fixes)

### Hydra CLI vs. double‑dash flags in docs

- **Impact:** High (blocks human commands in H2 if copied verbatim)

- **Why:** Hydra expects key=value overrides (e.g., mode=teach), not --mode teach.

- **Where:** MILESTONE_9_5_PLAN.md, README.md, PROJECT_PLAN.md

- **Fix:** Change all examples to Hydra overrides: `mode=teach persist=true store_dir=runs/... session_id=$SID`.


### Unreferenced/placeholder model config

- **Impact:** Low

- **Why:** configs/model/qwen2-1_5b.yaml is a stub with a TODO and isn’t wired. Human commands pass `model=Qwen/Qwen2.5-1.5B-Instruct` directly, which is fine.

- **Where:** configs/model/qwen2-1_5b.yaml

- **Fix:** Either remove the stub or flesh it out and document when to use it. Not a blocker.


### Experiment task stubs

- **Impact:** None (docs quality)

- **Why:** `experiments/*/tasks.md` still say TODO.

- **Where:** experiments/hei_nw/tasks.md, experiments/sgc_rss/tasks.md, experiments/smpd/tasks.md

- **Fix:** Populate with concrete steps or delete to avoid confusion.


## Mitigation steps (actionable)

- **Fix command examples (Hydra syntax)** — Update all docs and MILESTONE_9_5_PLAN H2/H3/H5 commands to use key=value overrides. Example:
```bash
DATE=$(date +%Y%m%d_%H%M); SID=seed1337
python scripts/eval_model.py preset=memory/hei_nw task=episodic n=200 seed=1337 \
  mode=teach persist=true store_dir=runs/$DATE/stores session_id=$SID \
  model=Qwen/Qwen2.5-1.5B-Instruct outdir=runs/$DATE/memory/teach
```

- **Sanity‑check the pipeline end‑to‑end (dry runs)** — With `models/tiny-gpt2` to avoid downloads:
```bash
python scripts/eval_model.py preset=baselines/core task=episodic n=5 seed=0 \
  model=models/tiny-gpt2 outdir=runs/smoke/core
python scripts/eval_model.py preset=memory/hei_nw task=episodic n=5 seed=0 \
  mode=teach persist=true store_dir=runs/smoke/stores session_id=smoke \
  model=models/tiny-gpt2 outdir=runs/smoke/teach
python scripts/test_consolidation.py --phase pre --suite episodic --n 5 --seed 0 \
  --model models/tiny-gpt2 --outdir runs/smoke/pre
python scripts/test_consolidation.py --phase post --suite episodic --n 5 --seed 0 \
  --model models/tiny-gpt2 --adapter runs/smoke/adapter --pre_dir runs/smoke/pre \
  --outdir runs/smoke/post || true
```

- **Optional: accept both syntaxes** — If you want to keep `--mode` style, add a tiny wrapper (e.g., `scripts/teach.py`) translating `--flags` to Hydra overrides and then calling the harness.

- **Housekeeping** — Either remove or finish `configs/model/qwen2-1_5b.yaml` and fill `experiments/*/tasks.md` to avoid confusion.


## Go/No‑Go

- **Go**, after updating the Hydra syntax in human‑task commands. Everything else is optional polishing.
