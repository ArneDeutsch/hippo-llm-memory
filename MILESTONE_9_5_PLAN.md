# MILESTONE_9_5_PLAN — Cross‑Session Consolidation (Hippocampal → Cortical)
**Generated:** 2025-08-25 13:30


## 0) Rationale (from research/experiment-synthesis.md)
The research frames the hippocampus as a **fast, labile, sparse store** (episodic/relational/spatial) that supports **replay** to gradually consolidate knowledge into a **stable, parametric cortex**. In LLM terms:
- Hippocampal analogue → non‑parametric stores + retrieval + gating + replay.
- Cortical analogue → LLM **weights** (or adapters) trained **offline** via prioritized replay (systems consolidation).
- Persistence (JSON/Parquet) is **plumbing**, not the goal: it enables replay across sessions; **consolidation** is demonstrated by **better answers with memory OFF** after training.

---

## 1) Objectives & Success Criteria
**Objective:** Implement and validate **systems consolidation across sessions**: store experiences, replay them offline to train LoRA/adapters, then show **improved recall with memory disabled**.

**Primary success criteria**
1. On `episodic@50` (seed=1337), **EM uplift ≥ +0.20** absolute **with memory OFF** after LoRA vs before LoRA (core baseline).
2. **Cross‑session retention**: after process restart, answers remain available **only** when loading the saved store pre‑consolidation, and **even with memory OFF after consolidation** due to parametric learning.
3. **No catastrophic forgetting**: a small sanity suite regresses by ≤ 1% after LoRA.
4. **Telemetry correctness**: memory presets show `retrieval.requests > 0` and replay counts > 0; consolidation runs log LoRA config hash and replay statistics.

**Secondary criteria**
- Relational & spatial suites show non‑zero gains post‑LoRA.
- Ablations (long‑context, RAG, span‑short) clarify the source of improvements.

---

## 2) Deliverables
- **Persistence layer** for episodic/relational/spatial stores: `save(dir, session_id)` / `load(dir, session_id)`.
- **Two‑phase+ harness**: `mode={teach,replay,test}` with CLI support and run scripts.
- **ReplayDataset** with prioritized sampling (salience/novelty/usage) and optional teacher‑distillation targets.
- **Consolidation training**: PEFT LoRA adapters attached to attention & MLP; training script with configs.
- **Evaluation scripts** for pre/post consolidation (memory OFF) + ablations.
- **Baselines**: `baselines/span_short.yaml`, `baselines/longctx.yaml`, `baselines/rag.yaml`.
- **Telemetry & CI**: preflight resolved‑config dump; post‑run assertions; refusal‑rate guard.
- **Documentation** updates: README, PROJECT_PLAN, EVAL_PLAN, MILESTONE_9_PLAN; this `MILESTONE_9_5_PLAN.md` added.

**Artifacts**
- Saved stores under `runs/<DATE>/stores/<preset>/<suite>/<model>/<session_id>/...` (JSONL or Parquet).
- LoRA checkpoints & config under `runs/<DATE>/consolidation/<model>/<session_id>/...`.
- Metrics JSON/CSV with retrieval/replay counters and pre/post EM/F1.

---

## 3) Work Breakdown — Codex Tasks (C12–C28)

### Store persistence & CLI (C12–C16)
- **C12.** Implement `save/load` for all stores (episodic/relational/spatial). Format: JSONL (default) with schema version; support Parquet optionally.
- **C13.** Thread-safe I/O utilities; atomic writes with temp files + rename.
- **C14.** Extend `eval_model.py` (or factor new `scripts/teach.py`) with CLI:
  - `mode={teach,replay,test}`
  - `store_dir`, `session_id`, `persist={true,false}`
- **C15.** On `mode=teach`: enable writes & gates; **do not** compute EM; save stores to `store_dir/session_id`.
- **C16.** On `mode=test`: load stores if present (for memory‑on runs); allow flag `--memory_off` to explicitly disable retrieval for consolidation tests.

### Replay & dataset (C17–C19)
- **C17.** Implement `ReplayDataset` that samples from saved stores with ratios (episodic/relational/spatial) and prioritization (usage/salience/novelty).
- **C18.** Optional teacher distillation: generate teacher targets by running the same prompts **with memory ON**; store text/logits if available.
- **C19.** Add sampler knobs: `replay.policy={uniform,priority,spaced}`, `replay.cycles`, `replay.noise_level`, `replay.max_items`.

### PEFT LoRA training (C20–C23)
- **C20.** Integrate Hugging Face PEFT; attach LoRA to `{q_proj,v_proj,o_proj,up_proj,down_proj}` with default ranks r=8–16.
- **C21.** New script `scripts/replay_consolidate.py`:
  - Ingests `ReplayDataset`, supports SFT + optional KL distillation.
  - Logs: steps, LR, loss, replay counts, LoRA config hash.
- **C22.** Configs: `configs/consolidation/lora_small.yaml` (LR, rank, steps, warmup, grad_accum).
- **C23.** Export LoRA adapters and merge option (`--merge`) to bake into base if desired.

### Evaluation, telemetry & CI (C24–C28)
- **C24.** `scripts/test_consolidation.py`: run **pre** (memory OFF), then **post** (memory OFF with LoRA attached); report deltas.
- **C25.** Add ablations presets:
  - `configs/eval/baselines/span_short.yaml` (chat ON, shortest‑span system prompt, `max_new_tokens≤8`),
  - `configs/eval/baselines/longctx.yaml` (concat evidence, retrieval OFF),
  - `configs/eval/baselines/rag.yaml` (NN retrieve only).
- **C26.** Telemetry: ensure `metrics.json` contains `retrieval.requests`, `gates.attempts/accepts`, `replay.samples`, `store.size`, `refusal_rate`.
- **C27.** CI guardrails:
  - Fail memory presets if `retrieval.requests == 0`.
  - Fail consolidation suite if post‑LoRA uplift < +0.20 EM on `episodic@50` (seed=1337).
  - Fail if refusal rate > 0.5 on span suites.
- **C28.** Docs: update README, PROJECT_PLAN, EVAL_PLAN, MILESTONE_9_PLAN; include this plan and runbook.

---

## 4) Human Tasks (H2–H7)

- **H2.** Run **teach** session to create stores:\
  ```bash
    DATE=$(date +%Y%m%d_%H%M); SID=seed1337
    python scripts/eval_model.py preset=memory/hei_nw task=episodic n=200 seed=1337 \
      mode=teach persist=true store_dir=runs/$DATE/stores session_id=$SID \
      model=Qwen/Qwen2.5-1.5B-Instruct outdir=runs/$DATE/memory/teach
  ```
- **H3.** Pre‑consolidation baseline (memory OFF):\
  ```bash
  python scripts/test_consolidation.py --phase pre \
    --suite episodic --n 50 --seed 1337 --memory_off true \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --outdir runs/$DATE/consolidation/pre
  ```
- **H4.** Train **LoRA via replay**:\
  ```bash
  python scripts/replay_consolidate.py \
    --store_dir runs/$DATE/stores --session_id $SID \
    --config configs/consolidation/lora_small.yaml \
    --outdir runs/$DATE/consolidation/lora
  ```
- **H5.** Post‑consolidation test (memory OFF):\
  ```bash
  python scripts/test_consolidation.py --phase post \
    --suite episodic --n 50 --seed 1337 --memory_off true \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --lora runs/$DATE/consolidation/lora \
    --outdir runs/$DATE/consolidation/post
  ```
- **H6.** Run **ablations** for attribution:\
  ```bash
  # Span-short baseline
  python scripts/eval_model.py preset=baselines/span_short task=episodic n=50 seed=1337 \
    model=Qwen/Qwen2.5-1.5B-Instruct outdir=runs/$DATE/baselines/span_short

  # Long-context (concat)
  python scripts/eval_model.py preset=baselines/longctx task=episodic n=50 seed=1337 \
    model=Qwen/Qwen2.5-1.5B-Instruct outdir=runs/$DATE/baselines/longctx

  # Simple RAG
  python scripts/eval_model.py preset=baselines/rag task=episodic n=50 seed=1337 \
    model=Qwen/Qwen2.5-1.5B-Instruct outdir=runs/$DATE/baselines/rag
  ```
- **H7.** Spot‑check relational & spatial suites (n=50 each) pre/post LoRA.

---

## 5) Runbook Notes & Defaults

**Decoding for span tasks** (applies to baselines and memory presets when scoring EM):
- Keep chat template ON.
- System prompt: “Answer with the **exact shortest span** from the prompt. No explanations.”
- `max_new_tokens`: 8–16.

**Replay sampling** (default):
- Ratios: episodic 60%, relational 30%, spatial 10%.
- Policy: `priority` (higher when recently used or high salience).
- Spacing: `replay.cycles=1` initially; can iterate.

**LoRA defaults**:
- r=8 (first pass), α=16–32, dropout=0.05, LR=1e‑4..3e‑4, steps=2k–10k depending on store size.
- Targets: `{q_proj, v_proj, o_proj, up_proj, down_proj}`.

**Data hygiene**:
- Version store schema; include hash of `configs` in metrics.
- Keep “teacher-with-memory” outputs separate for optional distillation.

---

## 6) Acceptance Checklist

**Automation (CI)**
- [ ] Memory preset runs: `retrieval.requests > 0`; `replay.samples > 0` when enabled.
- [ ] Consolidation suite: post‑LoRA EM uplift ≥ +0.20 on `episodic@50` (seed=1337) with **memory OFF**.
- [ ] Refusal‑rate guard: ≤ 0.5 on span suites.
- [ ] Sanity suite: ≤ 1% degradation post‑LoRA.
- [ ] Artifacts uploaded: stores, LoRA adapters, metrics JSON/CSV, resolved configs.

**Manual review (Human)**
- [ ] Stores reload in a new process; answers appear with memory ON pre‑consolidation.
- [ ] After consolidation, answers appear with **memory OFF** (proves parametric transfer).
- [ ] Ablations: long‑context & RAG do not fully explain gains.

---

## 7) Risks & Mitigations
- **Overfitting / forgetting:** use low LR, small LoRA ranks, mixed replay, sanity suite checks.
- **Data quality:** log refusal rate and store size; deduplicate near‑identical episodes.
- **Infra drift:** hash resolved configs; deterministic seeds; pin versions for PEFT/training stack.

---

## 8) Documentation Updates
- Add sections to **README.md** and **EVAL_PLAN.md** explaining the cross‑session protocol and how consolidation is measured.
- Update **MILESTONE_9_PLAN.md** to reference the `span_short` baseline and link to this milestone for consolidation.

---

## 9) Timeline (suggested)
- **Day 1–2:** Persistence & CLI (C12–C16).
- **Day 3:** Replay dataset & sampler (C17–C19).
- **Day 4–5:** LoRA training script & configs (C20–C23).
- **Day 6:** Evaluation, telemetry & CI (C24–C28); docs; dry‑run.

---

## 10) Appendix — Example Config Stubs

**configs/eval/baselines/span_short.yaml**
```yaml
memory: null
replay: { cycles: 0 }
retrieval: { enabled: false }
long_context: { enabled: false }
gating_enabled: false
use_chat_template: true
system_prompt: "Answer with the exact shortest span from the prompt. No explanations."
max_new_tokens: 8
```

**configs/consolidation/lora_small.yaml**
```yaml
peft:
  method: lora
  rank: 8
  alpha: 16
  dropout: 0.05
targets: [q_proj, v_proj, o_proj, up_proj, down_proj]
train:
  lr: 2.0e-4
  steps: 4000
  warmup_steps: 200
  grad_accum: 2
  batch_size: 16
replay:
  policy: priority
  cycles: 1
  mix: { episodic: 0.6, relational: 0.3, spatial: 0.1 }
```

**configs/replay/scheduler.yaml**
```yaml
policy: priority
priority:
  weights: { salience: 0.5, usage: 0.3, novelty: 0.2 }
noise_level: 0.0
max_items: 100000
```

---

### One‑line outcome
This milestone proves **true systems consolidation**: after replay‑driven **LoRA training**, the model answers from its **weights** (memory OFF), not just from an external store.
