# Consolidation vs. Persistence — What the Research Demands and the Best Implementation
**Generated:** 2025-08-25 13:21

This note re-reads `research/experiment-synthesis.md` and reconciles it with our current design/impl. It answers:
1) *What are we doing here, and why?*  
2) *Is “consolidation” supposed to be LoRA/adapters/distillation into weights (cortex) or JSON persistence?*  
3) *What’s the best implementation path for us?*

---

## 1) What the research document actually says (essentials)
The synthesis models the hippocampus as a **fast, sparse, labile store** (episodic/relational/spatial) which **replays** to gradually train more **stable, parametric cortical knowledge**. In LLM terms:

- **Hippocampal analogue** = non‑parametric **stores + retrieval + gating + replay** (episodes, tuples, places).  
- **Cortical analogue** = the **LLM’s weights** (or very stable adapters) that acquire knowledge **offline** via **prioritized replay/distillation** (“systems consolidation”).  
- **Replay** is not just fetching at inference; it’s a *training-time* process that uses stored traces to update the cortex.  
- **Short-term vs long-term**: early traces live in the hippocampal store; later, valuable regularities migrate into weights (reduced dependency on retrieval).

**Implication:** JSON/Parquet **persistence is only a substrate for the hippocampal store** (so we can replay later and span sessions). **Consolidation** is a **training step** (LoRA/adapters/distillation) that moves knowledge **into weights**.

---

## 2) Why your “LoRA with recalled memories” intuition is right
The document explicitly frames **systems consolidation** as *offline replay to cortex*. In practice for an LLM, this means:
- Sample **prioritized episodes/schemas** from the store (“what mattered”).  
- Run **replay batches** that *teach the base model* without the store at inference.  
- Use **parameter‑efficient training** (LoRA/adapters) or **distillation** so the core model internalizes the knowledge.

Therefore:
- **JSON persistence ≠ consolidation.** It’s an **engineering enabler** (to save episodes across runs).  
- **Consolidation = weight updates** (LoRA/adapters/distill) **driven by the persisted episodes** via replay.

---

## 3) What we have today
- **Intra‑run memory:** episodic retrieval, gating, and replay toggles exist, but effects are evaluated within a single run.  
- **Cross‑run persistence:** not wired yet (this caused confusion).  
- **Consolidation into weights:** not implemented (no LoRA/adapters/distillation step).  
- **Baselines:** core, some memory presets; missing **span‑short decoding**, **long‑context**, and **RAG** ablations that clarify where gains come from.

---

## 4) Best implementation path (pragmatic & faithful to the research)

### Stage A — Make hippocampal stores durable across runs (engineering substrate)
**Goal:** Save and reload episodic/relational/spatial stores to enable replay later.  
- API: `store.save(dir, session_id)` / `store.load(dir, session_id)` (use JSONL or Parquet; JSON is fine for traceability).  
- CLI: `--store_dir`, `--session_id`, `--persist=true`.  
- Protocol: `mode=teach` (ingest + write), `mode=test` (load + read), `mode=replay` (sampling worker).  
- Acceptance: after process restart, the model with **memory disabled** cannot answer; with store **enabled**, it can (proves the store carries knowledge forward).

> Why this first? Without a durable store, there is **nothing to replay** for consolidation.

### Stage B — Add **offline systems consolidation** (LoRA/adapters/distillation)
**Goal:** Train the “cortex” (weights) from the hippocampal store via replay.  
- **PEFT**: attach **LoRA** (r=8–16) or **parallel adapters** to attention and MLP blocks (target matrices: `q_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj`).  
- **Data**: build a **replay dataset** from saved stores:
  - Positive examples: (prompt → short answer), possibly with *teacher outputs* produced **with memory enabled** to distill rationale/format.  
  - Anti‑interference: mix old and new episodes; add distractors and near‑duplicates.  
- **Objectives**:
  - **Supervised**: next‑token loss on short answers / rationales.  
  - **Distillation** (optional): KL on logits vs “teacher-with-memory” outputs.  
- **Scheduling**: prioritized replay (by salience/novelty/usage), spaced intervals (`cycles`), small LR, early stopping.  
- **Evaluation**:
  1. **Pre**: run tasks with **memory OFF** → baseline EM/F1.  
  2. **Post**: after LoRA training, run same tasks with **memory OFF** → **improvement proves consolidation**.  
  3. **Ablations**: compare to long‑context & RAG; ensure we’re not just overfitting phrasing.  
- **Safety**: elastic weight consolidation is overkill initially; stick to small LoRA ranks, low LR, and replay mixing to reduce forgetting.

### Stage C — Integrate at inference (choose policy)
- **Preferred**: keep LoRA **attached** at inference (cheap & reversible), optionally **merge** into base if stable.  
- **Routing**: light **gate** that prefers parametric answers; if confidence low, fall back to store retrieval (best of both).

---

## 5) Minimal concrete plan & tasks

### 5.1 New scripts & configs
- `scripts/teach.py` — ingest facts with memory **writes on**, save stores.  
- `scripts/replay_consolidate.py` — build replay dataset from stores and **train LoRA/adapters** (Hugging Face PEFT).  
- `scripts/test_consolidation.py` — evaluate **with memory OFF** before/after LoRA.  
- Configs:
  - `configs/eval/baselines/span_short.yaml` (chat ON, span-only prompt, `max_new_tokens≤8`).  
  - `configs/consolidation/lora_small.yaml` (r=8–16, target matrices above, LR~1e-4–3e-4, steps~2–10k).  
  - `configs/replay/scheduler.yaml` (priority, spacing, mix ratios).

### 5.2 Data pipeline (from stores → replay dataset)
- Build `ReplayDataset` that samples from `{episodic, relational, spatial}` with ratios (e.g., 60/30/10).  
- Each item yields: `input_ids` (question or cue) and `labels` (short span), plus optional **teacher** text from a memory-enabled run for distillation.  
- On-the-fly **hard negatives** (similar but wrong episodes) to stress gates.

### 5.3 Acceptance criteria
- **Cross-session retention**: answers become available in a new process **with store ON**.  
- **Consolidation**: with **store OFF**, **post‑LoRA EM improves by ≥ +0.20** vs pre-LoRA on `episodic@50`.  
- **No blow‑up**: core benchmarks within −1% of pre-LoRA on a small sanity set (guard against catastrophic forgetting).  
- **Telemetry**: artifacts include store sizes, replay counts, LoRA config hash, and pre/post EM/F1.

### 5.4 Codex task list (ready to paste)
1. **[stores]** Implement `save/load` for episodic/relational/spatial stores (JSONL/Parquet, versioned).  
2. **[cli]** Add `--store_dir`, `--session_id`, `--persist`, `--mode={teach,replay,test}` to `eval_model.py` (or new scripts).  
3. **[replay]** Implement `ReplayDataset` + prioritized sampler (salience/novelty/usage).  
4. **[peft]** Integrate Hugging Face **PEFT**: LoRA attach/detach; config under `configs/consolidation/lora_small.yaml`.  
5. **[trainer]** `scripts/replay_consolidate.py`: train LoRA from the replay dataset; support optional **distillation** from “teacher-with-memory”.  
6. **[eval]** `scripts/test_consolidation.py`: measure EM/F1 with **memory OFF** pre/post LoRA; log deltas.  
7. **[baselines]** Add `baselines/span_short.yaml`, `baselines/longctx.yaml`, `baselines/rag.yaml`; include in sweeps.  
8. **[ci]** Post-run assertions: for memory presets, `retrieval.requests>0`; for consolidation tests, require **post‑LoRA uplift** with memory OFF.  
9. **[docs]** Update `EVAL_PLAN.md`, `MILESTONE_9_PLAN.md`; add **MILESTONE_9_5_PLAN.md — Cross‑Session Consolidation** with the above criteria.

---

## 6) Why this is the “best” path for us
- **Faithful to the research**: separates **hippocampal store** (non‑parametric, replayable) from **cortical consolidation** (parametric, via LoRA/adapter/distillation).  
- **Practical**: PEFT keeps compute low and is reversible; JSON/Parquet persistence is simple, transparent, and auditable.  
- **Measurable**: the decisive test is **improvement with memory OFF after LoRA**, which directly demonstrates consolidation.

---

## 7) FAQ
- **Is JSON “awkward”?** It’s not the *goal*; it’s the **plumbing** that lets us replay later. Use Parquet if you prefer columnar IO; either is fine.  
- **Why not fine‑tune the full model?** We can later; **LoRA first** lowers risk and gives faster iteration.  
- **What about just using RAG?** RAG ≠ consolidation; it’s retrieval‑only. We include RAG as a baseline, not as the cortical mechanism.  
- **Do we need long-context?** Include it as an ablation to show that gains aren’t from brute force context alone.

---

### One‑line summary
**Consolidation** in our project should mean: *offline LoRA/adapter/distillation training driven by replayed episodes from a persisted hippocampal store* — not “just” saving JSON. JSON/Parquet persistence is the means; **LoRA replay** is the end.
