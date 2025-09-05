# Evaluation Strategy Review — LLM Hippocampus Project (run_20250904, n=50, seed=1337)

**Scope.** I read the ground-truth design in `research/experiment-synthesis.md`, the plan in `EVAL_PLAN.md` and `EVAL_PROTOCOL.md`, the dataset generators under `hippo_eval/tasks/`, and inspected the latest artifacts in `runs/run_20250904/` and `reports/run_20250904/`.

---

## TL;DR — Are the concerns valid?

**Yes.** The current evaluation mostly keeps *all relevant facts inside the same prompt*, which saturates baselines and blurs what “memory” modules add. The “FLUSH” marker in `episodic_cross` is a **text token**, not a session boundary, so nothing is truly out‑of‑context. Memory leakage between items is also possible within a teaching pass because retrieval is enabled while writing. And the **SMPD** tests do not probe iterative map building or macro reuse; they pose one‑shot shortest‑path riddles.

**Bottom line:** the pipeline needs a conceptual update to (a) separate *teach* vs *test* across sessions/files, (b) enforce isolation so memory from one item does not spill into others unless explicitly intended, and (c) create SMPD tasks that require multi‑episode learning.

---

## 1) What the three algorithms are supposed to do (from `experiment-synthesis.md`)

- **HEI‑NW — Hippocampal Episodic Index with Neuromodulated Writes.**  
  Persistent episodic store keyed by sparse DG‑like codes; salience‑gated writes; prioritized offline replay; cross‑attend recalled traces during inference.

- **SGC‑RSS — Schema‑Guided Consolidation with a Relational Semantic Store.**  
  An explicit relational graph/tuple store; gating on schema fit/novelty/conflict; dual‑path adapter to fuse graph features with episodic traces; orchestration for replay → weight updates.

- **SMPD — Spatial Map + Replay‑to‑Policy Distillation.**  
  Builds a topological/metric *map* over contexts; plans over the graph for navigation; replays successful episodes to distill procedural “macros” that speed future control.

These designs all imply **benefits when required information is *not* in the current context** (or when structure must be accumulated across episodes).

---

## 2) What the current tests actually do

### 2.1 Dataset generators keep facts in‑prompt

- **Semantic.** `generate_semantic(..., require_memory=False)` composes the *facts* and the *question* into a single prompt. The CLI (`hippo_eval/datasets/cli.py`) does **not** expose/enable `require_memory=True`. ⇒ Baseline can solve from context alone.

- **Episodic_cross.** `generate_episodic_cross(...)` emits prompts like:  
  `"<FACT>. --- FLUSH --- <distractors>. Where did WHO go?"`  
  The “FLUSH” is just a string inside one prompt; nothing is actually flushed. ⇒ Still in‑context.

- **Spatial.** `hippo_eval/tasks/spatial/generator.py` creates independent shortest‑path tasks and occasional “macro” sequences, but there is no multi‑episode environment building or reuse of a learned map across episodes. ⇒ No test of SMPD’s core promise.

### 2.2 Harness behavior can leak memory across items

- In `hippo_eval/eval/harness.py`, during **teach** the evaluation loop both **retrieves** and **writes**. Later items in the same pass can benefit from earlier writes. That’s acceptable for *episode‑level* curricula, but it **confounds per‑item metrics** unless we explicitly design it that way.
- Stores are not cleared between items; session boundaries are per run/suite, not per QID. If different items re‑use entities with contradictory attributes, the store can accumulate conflicts unless the gate/adapter handles time/context keys explicitly.

---

## 3) Evidence from the latest run (reports/run_20250904)

Saturated or near‑saturated baselines where facts live in‑prompt, tiny or negative memory uplift, and SMPD failing on one‑shot grids.

| Suite | Baseline EM | Memory EM | ΔEM | Baseline F1 | Memory F1 |
|---|---:|---:|---:|---:|---:|
| semantic | 0.94 | 0.96 | 0.02 | 0.94 | 0.96 |
| episodic | 0.10 | 0.20 | 0.10 | 0.19 | 0.31 |
| episodic_cross | 1.00 | 0.98 | -0.02 | 1.00 | 0.99 |
| episodic_capacity | 0.00 | 0.00 | 0.00 | 0.27 | 0.30 |
| spatial | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

- **semantic:** Baseline EM≈0.94, SGC‑RSS≈0.96 → small uplift despite retrieval/gating overhead; consistent with *facts-in‑context* design.
- **episodic_cross:** Baseline EM=1.00; memory lower at 0.98 (noise), again because everything is in one prompt despite “FLUSH”.
- **episodic_capacity:** Both baselines and memory at EM≈0.00 (with non‑zero F1). Filler forces length but not true long‑term recall.
- **spatial:** Both 0.00 EM; current SMPD eval doesn’t reflect iterative map learning and policy distillation.

*(Numbers from `runs/run_20250904/.../metrics.csv`.)*

---

## 4) Conclusions

- **Concern 1 (context window as “short‑term memory”): Valid.** The suites largely test in‑context reasoning, not long‑term memory. Memory modules then act as redundant retrieval over data already visible to the model.
- **Concern 2 (memory isolation / leakage): Valid.** Teach‑time retrieval + writes allow item‑to‑item leakage. There’s no enforced reset between unrelated items; contradictory facts can accumulate.
- **Concern 3 (SMPD needs iterative learning): Valid.** Current tasks don’t require building and exploiting a persistent map or macro library; they’re single‑shot pathfinding puzzles.

---

## 5) What to change — conceptual fixes to the pipeline

### 5.1 Split *teach* and *test* across **sessions and files**

Create paired datasets per suite:
- `suite_teach.jsonl`: only **facts or trajectories**, no questions/answers.
- `suite_test.jsonl`: only **questions/queries**, no supporting facts.

Protocol:
1. Run `mode=teach` on `suite_teach.jsonl` to populate stores. **Disable retrieval during teach** (flag) so we don’t shortcut by using half‑built memory.
2. Start a **fresh process** in `mode=test` (or `mode=replay` followed by `mode=test`), loading the persisted store (`--store_dir ... --session_id ...`). Now retrieval is allowed.
3. Ensure the test prompts contain *zero* supporting facts. Add a guardrail that fails the run if any test prompt includes schema keywords present in the facts set.

### 5.2 Enforce **isolation** and make leakage measurable

- Add `--isolate=per_item|per_episode|none`. For `per_item`, clear or fork the store after each test QID. For `per_episode`, allow writes within a multi‑turn mini‑episode but reset between episodes.
- Tag all writes with a **context key** (episode_id, time). Retrieval requests must include the context key; eval should fail if an answer is justified with traces from the wrong context.
- Add **leakage checks:** inject contradictions across items; the correct behavior is either (a) pick the right context or (b) abstain, depending on suite rules.

### 5.3 Make **memory-required** semantic and episodic variants first‑class

- Expose and use `require_memory=True` in semantic generation, or add a new suite `semantic_mem`. In this mode, the prompt includes only the *question*; all facts are provided in `*_teach.jsonl`.
- Replace “FLUSH” strings with actual **session boundaries**: generate `episodic_cross_teach.jsonl` (the first fact) and `episodic_cross_test.jsonl` (the question) referencing the earlier entity. No facts in the test prompt.

### 5.4 Fix **SMPD** evaluation to require map learning and macro reuse

- Produce **progressive episodes** on the *same grid/topology* (teach) followed by evaluations that:  
  - Query novel start→goal pairs **within** the explored region (tests whether a map was built).  
  - Reward **macro reuse**: measure reduced steps/latency after replay-to-policy distillation.
- Metrics beyond EM: success rate, optimality gap, plan length vs. optimum, **learning curves** across episodes.

### 5.5 Strengthen guardrails and reporting

- Fail fast if a *memory-required* suite has baseline EM > 0.2.  
- Add metrics/plots for **retrieval justification coverage** (“% answers with at least one supporting trace from the right context”) and **gate precision/recall** on synthetic salience labels where available.
- During teach: log `retrieval.requests==0` (good); during test: require `>0` for memory suites.

---

## 6) Suggested minimal changes to this repo (high level)

1. **Datasets**
   - Add paired `spatial_teach.jsonl` and `spatial_test.jsonl` generators; wire into `scripts/build_datasets.py` and `EVAL_PLAN.md`.
   - Expose `require_memory` in the CLI and use it for `semantic_mem`.
2. **Harness**
   - Add `--no-retrieval-during-teach`; default **on** for memory suites.
   - Implement `--isolate` to clear/fork stores between items/episodes.
   - Thread a `context_key` (episode_id/time) through gates, stores, and retrieval.
3. **SMPD tasks**
   - New generator that yields multi‑episode exploration on a fixed grid with replay stages; add metrics for macro reuse and optimality gap over episodes.
4. **Guardrails**
   - Enforce “no facts in test prompts” and “baseline must be low” for memory‑required suites.
5. **Reports**
   - Add leakage and justification panels; plot EM vs episode index for SMPD.

---

## 7) What this means for your current runs

- The current numbers do **not** provide a clean test of long‑term memory benefits (semantic, episodic_cross are in‑context).  
- SMPD results are not actionable because the tasks do not probe the algorithm’s intended behavior.  
- I recommend pausing broad sweeps and implementing the pipeline changes above, then re‑running the `n=50, seed=1337` smoke to verify the expected baseline drop and memory uplift on the *memory-required* suites.

---

## 8) Acceptance checks after the rework

- On `semantic_mem` and `episodic_cross` (true cross‑session): **Baseline EM ≤ 0.2**, memory EM **≫ baseline** (target uplift ≥ +0.5 at n=50).  
- On SMPD: learning curve shows **increasing success** and **decreasing suboptimality** across episodes; macro reuse measurably improves plan length/latency on held‑out start/goal pairs.
- Teach pass logs **zero retrieval**; Test pass logs **>0 retrieval** with supporting traces from the correct context ≥ 90% of the time.

---

*Prepared from repository state in `/mnt/data/hippo-llm-memory-main` and artifacts in `runs/run_20250904/` and `reports/run_20250904/`.*
