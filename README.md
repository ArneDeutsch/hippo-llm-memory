# hippo-llm-memory

Hippocampus-inspired memory modules for small language models.

hippo-llm-memory explores whether hippocampal algorithms can extend lightweight
open-source LLMs. It prototypes three modules:

- **HEI‑NW – Hippocampal Episodic Index with Neuromodulated Writes:** sparse
  episodic store that writes surprising or novel events once, recalls them from
  partial cues and replays salient traces for consolidation.
- **SGC‑RSS – Schema‑Guided Consolidation with a Relational Semantic Store:**
  parses text into (head, relation, tail) tuples, routes schema-fitting facts
  into a knowledge graph and fuses relational retrieval with episodic recall.
- **SMPD – Spatial Map with Replay-to-Policy Distillation:** builds a
  place/route graph, plans paths and distills successful trajectories into
  reusable macros.

Built for a single 12 GB GPU. Training runs locally while CI uses a CPU
container for tests and linting.

##

HEI-NW is an “episodic memory” add-on for an LLM. Its job is to capture single experiences the moment they happen, store them durably, and bring them back from a partial cue. It does this in three moves. First, it turns the model’s current hidden state into a very sparse key so that similar episodes don’t collide, echoing the hippocampus’ pattern separation. Second, it uses an associative lookup to complete a memory from that sparse key, so a fragmentary prompt can recall a full episode. Third, it decides whether to “write” the new episode using a salience score based on things like predictive surprise or an explicit reward flag, which imitates neuromodulatory gating. At inference time the LLM cross-attends to the retrieved episode through a small adapter layer, and offline a replay worker distills high-value episodes into the base model’s weights so the knowledge becomes semantic over time. In practice you hook it at the KV-cache to form queries, add one cross-attention adapter to read episodic traces, and run a background consolidation job that trains on prioritized replays. The end result is true one-shot episodic storage with partial-cue recall and a path to long-term consolidation, while keeping runtime controlled with efficient attention in the core model.

SGC-RSS is the “semantic memory” complement. It extracts tuples like who-did-what-where-when and aligns them to schemas so that routine knowledge is structured immediately while oddball cases stay episodic for a while. When a new episode fits an existing schema, SGC-RSS promotes its facts into a graph store right away and schedules only light replay; when it does not, the facts remain in the episodic buffer and get richer replay before any generalization. At query time the LLM reads both the graph and the most relevant episodes through separate adapters and merges them, which stabilizes multi-hop reasoning without losing case-specific detail. Integration is straightforward: you add a relational adapter that cross-attends over graph embeddings, keep a consolidation worker that interleaves schema facts with episodic replays, and let the same scheduler order reactivations to minimize interference. This deliberately mirrors complementary learning systems: fast hippocampal-like episodes feeding a slower cortical-like semantic store.

SMPD supplies spatial and procedural memory. It builds a topological map of “places” or contexts and the edges between them, and from successful trajectories it distills reusable macros, so the system can plan routes and reuse habits. Given a goal, it plans over the map and returns a skeletal route which the LLM then fills in with language or tool calls; success cases are replayed to a small policy head to become macros you can invoke later. The LLM integrates this module as a tool: it can write new places and edges, read the neighborhood around the current context, or ask for a plan, and it can cross-attend to the returned plan or macro suggestions just like any other retrieval. The spatial nodes link back to episodes and schema facts, so you can “recall by place,” and a light gate avoids map explosion by merging near-duplicates and aggregating repeats rather than endlessly adding edges.

Together the three modules form one memory system with clean responsibilities and shared plumbing. HEI-NW is the front door: it decides what to write now, guarantees you can recall it later from a hint, and feeds a prioritized replay queue. SGC-RSS listens to those episodes and, when a pattern is familiar, lifts it into a stable semantic graph so the model answers consistently even without the original context. SMPD listens as well, turning sequences that have a spatial or stepwise structure into maps and macros, and then uses those structures to propose plans that the LLM can execute or elaborate. A single replay scheduler, inspired by interference-aware hippocampal replay, orders what gets revisited offline so you consolidate the right things without forgetting others; the same provenance and decay rules keep the episodic store lean, the graph clean, and the map compact. At inference time the core Transformer stays fast thanks to efficient attention and grouped K/V tricks, while the adapters selectively pull in just a few tokens from the episodic store, just the subgraph you need, and just the relevant slice of the map, so quality rises without blowing the context window. Conceptually, HEI-NW gives you one-trial memory and cue-based recall, SGC-RSS gives you structured knowledge and stability, SMPD gives you planning and habits, and the consolidation loop ties them into a single system that learns quickly when it must and generalizes safely when it can.

## Repository layout

```
hippo-llm-memory/
├─ README.md
├─ AGENTS.md              # workflow instructions for coding agents
├─ CONTRIBUTING.md        # contribution process
├─ CODING_STANDARDS.md    # style guide
├─ DESIGN.md              # architecture and algorithms
├─ EVAL_PLAN.md           # evaluation protocol
├─ PROJECT_PLAN.md        # milestones and work packages
├─ LICENSE
├─ Makefile
├─ pyproject.toml
├─ codex-env/             # environment setup and dependencies
├─ configs/               # Hydra configs for models, training, eval, memory
├─ data/                  # synthetic benchmark datasets
├─ docs/                  # API surfaces, trace specs, inventories, experiments
├─ hippo_mem/             # core Python package with memory adapters and stores
├─ models/                # tiny model fixtures for tests
├─ research/              # literature reviews and experiment synthesis
├─ review/                # progress reviews and planning notes
├─ reports/               # aggregated evaluation outputs
├─ runs/                  # run artifacts and metrics
├─ scripts/               # dataset generation, training, evaluation utilities
├─ tests/                 # unit tests
└─ .github/               # CI workflows and templates
```

## Persistence layout

Persistent stores live under a common base directory:

```
runs/$RUN_ID/stores/
  hei_nw/<SID>/{episodic.jsonl, relational.jsonl, spatial.jsonl}
  sgc_rss/<SID>/kg.jsonl
  smpd/<SID>/spatial.jsonl
```

Pass this base path via `--store_dir`; wrappers create the algorithm subfolder and
nested `--session_id` directory.

## Quickstart (local, single 12 GB GPU)

1. Create and activate a Conda env (Python 3.10):

   ```bash
   conda create -n hippo python=3.10 -y
   conda activate hippo
   ```

2. Install dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r codex-env/requirements.txt
   ```

3. Smoke-test the tools:

   ```bash
   python scripts/eval_model.py --help
   ```

4. Choose a small base model and pass it via `model=<HF repo>` (e.g., `meta-llama/Llama-3.2-3B-Instruct`, `microsoft/Phi-3.5-mini-instruct`, or `Qwen/Qwen2.5-1.5B-Instruct`).

5. Dry‑run the trainer (no GPU training in CI):

   ```bash
   python scripts/train_lora.py model=meta-llama/Llama-3.2-3B-Instruct train/qlora train.dry_run=true
   ```

6. Train with QLoRA locally:

   ```bash
   python scripts/train_lora.py \
     model=meta-llama/Llama-3.2-3B-Instruct train/qlora \
     train.micro_batch=1 train.grad_accum=8 train.seq_len=1024
   ```

7. Evaluate:

   ```bash
   python scripts/eval_model.py suite=episodic preset=memory/hei_nw n=50 seed=1337
   ```

### Recommended base models (updated)

| Family  | HF repo                                  | Params | Context (input/output) | License | Notes |
|---------|-------------------------------------------|--------|-------------------------|---------|-------|
| Qwen    | Qwen/Qwen2.5-1.5B-Instruct               | 1.5B   | 32k in (128k capable) / 8k gen | Apache-2.0 | Latest small Qwen, strong math/coding |
| Phi     | microsoft/Phi-3.5-mini-instruct          | ~3.8B  | 128k / 8k               | MIT     | Strong small SLM; good long‑ctx |
| Llama   | meta-llama/Llama-3.2-3B-Instruct         | 3.2B   | up to 128k / 8k         | Llama 3.2 Community | Stable 3B baseline |
| Gemma   | google/gemma-3-1b-it                     | 1B     | 32k / 8k                | Gemma   | Optional extra‑small; text‑only |

**Note:** With Option B, presets no longer hardcode a model; pass `model=...` on the CLI.

### Quick sanity run

Verify the harness and scoring on a tiny slice before full runs:

```bash
RUN_ID=test_$(date +%s); SID=test
# Baseline with pre metrics
python scripts/eval_model.py suite=semantic preset=baselines/core run_id=$RUN_ID n=5 seed=1337 compute.pre_metrics=true
python -m hippo_eval.baselines --run-id $RUN_ID

# Memory variant with replay and persistence
python scripts/eval_model.py suite=semantic preset=memory/sgc_rss run_id=$RUN_ID n=5 seed=1337 \
  replay_cycles=1 persist=true store_dir=runs/$RUN_ID/stores session_id=$SID

# Expect non-zero pre_em in metrics.json and store_meta.source == "replay".
```

## Baselines

Presets live under `configs/eval/baselines/`:

- `core` – base model only.
- `rag` – nearest-neighbour retrieval with concatenated context.
- `longctx` – longest feasible context window without retrieval.
- `span_short` – chat templates on with a short-span decoding profile for exact-match metrics.
>
> <span style="color:red;font-weight:bold">MUST:</span> Use a `RUN_ID` slug consistently across all commands. Valid slugs match `^[A-Za-z0-9_-]{3,64}$`.

Before any memory run, generate baseline metrics:

```bash
python -m hippo_eval.baselines --run-id "$RUN_ID"
```

**Quickstart**

```bash
# 1) Teach: write experiences to stores (persist across runs)
RUN_ID=my_experiment; SID=seed1337
python scripts/eval_model.py preset=memory/hei_nw task=episodic n=200 seed=1337 \
  mode=teach persist=true store_dir=runs/$RUN_ID/stores session_id=$SID \
  model=Qwen/Qwen2.5-1.5B-Instruct outdir=runs/$RUN_ID/memory/teach

# 2) Pre-consolidation baseline (memory OFF)
python scripts/test_consolidation.py --phase pre   --suite episodic --n 50 --seed 1337 \
  --model Qwen/Qwen2.5-1.5B-Instruct   --outdir runs/$RUN_ID/consolidation/pre

# 3) Consolidate via replay → LoRA
python scripts/replay_consolidate.py   --store_dir runs/$RUN_ID/stores --session_id $SID \
  --config configs/consolidation/lora_small.yaml   --outdir runs/$RUN_ID/consolidation/lora

# 4) Post-consolidation test (memory OFF)
python scripts/test_consolidation.py --phase post   --suite episodic --n 50 --seed 1337 \
  --model Qwen/Qwen2.5-1.5B-Instruct   --adapter runs/$RUN_ID/consolidation/lora \
  --pre_dir runs/$RUN_ID/consolidation/pre   --outdir runs/$RUN_ID/consolidation/post
```

## Cross-session runs

Memory stores can persist across processes. `scripts/eval_model.py` accepts overrides
`store_dir=…`, `session_id=…`, `persist=true`, and `mode={teach,replay,test}` so a first run can **teach** facts,
an optional second run can **replay**, and a fresh process can **test** delayed recall. See
`MILESTONE_9_5_PLAN.md` for the protocol.

Pass either `--store_dir=runs/$RUN_ID/stores` (recommended; algo inferred) or
`--store_dir=runs/$RUN_ID/stores/hei_nw` for an explicit algorithm subfolder. Preflight resolves both
forms. To run multiple replay passes, use `replay_cycles=N` (or `replay.cycles=N`). For convenience,
`scripts/eval_cli.py` translates legacy `--mode`-style flags into these overrides.

## Key artifacts

- [research/experiment-synthesis.md](research/experiment-synthesis.md) –
  cross-domain mapping from hippocampal mechanisms to the HEI‑NW, SGC‑RSS and
  SMPD algorithms.
- [research/SUMMARY.md](research/SUMMARY.md) – high-level overview of the three
  memory hypotheses.
- [DESIGN.md](DESIGN.md) – detailed architecture, data structures and
  algorithms.
- [PROJECT_PLAN.md](PROJECT_PLAN.md) – milestones and work packages for
  development.
- [EVAL_PLAN.md](EVAL_PLAN.md) – datasets, baselines, metrics and ablations for
  validation.
- [MILESTONE_9_PLAN.md](MILESTONE_9_PLAN.md) and
  [MILESTONE_9_5_PLAN.md](MILESTONE_9_5_PLAN.md) – current milestone scopes.
- [docs/TRACE_SPEC.md](docs/TRACE_SPEC.md) – schema for memory traces exchanged
  with adapters.
- [docs/api_surface.md](docs/api_surface.md) – current public APIs.

## How to read reports

Reports live under `reports/<RUN_ID>/index.md` with per‑suite summaries. Rows carry
⚠️ warnings when invariants are violated:

- Baselines must show `retrieval.*.requests == 0` and `store.size == 0`.
- Memory presets with gates enabled should report `gate.*.attempts > 0`.
- `pre_em_norm ≥ 0.98` with a matching baseline `< 0.20` signals saturation.

See [EVAL_PROTOCOL.md](EVAL_PROTOCOL.md#telemetry-invariants) for the full list
and troubleshooting tips.

## Suggested shell aliases

```bash
alias M_LLAMA3S="meta-llama/Llama-3.2-3B-Instruct"
alias M_QWEN25S="Qwen/Qwen2.5-1.5B-Instruct"
alias M_PHI35S="microsoft/Phi-3.5-mini-instruct"
alias M_GEMMA3S="google/gemma-3-1b-it"
```
Use them like: `model=$M_QWEN25S`.

