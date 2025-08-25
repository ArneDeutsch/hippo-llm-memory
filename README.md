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
├─ docs/                  # API surfaces, trace specs, inventories
├─ experiments/           # run configs and task lists for each module
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

## Baselines

Presets live under `configs/eval/baselines/`:

- `core` – base model only.
- `rag` – nearest-neighbour retrieval with concatenated context.
- `longctx` – longest feasible context window without retrieval.
- `span_short` – chat templates on with a short-span decoding profile for exact-match metrics.

### Quickstart

```bash
DATE=$(date +%Y%m%d_%H%M)
python scripts/eval_model.py preset=baselines/span_short task=episodic n=50 seed=1337 \
  model=Qwen/Qwen2.5-1.5B-Instruct \
  outdir=runs/$DATE/baselines/span_short/Qwen2.5-1.5B
```

The `span_short` preset keeps chat templates on while forcing short span-only answers to reduce refusal-style responses in span-extraction tasks.

## Cross-session runs

Memory stores can persist across processes. `scripts/eval_model.py` accepts `--store_dir`,
`--session_id`, `--persist`, and `--mode={teach,replay,test}` so a first run can **teach** facts,
an optional second run can **replay**, and a fresh process can **test** delayed recall. See
`MILESTONE_9_5_PLAN.md` for the protocol.

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

## Suggested shell aliases

```bash
alias M_LLAMA3S="meta-llama/Llama-3.2-3B-Instruct"
alias M_QWEN25S="Qwen/Qwen2.5-1.5B-Instruct"
alias M_PHI35S="microsoft/Phi-3.5-mini-instruct"
alias M_GEMMA3S="google/gemma-3-1b-it"
```
Use them like: `model=$M_QWEN25S`.

