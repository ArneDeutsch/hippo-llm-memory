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
├─ AGENTS.md
├─ CONTRIBUTING.md
├─ CODING_STANDARDS.md
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

4. Choose a small base model (e.g., `llama32-3b`, `phi3-mini`, or `qwen2-1_5b`).

5. Dry‑run the trainer (no GPU training in CI):

   ```bash
   python scripts/train_lora.py model=llama32-3b train/qlora train.dry_run=true
   ```

6. Train with QLoRA locally:

   ```bash
   python scripts/train_lora.py \
     model=llama32-3b train/qlora \
     train.micro_batch=1 train.grad_accum=8 train.seq_len=1024
   ```

7. Evaluate:

   ```bash
   python scripts/eval_model.py suite=episodic preset=memory/hei_nw n=50 seed=1337
   ```

## Models known to fit in 12 GB with QLoRA (guidance)

- Llama‑3.2‑3B (text‑only), Qwen2‑1.5B, Phi‑3‑Mini (~3.8B). Use 4‑bit NF4,
  gradient checkpointing, sequence length ≤1024.

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
- [docs/TRACE_SPEC.md](docs/TRACE_SPEC.md) – schema for memory traces exchanged
  with adapters.
- [docs/api_surface.md](docs/api_surface.md) – current public APIs.

