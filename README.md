# hippo-llm-memory

**Hippocampus‑inspired memory modules for small LLMs (HEI‑NW, SGC‑RSS, SMPD) using Transformers + PEFT/QLoRA**

This monorepo contains three experiments that add hippocampal mechanisms to open LLMs via lightweight adapters and external memory:

- **HEI‑NW** — *Hippocampal Episodic Index with Neuromodulated Writes*: sparse episodic store, content‑addressable recall, surprise/novelty‑gated one‑shot writes, prioritized replay → consolidation.
- **SGC‑RSS** — *Schema‑Guided Consolidation with a Relational Semantic Store*: tuple/graph store, schema‑fit routing, dual‑path retrieval (semantic subgraphs + episodes).
- **SMPD** — *Spatial Map + Replay‑to‑Policy Distillation*: place/route graph for navigation & multi‑step procedures; replay distills macros.

Built for **one 12 GB GPU** on Ubuntu. Training is done locally; the **web Codex** (ChatGPT) integrates with GitHub to scaffold code, run tests in a CPU container, and open PRs.

## Repository layout

```
hippo-llm-memory/
├─ README.md
├─ AGENTS.md
├─ CONTRIBUTING.md
├─ CODING_STANDARDS.md
├─ DESIGN.md                         # high-level design links/notes
├─ EVAL_PLAN.md                      # plan to evaluate our results
├─ MILESTONE_8_PLAN.md
├─ LICENSE
├─ codex-env/
│  ├─ requirements.txt
│  └─ setup.sh
├─ configs/
│  ├─ model/{llama32-3b.yaml,phi3-mini.yaml,qwen2-1_5b.yaml}
│  ├─ train/qlora.yaml
│  ├─ memory/{episodic.yaml,relational.yaml,spatial.yaml}
│  └─ eval/*.yaml
├─ data/                              # baseline datasets
├─ docs/
│  ├─ api_surface.md
│  ├─ baselines.md
│  └─ inventory.md
├─ experiments/
│  ├─ hei_nw/{RUN.md, run.yaml, tasks.md}
│  ├─ sgc_rss/{RUN.md, run.yaml, tasks.md}
│  └─ smpd/{RUN.md, run.yaml, tasks.md}
├─ hippo_mem/
│  ├─ __init__.py
│  ├─ adapters/lora.py
│  ├─ retrieval/{embed.py,faiss_index.py}
│  ├─ episodic/{store.py,gating.py,replay.py}
│  ├─ relational/{tuples.py,kg.py,adapter.py}
│  └─ spatial/{map.py,macros.py}
├─ models/                            # test model fixtures
│  └─ tiny-gpt2/
├─ scripts/
│  ├─ train_lora.py                  # TRL + PEFT LoRA/QLoRA trainer (single GPU)
│  ├─ eval_bench.py                  # episodic/semantic/spatial evaluation
│  ├─ export_adapter.py              # save/merge LoRA adapters
│  └─ build_datasets.py              # synthetic tasks from Pass 4
├─ tests/
│  ├─ test_episodic.py
│  ├─ test_relational.py
│  ├─ test_spatial.py
│  └─ test_training.py
├─ research/                         # Research results these experiments are based on
│  ├─ experiment-synthesis.md
│  ├─ hippocampal-memory-storage.md
│  ├─ large-language-models.md
│  └─ validation.md
├─ pyproject.toml
├─ Makefile
└─ .github/
   ├─ workflows/ci.yml
   ├─ ISSUE_TEMPLATE/codex_task.md
   └─ PULL_REQUEST_TEMPLATE.md
```

## Quickstart (local, single 12 GB GPU)

1. Create and activate a Conda env (Python 3.10):
   ```bash
   conda create -n hippo python=3.10 -y
   conda activate hippo
   ```
2. Install dependencies (Hydra, HF stack, tooling):
   ```bash
   pip install --upgrade pip
   pip install -r codex-env/requirements.txt
   ```
3. Verify setup (optional):
   ```bash
   python scripts/eval_bench.py --help
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
   python scripts/eval_bench.py --config-path configs --config-name eval/bench.yaml
   ```

## Models known to fit in 12 GB with QLoRA (guidance)

- Llama‑3.2‑3B (text‑only), Qwen2‑1.5B, Phi‑3‑Mini (\~3.8B). Use 4‑bit NF4, gradient checkpointing, sequence length ≤1024.
