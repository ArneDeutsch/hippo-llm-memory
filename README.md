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
├─ CONTRIBUTING.md
├─ CODING_STANDARDS.md
├─ DESIGN.md                         # high-level design links/notes
├─ LICENSE
├─ codex-env/
│  ├─ requirements.txt               # deps Codex installs in its container
│  └─ setup.sh                       # paste into Codex → Environment → Setup script
├─ pyproject.toml                    # black/ruff/pytest settings
├─ Makefile                          # lint/test shortcuts for Codex and CI
├─ scripts/
│  ├─ train_lora.py                  # TRL + PEFT LoRA/QLoRA trainer (single GPU)
│  ├─ eval_bench.py                  # episodic/semantic/spatial evaluation
│  ├─ export_adapter.py              # save/merge LoRA adapters
│  └─ build_datasets.py              # synthetic tasks from Pass 4
├─ configs/
│  ├─ model/{llama32-3b.yaml,phi3-mini.yaml,qwen2-1_5b.yaml}
│  ├─ train/qlora.yaml
│  ├─ memory/{episodic.yaml,relational.yaml,spatial.yaml}
│  └─ eval/*.yaml
├─ hippo_mem/
│  ├─ __init__.py
│  ├─ adapters/lora.py
│  ├─ retrieval/{embed.py,faiss_index.py}
│  ├─ episodic/{store.py,gating.py,replay.py}
│  ├─ relational/{tuples.py,kg.py,adapter.py}
│  └─ spatial/{map.py,macros.py}
├─ experiments/
│  ├─ hei_nw/{RUN.md, run.yaml, tasks.md}
│  ├─ sgc_rss/{RUN.md, run.yaml, tasks.md}
│  └─ smpd/{RUN.md, run.yaml, tasks.md}
├─ tests/
│  ├─ test_episodic.py
│  ├─ test_relational.py
│  ├─ test_spatial.py
│  └─ test_training.py
└─ .github/
   ├─ workflows/ci.yml
   ├─ PULL_REQUEST_TEMPLATE.md
   └─ ISSUE_TEMPLATE/codex_task.md
```

## Quickstart (local, single 12 GB GPU)

1. **Python** 3.10+ and CUDA‑matched **PyTorch**.
2. Install deps:
   ```bash
   pip install -r codex-env/requirements.txt
   ```
3. Choose a small base model (e.g., `llama32-3b`, `phi3-mini`, or `qwen2-1_5b`).
4. Dry‑run the trainer (no GPU training in CI):
   ```bash
   python scripts/train_lora.py model=llama32-3b train/qlora train.dry_run=true
   ```
5. Train with QLoRA locally:
   ```bash
   python scripts/train_lora.py \
     model=llama32-3b train/qlora \
     train.micro_batch=1 train.grad_accum=8 train.seq_len=1024
   ```
6. Evaluate:
   ```bash
   python scripts/eval_bench.py --config-path configs --config-name eval/bench.yaml
   ```

## Using Codex (web, via ChatGPT)

- **Connect GitHub** in ChatGPT → Settings → Connected apps → GitHub (choose this repo).
- In Codex → **Environments**, set **Setup script** to contents of `codex-env/setup.sh`.
- Open a **Code** task with one of the prompts in each experiment’s `RUN.md`. Codex will:
  1. clone the repo in its container; 2) run the setup script; 3) run `make lint` & `make test`; 4) propose diffs; 5) open a PR. You review and merge.

## Models known to fit in 12 GB with QLoRA (guidance)

- Llama‑3.2‑3B (text‑only), Qwen2‑1.5B, Phi‑3‑Mini (\~3.8B). Use 4‑bit NF4, gradient checkpointing, sequence length ≤1024.
