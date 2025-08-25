# Eval Review — Qwen2.5‑1.5B‑Instruct (episodic, baselines/core)

_Generated: 2025-08-25 08:20:49 _

## TL;DR

- Current run is a **core baseline** with no memory modules or replay.
- Metrics show **EM=0.000**, **F1=0.065** over **n=50** items — predictions appear unusable due to prompt echo.

- The harness **does not apply chat templates** and **decodes the full input+output**, so `pred` often equals the prompt.
- `time_ms_per_100` in `metrics.json` is **buggy**; reported value does not match a correct computation.

## What I checked

- `meta.json`, `metrics.json`, and `metrics.csv` from your run.
- `scripts/eval_model.py` to see how tokenization, generation, and metrics are computed.
- `configs/eval/baselines/core.yaml` and defaults under `configs/eval/default.yaml`.

## Findings

### 1) Predictions are echoing the prompt (broken decoding)

`eval_model.py` decodes the entire sequence returned by `generate()` instead of slicing off the input tokens. This causes `pred` to include the full prompt, which torpedoes EM/F1.

**Examples (truncated):**
- **Prompt:** `Carol saw at the Park on Thursday. Where was Carol?`  
  **Answer:** `the Park`  
  **Pred:** `Carol saw at the Park on Thursday. Where was Carol?  A. New York City  B. California  C. Florida  D. The beach  E. Chica…`
- **Prompt:** `Bob helped at the Cafe on Thursday. When was Bob at the Cafe?`  
  **Answer:** `Thursday`  
  **Pred:** `Bob helped at the Cafe on Thursday. When was Bob at the Cafe?  "Took him a while to get there, but once he got there he …`

**Fix (core):**

Slice off the input length before decoding:
```python
# BEFORE
enc = tokenizer(item.prompt, return_tensors="pt").to(model.device)
out = model.generate(**enc, max_new_tokens=max_new_tokens)
pred = tokenizer.decode(out[0], skip_special_tokens=True)

# AFTER
inputs = tokenizer(item.prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
gen = out[:, inputs["input_ids"].shape[-1]:]
pred = tokenizer.decode(gen[0], skip_special_tokens=True).strip()
```

### 2) Chat/instruction templates are **not** applied

The harness calls `AutoTokenizer(...); tokenizer(prompt)` directly. For chat‑tuned models (e.g., Qwen‑Instruct), you must apply the model’s chat template, otherwise models tend to echo the prompt or return malformed outputs.

**Fix (chat‑aware encoding):**

```python
def encode_prompt(tokenizer, model, prompt: str, device):
    # Prefer chat templates when available
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
    else:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    return {"input_ids": input_ids.to(device)}
```
Then use `inputs = encode_prompt(tokenizer, model, item.prompt, model.device)`.

### 3) `time_ms_per_100` is miscomputed

- **Reported:** 73429.3 ms/100  
- **Correct:**  5790.6 ms/100 (computed from per‑item latencies and token count)

**Fix (compute block):**

```python
compute = {
    "tokens": total_tokens,
    "time_ms_per_100": (100.0 * total_time) / max(1, total_tokens),
    "rss_mb": _rss_mb(),
    "latency_ms_mean": sum(latencies) / max(1, len(latencies)),
}
```

### 4) Coverage vs. EVAL_PLAN

- This run uses `baselines/core` and `replay_cycles=0`. Retrieval/gates are all zero. So it **does not yet verify** HEI‑NW / SGC‑RSS / SMPD; it only provides a (currently broken) baseline.

- To satisfy the plan you’ll need:
  1) Valid **core** baseline (fixed decoding/template),
  2) **RAG** and **long‑context** baselines, and
  3) **memory presets** (`memory/hei_nw`, `memory/sgc_rss`, `memory/smpd`) with **pre/post replay** and gates enabled.

### 5) Seeds: keep 3 for *final* validation, 1 for speed during iteration

- In the current harness, generation is greedy (no sampling), so the **seed only affects dataset sampling**, not decoding.
- Using a single seed speeds runs up by 66% but reduces coverage and makes metrics noisier for small `n`.
- **Recommendation:** use `seeds: [1337]` for smoke tests and dev loops. For milestone gates and reports, run all three seeds and macro‑average.

## Actionable checklist (Codex‑ready)

1. **Fix decoding** to slice generated tokens (see §1).  
2. **Introduce `encode_prompt(...)`** that uses `apply_chat_template` when available (see §2).  
3. **Correct `time_ms_per_100`** (see §3).  
4. Add `compute.generated_tokens` and `compute.input_tokens` for clarity.  
5. Re‑run `baselines/core` on episodic/semantic/spatial with `n ∈ {50,200}` and seed=1337 for a quick smoke check.  
6. If healthy, run **full matrix** (`n ∈ {50,200,1000}`, seeds `{1337,2025,4242}`) to lock in baselines.  
7. Run memory presets with `replay.cycles ∈ {1,3}` and ensure gates/retrieval metrics populate.  
8. Add an assertion that EM>0 or F1>0.2 for episodic@50 as a sanity check to fail fast on future regressions.

## Appendix — Run metadata & quick stats

- Items: **50**  
- EM: **0.000**, F1: **0.065**  
- Tokens: **634**  
- Mean latency/item: **734.2 ms**  
- `time_ms_per_100` (reported): **73429.3**  
- `time_ms_per_100` (recomputed): **5790.6**
