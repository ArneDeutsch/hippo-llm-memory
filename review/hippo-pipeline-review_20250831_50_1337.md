# Pipeline Review — Run 20250831_50_1337

_Generated 2025-08-31 09:06 UTC_

## Executive Summary
- The current reports **do not provide trustworthy telemetry** for retrieval or gating. We observe 50/50 retrieval hits even in baselines and in ablations that should disable retrieval, and zero gate attempts throughout.
- Memory stores exist for each algorithm, but their contents look **synthetic/stub-like** (e.g., identical sizes across algorithms, placeholder embeddings, and 'dummy' or toy provenance). This makes most EM improvements **uninformative** about a real implementation.
- Several suites show **saturated EM (norm ≈ 1.0)** for memory presets while all baselines remain at **0.0**. With the above telemetry issues and stub stores, this suggests the tasks are currently **not discriminative** or the evaluation is **leaking** answers via pre-populated stores.
- Immediate action: **fix telemetry plumbing**, **isolate/clear stores for baselines and ablations**, and **instrument gate writes**. Without these, we cannot conclude whether the algorithms are working.

## What I Reviewed
- Reports: `reports/20250831_50_1337/index.md` and per-suite `summary.md` files.
- Run artifacts: `runs/20250831_50_1337/` including baselines, memory, and ablation metrics and the `stores/` directory.

## Key Findings
### 1) Metrics Overview (per suite + preset)
The table below shows core metrics extracted from every `metrics.json` under this run. Flags indicate suspicious conditions detected automatically.

| group | preset | suite | pre_em_raw | pre_em_norm | pre_f1 | retrieval_hits | store_size | flag |
|---|---|---|---|---|---|---|---|---|
| longctx_no_retrieval | semantic/50_1337 | 50_1337 | 0.000 | 0.000 | 0.000 | 50 | 5 | gate_never_attempted |
| sgc_rss_no_gate | semantic/50_1337 | 50_1337 | 0.520 | 1.000 | 0.520 | 50 | 155 | gate_never_attempted |
| core | episodic/50_1337 | 50_1337 | 0.000 | 0.000 | 0.000 | 50 | 5 | gate_never_attempted |
| core | episodic_capacity/50_1337 | 50_1337 | – | – | – | 50 | 5 | gate_never_attempted |
| core | episodic_cross/50_1337 | 50_1337 | – | – | – | 50 | 5 | gate_never_attempted |
| core | episodic_multi/50_1337 | 50_1337 | – | – | – | 50 | 5 | gate_never_attempted |
| core | semantic/50_1337 | 50_1337 | 0.000 | 0.000 | 0.000 | 50 | 5 | gate_never_attempted |
| core | spatial/50_1337 | 50_1337 | 0.000 | 0.000 | 0.000 | 50 | 5 | gate_never_attempted |
| longctx | episodic/50_1337 | 50_1337 | 0.000 | 0.000 | 0.000 | 50 | 5 | gate_never_attempted |
| longctx | episodic_capacity/50_1337 | 50_1337 | – | – | – | 50 | 5 | gate_never_attempted |
| longctx | episodic_cross/50_1337 | 50_1337 | – | – | – | 50 | 5 | gate_never_attempted |
| longctx | episodic_multi/50_1337 | 50_1337 | – | – | – | 50 | 5 | gate_never_attempted |
| longctx | semantic/50_1337 | 50_1337 | 0.000 | 0.000 | 0.000 | 50 | 5 | gate_never_attempted |
| longctx | spatial/50_1337 | 50_1337 | 0.000 | 0.000 | 0.000 | 50 | 5 | gate_never_attempted |
| rag | episodic/50_1337 | 50_1337 | 0.000 | 0.000 | 0.000 | 50 | 5 | gate_never_attempted |
| rag | episodic_capacity/50_1337 | 50_1337 | – | – | – | 50 | 5 | gate_never_attempted |
| rag | episodic_cross/50_1337 | 50_1337 | – | – | – | 50 | 5 | gate_never_attempted |
| rag | episodic_multi/50_1337 | 50_1337 | – | – | – | 50 | 5 | gate_never_attempted |
| rag | semantic/50_1337 | 50_1337 | 0.000 | 0.000 | 0.000 | 50 | 5 | gate_never_attempted |
| rag | spatial/50_1337 | 50_1337 | 0.000 | 0.000 | 0.000 | 50 | 5 | gate_never_attempted |
| span_short | episodic/50_1337 | 50_1337 | 0.000 | 0.000 | 0.000 | 50 | 5 | gate_never_attempted |
| span_short | episodic_capacity/50_1337 | 50_1337 | – | – | – | 50 | 5 | gate_never_attempted |
| span_short | episodic_cross/50_1337 | 50_1337 | – | – | – | 50 | 5 | gate_never_attempted |
| span_short | episodic_multi/50_1337 | 50_1337 | – | – | – | 50 | 5 | gate_never_attempted |
| span_short | semantic/50_1337 | 50_1337 | 0.000 | 0.000 | 0.000 | 50 | 5 | gate_never_attempted |
| span_short | spatial/50_1337 | 50_1337 | 0.000 | 0.000 | 0.000 | 50 | 5 | gate_never_attempted |
| hei_nw | episodic/50_1337 | 50_1337 | 0.120 | 0.620 | 0.332 | 50 | 155 | gate_never_attempted |
| hei_nw | episodic_capacity/50_1337 | 50_1337 | – | – | – | 50 | 155 | gate_never_attempted |
| hei_nw | episodic_cross/50_1337 | 50_1337 | – | – | – | 50 | 155 | gate_never_attempted |
| hei_nw | episodic_multi/50_1337 | 50_1337 | – | – | – | 50 | 155 | gate_never_attempted |
| sgc_rss | semantic/50_1337 | 50_1337 | 0.660 | 1.000 | 0.660 | 50 | 155 | gate_never_attempted |
| smpd | spatial/50_1337 | 50_1337 | 0.000 | 0.020 | 0.000 | 50 | 155 | gate_never_attempted |
| ? | ? | ? | 0.100 | 0.560 | 0.344 | 50 | 155 | gate_never_attempted |
| ? | ? | ? | 0.100 | 0.660 | 0.266 | 50 | 155 | gate_never_attempted |
| ? | ? | ? | 0.120 | 0.640 | 0.344 | 50 | 155 | gate_never_attempted |
| ? | ? | ? | 0.480 | 1.000 | 0.480 | 50 | 155 | gate_never_attempted |
| ? | ? | ? | 0.600 | 1.000 | 0.600 | 50 | 155 | gate_never_attempted |
| ? | ? | ? | 0.540 | 1.000 | 0.540 | 50 | 155 | gate_never_attempted |
| ? | ? | ? | 0.000 | 0.020 | 0.000 | 50 | 155 | gate_never_attempted |
| ? | ? | ? | 0.000 | 0.000 | 0.000 | 50 | 155 | gate_never_attempted |
| ? | ? | ? | 0.000 | 0.020 | 0.000 | 50 | 155 | gate_never_attempted |

**Interpretation:**
- Baselines and ablate/longctx_no_retrieval **should not** have any retrieval hits or nonzero store sizes. However, `retrieval_hits` is **always 50** and `store_size` is **nonzero** for baselines in this run. This strongly suggests telemetry is either hardcoded or incorrectly shared across presets.
- Many memory presets show **`pre_em_norm ≈ 1.0`**, including semantic (`memory/sgc_rss`) and multiple episodic suites (`episodic_capacity`, `episodic_cross`, `episodic_multi`). With baselines at 0.0 and telemetry issues, these saturations are not credible evidence of algorithmic success.

### 2) What the algorithms actually “memorized” (from `stores/`)
- **Store root:** `hippo-llm-memory-main/runs/20250831_50_1337/stores/hei_nw/hei_20250831_50_1337/`
  - episodic entries: **151**; sample key norms (first 10): [0.999999982885729, 0.0, 0.999999982885729, 0.999999982885729, 1.000000067179426, 0.999999982885729, 1.000000067179426, 1.000000067179426, 0.999999982885729, 0.999999982885729]
  - sample provenance labels (first 10, unique): ['dummy', 'the Cafe', 'the Library', 'the Mall', 'the Park']
  - relational lines: **3** (nodes=2, edges=1)
  - spatial lines: **5**
- **Store root:** `hippo-llm-memory-main/runs/20250831_50_1337/stores/sgc_rss/sgc_20250831_50_1337/`
  - episodic entries: **151**; sample key norms (first 10): [0.999999982885729, 0.0, 0.999999982885729, 0.999999982885729, 1.000000067179426, 0.999999982885729, 1.000000067179426, 1.000000067179426, 0.999999982885729, 0.999999982885729]
  - sample provenance labels (first 10, unique): ['Berlin', 'London', 'Paris', 'Rome', 'dummy']
  - relational lines: **3** (nodes=2, edges=1)
  - spatial lines: **5**
- **Store root:** `hippo-llm-memory-main/runs/20250831_50_1337/stores/smpd/smpd_20250831_50_1337/`
  - episodic entries: **151**; sample key norms (first 10): [0.999999982885729, 0.0, 0.999999982885729, 0.999999982885729, 1.000000067179426, 0.999999982885729, 1.000000067179426, 1.000000067179426, 0.999999982885729, 0.999999982885729]
  - sample provenance labels (first 10, unique): ['4', '7', 'DLLLD', 'LD', '[2, 0]', '[4, 2]', 'dummy']
  - relational lines: **3** (nodes=2, edges=1)
  - spatial lines: **5**

**Interpretation:**
- All three algorithms have **identical store sizes** (151 episodic, 3 relational lines, 5 spatial lines). The first episodic record has provenance `'dummy'` and many keys have **unit norms or zero vectors**, indicative of **placeholder embeddings** rather than learned traces.
- For relational stores, counts suggest that only **1 edge** is considered in `per_memory` despite 2 nodes + 1 edge recorded; this mismatch should be clarified in store accounting.

### 3) Gate Telemetry is inert
- Across memory runs, gate counters (attempts, accepts, inserted, aggregated, routed, blocked) are **all zero**. Either gating was never exercised or the counters aren’t wired. This prevents any learning about **when/how** memories are written.

### 4) Compute telemetry in reports appears placeholder
- The high-level `index.md` shows **`time_ms_per_100 = 5.000`** for baselines, which is a classic placeholder default, while memory runs report realistic figures. All compute metrics should be collected **uniformly** to avoid bias in cost/perf comparisons.

## Does the evaluation answer our research questions?
- **Do the algorithms work?** Not verifiable yet. Saturated EM with stub-like stores and broken telemetry cannot demonstrate real capability.
- **Do they deliver the intended functionality?** Unknown. With no gate attempts or meaningful write telemetry, we cannot confirm episodic capture, semantic consolidation, or spatial mapping beyond toy placeholders.
- **Is the pipeline designed to reveal this?** Not yet. The pipeline needs hard **invariants/tests** to prevent leakage, ensure per-preset isolation, and validate that telemetry is internally consistent.

## Concrete Issues and How to Fix Them
1. **Retrieval telemetry fires in baselines and no-retrieval ablation.**
   - **Fix:** In the harness, when `preset ∈ baselines` or when `replay.retrieval_k==0`, ensure the retrieval layer is **not constructed** and that telemetry counters are **disabled or zeroed**. Add a unit test: if `group in {baselines, ablate/longctx_no_retrieval}` then `retrieval_hits==0` must hold.
2. **Store isolation & nonzero store size in baselines.**
   - **Fix:** Ensure each run writes to a unique, empty store path. For baselines, mount a **null store** (no read/write) and assert the store directory is absent or empty after the run. Add a post-run check that `store_size==0` for baselines.
3. **Gate counters never increment.**
   - **Fix:** Wire gate instrumentation at the decision boundary (attempt → accept/deny). Log: attempts, accepts, inserted traces, aggregation events, and blocks. Add regression tests that a memory-enabled preset produces **nonzero attempts** on a small sample.
4. **Stub/placeholder store contents.**
   - **Fix:** Replace placeholder embeddings and provenance with real traces (token spans, entity slots, state sketches). Validate key norms distribution (e.g., mean ~O(1/√d), no spikes at exactly 0 or 1) and provenance coverage.
5. **Saturation & task discriminability.**
   - **Fix:** Calibrate datasets so baselines achieve **nonzero** EM/f1 (e.g., 15–40%) while memory algorithms can plausibly add uplift. Avoid pre-populating answers in stores; write only from **observations during the run**.
6. **Compute telemetry parity.**
   - **Fix:** Collect `input_tokens`, `generated_tokens`, `total_tokens`, `latency`, `rss_mb` for **all** presets. Disallow placeholders in final reports.
7. **Cross-run contamination.**
   - **Fix:** Namescope stores by `{RUN_ID}/{preset}/{suite}/{seed}` and verify no carry-over. Add a checksum of store files to the report for traceability.
8. **Confidence & variability.**
   - **Fix:** Enable multiple seeds (≥3) and sizes. Compute per-preset **CI bands** and **effect sizes** (Hedges’ g) to avoid over-interpreting single-seed results.

## Quick Win Validations to Add Now
- **Invariant A:** For any run with `group == baselines`, `retrieval_hits==0` and `store_size==0`.
- **Invariant B:** For `ablate/longctx_no_retrieval`, `retrieval_hits==0`.
- **Invariant C:** If `pre_em_norm≥0.98` for a memory preset, require that baselines achieve `pre_em_norm≥0.20` on the same suite—or flag as **saturated benchmark**.
- **Invariant D:** Gate attempts must be **>0** whenever a memory preset is used, else raise a report warning.

## Early Take on the Three Algorithms (based on this run only)
- **HEI‑NW (episodic):** Reported uplift on `episodic` (norm≈0.62) but **saturates** on `episodic_capacity` and `episodic_cross`. Given the telemetry flaws and stub store, this likely reflects **benchmark leakage** rather than genuine encoding/recall dynamics.
- **SGC‑RSS (semantic):** `pre_em_norm=1.0` with and without gating (ablation) ⇒ the **gate is not affecting** outcomes; either the gate path is bypassed or the task is trivialized by the store.
- **SMPD (spatial):** Very low EM (norm≈0.02). With inert gating and placeholder spatial store, this result is **not actionable**; we need real spatial traces and an evaluation that probes path planning or coordinate recall.

## Recommendation
Before attempting Step 9 again, run a **telemetry integrity pass**:
1. Clear stores, run `baselines/core` for one suite, and verify retrieval hits==0 and store_size==0.
2. Enable a memory preset on the same suite; verify gate attempts>0 and store_size increases by ≈n.
3. Confirm that `metrics.json` values differ between presets in ways that match expectations (e.g., cost ↑ with retrieval).

---
_Artifacts reviewed: `runs/20250831_50_1337/` and `reports/20250831_50_1337/` from the uploaded ZIP._
