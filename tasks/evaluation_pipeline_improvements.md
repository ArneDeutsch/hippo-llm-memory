## 1) Fix episodic retrieval hit-rate inflation

**Why**: Telemetry counts “cue injection” as a hit, forcing `hit_rate_at_k=1.0` even with no recalled traces.&#x20;
**Files**: `hippo_mem/episodic/retrieval.py`, `tests/test_episodic_retrieval.py`
**Steps**

* In `_apply_hopfield`, when no traces are recalled, **do not** increment hits. Keep placeholder tokens for the model but set `hits=0` for telemetry.
* Ensure telemetry fields (`requests`, `total_k`, `hits`, `hit_rate_at_k`) reflect **actual recalled traces** only.
* Add unit test: empty store ⇒ `hit_rate_at_k == 0.0` while `tokens_returned == k`.
  **Acceptance**
* Test passes; episodic\@50 no longer logs `hit_rate_at_k=1.0` with an empty/irrelevant store.&#x20;

---

## 2) Make consolidation uplift gate configurable + CI-based

**Why**: Hard gate `EM uplift < +0.20` is too aggressive at n=50; blocks writing `post_*` metrics.&#x20;
**Files**: `scripts/test_consolidation.py`, `docs/EVAL_PROTOCOL.md`
**Steps**

* Add CLI flags: `--uplift-mode=[fixed,ci]`, `--min-uplift=FLOAT`, `--alpha=FLOAT` (default `fixed`, `min=0.05`, `alpha=0.05`).
* If `ci`, compute 95% CI of `(post - pre)` over seeds (bootstrap or t-interval) and require uplift > `max(0, CI_upper_bound_of_0)`.
* Print both `pre`, `post`, `delta` and gate decision.
  **Acceptance**
* At n=50, seed=1337, `--min-uplift=0.05` runs to completion and writes metrics; CI mode works when multiple seeds are present.&#x20;

---

## 3) Ensure replay writes `post_*` and `delta_*` metrics

**Why**: Current artifacts only contain `pre_*` fields; cannot measure consolidation.&#x20;
**Files**: `scripts/eval_model.py`, `hippo_mem/eval/harness.py` (or equivalent), `tests/test_metrics_post_delta.py`
**Steps**

* In replay (`mode=replay` + `persist=true`), compute and persist `post_em`, `post_em_raw`, etc., and `delta_* = post_* - pre_*`.
* Write into the same per-run `metrics.json` where `pre_*` lives.
* Add a tiny integration test that runs a mini suite and asserts keys `post_*` and `delta_*` exist.
  **Acceptance**
* Metrics files include `post_*` and `delta_*` fields after replay.&#x20;

---

## 4) Report uplift tables with CIs and plots

**Why**: Reports currently surface `pre_*` only; need clear visualization of uplift.&#x20;
**Files**: `scripts/report.py`, `reports/templates/*`, `tests/test_report_uplift.py`
**Steps**

* Add per-suite table: `pre`, `post`, `Δ`, and 95% CI over seeds when available.
* Add uplift plot per suite (distribution across seeds).
* Flag suites as **saturated** when `pre_em(norm) >= 0.98`.
  **Acceptance**
* Generated `reports/$RUN_ID/*` includes new tables/plots; CI columns appear only when ≥2 seeds.&#x20;

---

## 5) Standardize RUN\_ID prelude + stable session IDs

**Why**: Stores/reports must line up across baseline → memory → replay.&#x20;
**Files**: `scripts/_env.sh` (new), all `scripts/*.py|*.sh`
**Steps**

* Create `scripts/_env.sh` with the agreed prelude (`RUN_ID` primary; `DATE="$RUN_ID"` for back-compat).
* Source it from all driver scripts.
* Derive `session_id` deterministically (e.g., `hei_$RUN_ID`) and pass it to replay commands.
  **Acceptance**
* One `RUN_ID` drives the whole pipeline; path & session\_id are stable and found by `replay`.&#x20;

---

## 6) Add dataset difficulty profiles to avoid saturation

**Why**: Some suites cap at EM(norm)≈1.0, hiding differences.&#x20;
**Files**: `conf/datasets/*.yaml` (new), `scripts/make_datasets.py`, `docs/EVAL_PROTOCOL.md`
**Steps**

* Introduce `base`/`hard` profiles (entities count, span lengths, distractor rate, capacity sizes).
* Wire flags into dataset generation and EVAL protocol (e.g., use `hard` for `episodic_cross`/`capacity`).
  **Acceptance**
* `make datasets` creates both profiles; “hard” raises task difficulty and reduces saturation in pilot runs.&#x20;

---

## 7) Gate ON/OFF ablation + reporter section

**Why**: Gate telemetry shows zero activity; need explicit ablations and logs.&#x20;
**Files**: `scripts/run_memory.py`, gate modules under `hippo_mem/*`, `scripts/report.py`, `tests/test_gates_ablation.py`, fixtures under `tests/fixtures/gates/*`
**Steps**

* Add CLI flags to force gates ON/OFF per memory type (episodic/relational/spatial).
* Add minimal gate-stress fixtures so inserts/accepts > 0 when ON.
* Reporter: add “Gate ON vs OFF” table (store size, accepts, ΔEM).
  **Acceptance**
* Ablation runs populate the new table with non-zero gate stats when ON.&#x20;

---

## 8) Use EM(raw) as primary metric for `semantic`; keep both

**Why**: EM(norm)=1.0 while EM(raw)=0.56–0.74 can mask differences.&#x20;
**Files**: `scripts/report.py`, `docs/EVAL_PROTOCOL.md`
**Steps**

* For the `semantic` suite, sort/rank by EM(raw) in summary tables; show EM(norm) adjacent.
* Document metric choice per suite.
  **Acceptance**
* Semantic summary orders by EM(raw); docs explain metric selection.&#x20;

---

## 9) Tighten telemetry sanity checks

**Why**: Prevent future regressions (e.g., impossible hit-rates).&#x20;
**Files**: `hippo_mem/common/telemetry.py` (new), `tests/test_telemetry_sanity.py`
**Steps**

* Add runtime check helpers (e.g., `hits <= total_k`, `hit_rate_at_k == hits/total_k` within epsilon).
* Optional `--strict-telemetry` flag to fail fast during dev/CI.
  **Acceptance**
* With strict mode on and synthetic bad input, tests fail; normal runs pass.&#x20;

---

## 10) Reporter: Retrieval section reflects fixed semantics

**Why**: After Task 1, retrieval tables must align with new hit semantics.&#x20;
**Files**: `scripts/report.py`, `reports/templates/*`
**Steps**

* Update retrieval summary text to clarify that “hit” = **actual recall**, not cue injection.
* Show per-memory (`episodic/relational/spatial`) request/hit tables.
  **Acceptance**
* Reports describe the corrected definition; numbers match telemetry post-fix.&#x20;

---

## 11) Smoke E2E script for n=50, seed=1337 (pre→replay→report)

**Why**: Fast validation path for contributors; ensures `post_*` and Δ get produced.&#x20;
**Files**: `scripts/smoke_n50.sh` (new)
**Steps**

* One-shot script that runs: baselines → memory (persist) → replay (cycles=3) → report; all with stable `RUN_ID`.
* Exit non-zero if `post_*` or `delta_*` missing.
  **Acceptance**
* Running the script produces reports with uplift tables and exits 0.&#x20;

---

## 12) Store/session layout helper + validator

**Why**: Reduce path mistakes during replay.
**Files**: `scripts/store_paths.py` (new), `scripts/validate_store.py` (new), call-sites in `scripts/eval_model.py`
**Steps**

* Centralize logic to compute `store_dir`, `session_id` from `RUN_ID`.
* Validator checks expected files exist before replay; prints remediation hints.
  **Acceptance**
* Validator flags mis-paths; replay succeeds when paths are correct.

---

## 13) Documentation updates (protocol & plan)

**Why**: Keep contributors aligned and remove ambiguity.&#x20;
**Files**: `docs/EVAL_PROTOCOL.md`, `docs/EVAL_PLAN.md`, `docs/REPORTS.md` (new)
**Steps**

* Update protocol with: stable `RUN_ID`, replay step, new CLI flags, gate ablations, metric choices per suite, difficulty profiles, and the smoke script.
* Add a short “Troubleshooting Step 9” section.
  **Acceptance**
* Docs include exact commands and explain *why* each step exists; reviewers can follow them verbatim to reproduce uplift.&#x20;

---

## 14) (Optional) Bootstrap CI over seeds

**Why**: Automate CI-based uplift gating when multiple seeds present.&#x20;
**Files**: `scripts/test_consolidation.py`, `scripts/report.py`, `docs/EVAL_PLAN.md`
**Steps**

* If seed list length ≥ 2, automatically switch to `--uplift-mode=ci` and print CI in both test output and report.
  **Acceptance**
* Multi-seed runs show CI-based pass/fail and report the CI bands.&#x20;
