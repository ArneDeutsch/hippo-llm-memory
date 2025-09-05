# TASKS — M0: Deprecation & Cleanup

## Context Recap
Legacy datasets and code evaluate in-context reasoning rather than memory. We must **remove** them from `main` to avoid dual paths and confusion. A `git tag` preserves reproducibility.


> **Path hints (adjust to your repo):**
> - Datasets: `data/{semantic,semantic_hard,episodic*,spatial*}`
> - Generators: `hippo_eval/tasks/generators.py`, `hippo_eval/tasks/spatial/generator.py`
> - Harness & eval: `hippo_eval/eval/harness.py`, `hippo_eval/harness/*`, `scripts/eval_cli.py`
> - Stores: `hippo_mem/{episodic,relational,spatial}/*`, `hippo_eval/stores/*`
> - Reporting: `hippo_eval/reporting/*`, `reports/*`
> - Configs: `configs/datasets/*`, `configs/presets/*`
> - Docs: `EVAL_PLAN.md`, `EVAL_PROTOCOL.md`, `DESIGN.md`, `MILESTONE_9_PLAN.md`
> - Artifacts (your run): `runs/run_20250904/`, `reports/run_20250904/`


## Goal
Archive legacy assets, remove them from `main`, update configs/docs/tests, and add a CI guard to prevent reintroduction.

## Tasks

### T0.1 — Archive legacy snapshot
**Commands**
```bash
git tag legacy-datasets-v1
git push origin legacy-datasets-v1
git branch archive/legacy-datasets-v1
git push origin archive/legacy-datasets-v1
```
**Accept**
- Tag and branch visible on remote.

### T0.2 — Remove legacy datasets
**Commands**
```bash
git rm -r data/semantic data/semantic_hard
git rm -r data/episodic data/episodic_hard data/episodic_cross data/episodic_cross_hard \
         data/episodic_multi data/episodic_multi_hard data/episodic_capacity data/episodic_capacity_hard
git rm -r data/spatial data/spatial_hard
```
**Accept**
- `git grep -nE 'data/(semantic|episodic|spatial)'` returns **no hits** (except inside `DEPRECATIONS.md`).

### T0.3 — Remove legacy generators & dataset tooling
**Commands**
```bash
git rm hippo_eval/tasks/generators.py hippo_eval/tasks/spatial/generator.py
git rm scripts/datasets_cli.py scripts/audit_datasets.py
```
**Accept**
- `git grep -nE '(generators.py|datasets_cli.py|audit_datasets.py)'` returns **no hits**.

### T0.4 — Prune configs and presets
**Actions**
- Edit or delete any entries under `configs/datasets/*.yaml` referencing removed suites.
- Ensure only `semantic_closed_book`, `episodic_closed_book`, `spatial_explore` remain.

**Accept**
- `git grep -nE '(semantic_hard|episodic_|spatial_hard|\bsemantic\b|\bepisodic\b|\bspatial\b)' configs/datasets/*.yaml` → **no legacy hits**.

### T0.5 — Remove/replace tests
**Actions**
- Delete tests that depend on legacy datasets.
- Add placeholder tests (to be replaced in M1–M3).

**Accept**
- `pytest -q` passes locally.

### T0.6 — Docs cleanup + DEPRECATIONS.md
**Actions**
- Update `EVAL_PLAN.md`, `EVAL_PROTOCOL.md`, `DESIGN.md`, `MILESTONE_9_PLAN.md` to remove legacy references.
- Add `DEPRECATIONS.md` enumerating removals and the archival tag.

**Accept**
- `markdownlint` passes.
- `git grep` shows no legacy references in docs except `DEPRECATIONS.md`.

### T0.7 — CI grep‑guard
**Add a CI step** that fails if any forbidden patterns appear:
```
data/semantic
data/semantic_hard
data/episodic
data/episodic_hard
data/episodic_cross
data/episodic_cross_hard
data/episodic_multi
data/episodic_multi_hard
data/episodic_capacity
data/episodic_capacity_hard
data/spatial
data/spatial_hard
hippo_eval/tasks/generators.py
hippo_eval/tasks/spatial/generator.py
scripts/datasets_cli.py
scripts/audit_datasets.py
```
**Accept**
- CI fails on PRs that reintroduce these paths.

## Out of Scope
Do not delete `runs/*` or `reports/*`; optionally move them to `archive/` but preserve for context.
