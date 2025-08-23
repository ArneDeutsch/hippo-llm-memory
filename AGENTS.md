On session start read: README.md, DESIGN.md, PROJECT_PLAN.md, EVAL_PLAN.md, research/SUMMARY.md, CODING_STANDARDS.md, pyproject.toml and this AGENTS.md.

Just before session end: Execute linter to ensure build server does not complain.

When running tests with pytest finding the output [100%] in the log signals success of the test suite, except there is at least one F between the dots, signaling one or more failing tests. Running the full test suite can take around three minutes, hence some patience is needed.
