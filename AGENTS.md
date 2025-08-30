On session start read: README.md, CODING_STANDARDS.md and pyproject.toml.

Just before session end: Execute linter to ensure build server does not complain.

When running tests with pytest finding the output [100%] in the log signals success of the test suite, except there is at least one F between the dots, signaling one or more failing tests. Running the full test suite can take around a minute. Running the slow tests is not necessary in general. The build server will run these. Codex only need to run the fast test suite. Slow tests are marked with `@pytest.mark.slow` and are skipped by default. Pass `--runslow` to include them or `-m slow --runslow` to run only the slow tests.
