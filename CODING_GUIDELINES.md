# Coding Guidelines

This document complements `CODING_STANDARDS.md` with conventions for documentation and inline comments.

## Comments
- Use NumPy-style triple-quoted docstrings for public modules, classes, functions, and dataclasses.
- Omit sections for returns, raises, side effects, or complexity when they match the defaults:
  - returns nothing
  - raises nothing
  - no side effects
  - complexity ``O(1)``
- Describe tensor shapes and units for times or scores.
- Keep comments short and intentional; avoid narrating code.
- Precede non-obvious logic with a rationale using `# why:`.
- Mark conditions or guarantees with `# pre:`, `# post:`, and `# invariant:`.
- Prefer logger calls for rationale; if comments are needed, prefix with `# log:`.
- Limit line length to 100 characters and avoid trailing spaces.
