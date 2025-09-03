"""Run identifier validation helpers."""

import re

SLUG_RE = re.compile(r"^[A-Za-z0-9_-]{3,64}$")


def validate_run_id(value: str) -> str:
    """Validate a run identifier slug.

    Parameters
    ----------
    value: str
        Candidate RUN_ID value.

    Returns
    -------
    str
        The original value if it is a valid slug.

    Raises
    ------
    TypeError
        If ``value`` is not a string.
    ValueError
        If ``value`` contains invalid characters or length.
    """
    if not isinstance(value, str):
        raise TypeError("RUN_ID must be a string")
    if not SLUG_RE.fullmatch(value):
        raise ValueError("Invalid RUN_ID. Use 3–64 chars from [A-Za-z0-9_-].")
    return value
