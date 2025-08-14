"""Text embedding utilities."""

from __future__ import annotations

from typing import List


def embed_text(text: str, dim: int = 16) -> List[float]:
    """Return a deterministic placeholder embedding for ``text``.

    The real system will swap this out for an actual model‑based embedding
    generator.  For now we simply map characters to float values so that the
    retrieval code paths can be exercised in tests.

    Parameters
    ----------
    text:
        Input text to embed.
    dim:
        Length of the resulting embedding vector.  Default is 16 to keep the
        vectors small and CPU friendly.
    """

    # Simple character level encoding: normalise byte values to [0, 1).
    raw = [b / 255.0 for b in text.encode("utf-8")]

    # Pad or truncate to the desired dimension.
    if len(raw) < dim:
        raw.extend([0.0] * (dim - len(raw)))
    else:
        raw = raw[:dim]

    return raw
