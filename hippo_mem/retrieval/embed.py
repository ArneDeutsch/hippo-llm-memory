"""Deterministic text embeddings for the shared retrieval fabric.

Summary
-------
Provides a lightweight embedding function used by all stores and the replay
pipeline's 50/30/20 mix.  Maintenance jobs may prune embeddings via the
associated ANN index.

See Also
--------
hippo_mem.retrieval.faiss_index
    ANN index consuming these embeddings.
"""

from __future__ import annotations

from typing import List


def embed_text(text: str, dim: int = 16) -> List[float]:
    """Return a deterministic placeholder embedding.

    Summary
    -------
    Maps characters to floats so retrieval code paths can run without a model.

    Parameters
    ----------
    text:
        Input text to embed.
    dim:
        Length of the resulting embedding vector. Defaults to 16 to keep vectors
        small and CPU friendly.

    Returns
    -------
    list[float]
        Embedding vector of shape ``(dim,)``.

    Raises
    ------
    None

    Side Effects
    ------------
    None

    Complexity
    ----------
    ``O(len(text))``.

    Examples
    --------
    >>> embed_text("hi", dim=4)
    [0.41, 0.36, 0.0, 0.0]

    See Also
    --------
    hippo_mem.retrieval.faiss_index.FaissIndex
    """

    # Simple character level encoding: normalise byte values to [0, 1).
    raw = [b / 255.0 for b in text.encode("utf-8")]

    # Pad or truncate to the desired dimension.
    if len(raw) < dim:
        raw.extend([0.0] * (dim - len(raw)))
    else:
        raw = raw[:dim]

    return raw
