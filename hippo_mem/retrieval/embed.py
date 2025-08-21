"""Text embedding utilities.

Summary
-------
Provides a deterministic, CPU-friendly placeholder embedding used by the
retrieval fabric during tests. Real deployments swap this for a model-based
encoder.

See Also
--------
hippo_mem.retrieval.faiss_index
"""

from __future__ import annotations

from typing import List


# TODO: appears unused; consider referencing or removing.
def embed_text(text: str, dim: int = 16) -> List[float]:
    """Return a deterministic placeholder embedding.

    Summary
    -------
    Maps characters to floats so retrieval code paths can be tested without a
    heavy model.

    Parameters
    ----------
    text : str
        Input text to embed.
    dim : int, optional
        Length of the resulting vector ``(dim,)``; default is ``16``.

    Returns
    -------
    list of float
        Embedding vector of length ``dim``.
    Complexity
    ----------
    ``O(dim)``.

    Examples
    --------
    >>> embed_text("hi", dim=4)
    [0.41, 0.36, 0.0, 0.0]

    See Also
    --------
    hippo_mem.retrieval.faiss_index.FaissIndex
    """

    # why: deterministic placeholder for tests
    raw = [b / 255.0 for b in text.encode("utf-8")]

    # why: keep vectors small for CPU-friendly ops
    if len(raw) < dim:
        raw.extend([0.0] * (dim - len(raw)))
    else:
        raw = raw[:dim]

    return raw
