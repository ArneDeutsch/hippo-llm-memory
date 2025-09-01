"""Plot retrieval statistics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict


def plot_retrieval(retrieval: Dict[str, dict[str, float]], out_dir: Path) -> None:
    """Render a bar chart of retrieval requests and hits."""
    try:  # pragma: no cover - matplotlib optional
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib missing
        logging.getLogger(__name__).warning("matplotlib unavailable: %s", exc)
        return

    mems = sorted(retrieval)
    reqs = [retrieval[m].get("requests", 0) for m in mems]
    hits = [retrieval[m].get("hits_at_k", retrieval[m].get("hits", 0)) for m in mems]
    x = range(len(mems))
    width = 0.4
    plt.figure()
    plt.bar([i - width / 2 for i in x], reqs, width=width, label="requests")
    plt.bar([i + width / 2 for i in x], hits, width=width, label="hits_at_k")
    plt.xticks(list(x), mems)
    plt.ylabel("count")
    plt.title("retrieval")
    plt.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / "retrieval.png")
    plt.close()
