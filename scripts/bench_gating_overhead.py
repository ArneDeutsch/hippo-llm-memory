"""Benchmark write-gate overhead on synthetic traces.

Summary
-------
Measure episodic ingest throughput with the write gate enabled vs disabled
using small synthetic traces.  Prints ops/sec for both cases and the
percentage overhead introduced by the gate.  Exits with code ``1`` when the
overhead exceeds 10%%.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from hippo_mem.episodic.gating import WriteGate
from hippo_mem.episodic.store import EpisodicStore

N_ITEMS = 10_000
DIM = 64
RNG = np.random.default_rng(0)


def _ingest_without_gate() -> float:
    """Insert ``N_ITEMS`` traces without gating and return throughput."""
    store = EpisodicStore(DIM)
    queries = RNG.random((N_ITEMS, DIM), dtype="float32")
    start = time.perf_counter()
    for q in queries:
        store.write(q, "bench")
    end = time.perf_counter()
    return N_ITEMS / (end - start)


def _ingest_with_gate() -> float:
    """Insert ``N_ITEMS`` traces with the write gate and return throughput."""
    store = EpisodicStore(DIM)
    gate = WriteGate(tau=1.0)
    probs = RNG.random(N_ITEMS)
    queries = RNG.random((N_ITEMS, DIM), dtype="float32")
    keys = np.zeros((0, DIM), dtype="float32")
    start = time.perf_counter()
    for p, q in zip(probs, queries):
        gate(p, q, keys, pin=True)
        store.write(q, "bench")
    end = time.perf_counter()
    return N_ITEMS / (end - start)


def main() -> None:
    """Run the benchmark and enforce the overhead threshold."""
    off = _ingest_without_gate()
    on = _ingest_with_gate()
    overhead = (off - on) / off * 100.0
    print(f"without_gate_ops_per_sec={off:.2f}")
    print(f"with_gate_ops_per_sec={on:.2f}")
    print(f"overhead_percent={overhead:.2f}")
    if overhead > 10.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
