#!/usr/bin/env python3
"""Run memory evaluation with optional gate toggles.

This is a thin wrapper around :mod:`scripts.eval_model` that exposes
explicit flags to force memory gates ON or OFF for episodic, relational,
and spatial modules.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _gate_arg(flag: str, value: str) -> str | None:
    """Return Hydra override for ``flag`` given ``value``.

    Parameters
    ----------
    flag:
        Config key such as ``episodic.use_gate`` or
        ``relational.gate.enabled``.
    value:
        Either ``"on"`` or ``"off"``; ``"auto"`` yields ``None``.
    """

    if value == "auto":
        return None
    enabled = "true" if value == "on" else "false"
    return f"+{flag}={enabled}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--mode", default="teach")
    parser.add_argument("--model", default="models/tiny-gpt2")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--episodic-gate", choices=["on", "off", "auto"], default="auto")
    parser.add_argument("--relational-gate", choices=["on", "off", "auto"], default="auto")
    parser.add_argument("--spatial-gate", choices=["on", "off", "auto"], default="auto")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="extra Hydra overrides")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "eval_model.py"),
        f"suite={args.suite}",
        f"preset={args.preset}",
        f"n={args.n}",
        f"seed={args.seed}",
        f"outdir={args.outdir}",
        f"mode={args.mode}",
        f"model={args.model}",
    ]
    if args.dry_run:
        cmd.append("dry_run=true")

    epi_override = _gate_arg("memory.episodic.gate.enabled", args.episodic_gate)
    if epi_override:
        cmd.append(epi_override)
    rel_override = _gate_arg("memory.relational.gate.enabled", args.relational_gate)
    if rel_override:
        cmd.append(rel_override)
    spa_override = _gate_arg("memory.spatial.gate.enabled", args.spatial_gate)
    if spa_override:
        cmd.append(spa_override)

    if args.overrides:
        cmd.extend(args.overrides)

    subprocess.run(cmd, check=True)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
