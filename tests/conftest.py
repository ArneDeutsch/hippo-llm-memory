"""Pytest configuration for path setup and marker handling."""

import os
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("RUN_ID", "test")


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run tests marked as slow"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
