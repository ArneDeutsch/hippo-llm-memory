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
    parser.addoption(
        "--runintegration",
        action="store_true",
        default=False,
        help="run integration tests",
    )


def pytest_collection_modifyitems(config, items):
    skip_slow = not config.getoption("--runslow")
    skip_integration = not config.getoption("--runintegration")

    for item in items:
        fspath = str(getattr(item, "fspath", ""))
        if "tests/cli/" in fspath.replace("\\", "/"):
            item.add_marker(pytest.mark.integration)

        if skip_slow and "slow" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="need --runslow to run slow tests"))
        if skip_integration and "integration" in item.keywords:
            item.add_marker(
                pytest.mark.skip(reason="need --runintegration to run integration tests")
            )
