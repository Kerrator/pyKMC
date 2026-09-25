"""Opt-in physical validation: slow checks of physical results against oracles.

Every test in this directory is skipped unless the environment variable
``PYKMC_RUN_VALIDATION`` is set to ``1``, so the default serial lane and CI
never run these multi-minute calculations. The tests carry the ``validation``
marker declared in ``pyproject.toml`` (the lanes run with ``--strict-markers``).

Tests hand the tables they measure to the ``validation_tables`` fixture; the
tables are printed in the terminal summary whether the tests pass or fail, so a
log of the run carries the measured numbers.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

VALIDATION_ENV = "PYKMC_RUN_VALIDATION"

_TABLES_KEY = pytest.StashKey[list[tuple[str, Path, str]]]()


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip every test under this directory unless ``PYKMC_RUN_VALIDATION=1``.

    Parameters
    ----------
    config : pytest.Config
        The session configuration (unused; required by the hook signature).
    items : list of pytest.Item
        The collected items; those under this directory receive a skip marker.

    """
    if os.environ.get(VALIDATION_ENV) == "1":
        return
    here = Path(__file__).resolve().parent
    skip = pytest.mark.skip(
        reason=f"opt-in physical validation: set {VALIDATION_ENV}=1 to run"
    )
    for item in items:
        if here in Path(str(item.path)).resolve().parents:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def validation_tables(
    request: pytest.FixtureRequest,
) -> Callable[[str, Path, str], None]:
    """Return a recorder for tables to print in the terminal summary.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The requesting context; its ``config.stash`` holds the tables.

    Returns
    -------
    Callable[[str, Path, str], None]
        ``record(title, path, text)``: ``path`` is where the test wrote the
        table and ``text`` its Markdown content.

    """
    tables = request.config.stash.setdefault(_TABLES_KEY, [])

    def record(title: str, path: Path, text: str) -> None:
        tables.append((title, path, text))

    return record


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    """Print every recorded table after the test summary.

    Parameters
    ----------
    terminalreporter : pytest.TerminalReporter
        The reporter that writes the summary section.
    exitstatus : int
        The session exit status (unused).
    config : pytest.Config
        The session configuration holding the recorded tables.

    """
    tables = config.stash.get(_TABLES_KEY, [])
    if not tables:
        return
    terminalreporter.section("validation tables")
    for title, path, text in tables:
        terminalreporter.write_line(f"{title} (written to {path})")
        terminalreporter.write_line("")
        for line in text.splitlines():
            terminalreporter.write_line(line)
        terminalreporter.write_line("")
