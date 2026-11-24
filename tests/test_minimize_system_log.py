"""The start minimization's log line names the group engine's rank count.

The group's rank count sets LAMMPS's summation order, so it changes the last
bits of every whole-system minimization (see ``group_size`` in config.py).
"""

from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pytest

from pykmc.kmc import KMC


class _Manager:
    """Group-engine stand-in for the two calls ``minimize_system`` makes."""

    def group_minimize_with_results(self, **kwargs: Any) -> tuple[np.ndarray, float]:
        """Return fixed positions and energy."""
        return np.zeros((2, 3)), -1.0

    def group_get_potential_energy(self) -> float:
        """Return a fixed energy."""
        return -1.0


class _Loggers:
    """Record the log lines."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def info(self, logger_name: str, msg: str) -> None:
        """Record one line."""
        self.lines.append(msg)


def _fake_kmc(group_size: Optional[int], restart_file: Optional[str]) -> Any:
    """Build the attributes ``minimize_system`` reads."""
    return SimpleNamespace(
        config=SimpleNamespace(control=SimpleNamespace(restart_file=restart_file)),
        loggers=_Loggers(),
        manager=_Manager(),
        system=SimpleNamespace(types=np.ones(2), update_positions=lambda p: None),
        group_size=group_size,
    )


@pytest.mark.parametrize(
    "group_size, restart_file, expected",
    [
        (7, None, ":=> Minimizing the system (group engine, 7 MPI ranks)"),
        (1, "restart.npz", ":=> Computing energies (group engine, 1 MPI rank)"),
        (None, None, ":=> Minimizing the system"),
    ],
)
def test_minimize_log_names_group_ranks(
    group_size: Optional[int], restart_file: Optional[str], expected: str
) -> None:
    """The line names the rank count when run.py has set it, and only then."""
    kmc = _fake_kmc(group_size, restart_file)
    KMC.minimize_system(kmc)
    assert kmc.loggers.lines == [expected]
