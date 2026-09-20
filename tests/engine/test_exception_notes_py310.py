"""A promoted restore/cleanup warning survives on Python 3.10 too.

``BaseException.add_note`` exists from 3.11. On the declared 3.10 floor a
secondary failure (a restore that fails after the initiating failure, or an
endpoint cleanup failure) used to be discarded when a warning policy promoted
the ``RuntimeWarning``: the original error was raised without any record. The
engine now routes every note through ``_attach_note``, which falls back to the
``__notes__`` list attribute when ``add_note`` is unavailable. The module seam
``_ADD_NOTE`` emulates 3.10 on any interpreter.
"""

from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from pykmc.engine import lammps as lammps_module
from pykmc.engine.lammps import FullSystem, LammpsEngine

from tests.engine.test_endpoint_constraints import (
    CELL,
    PBC,
    TYPES,
    CommandEndpoint,
    config,
    event,
    payload,
)


def _notes(exc):
    return "\n".join(getattr(exc, "__notes__", []) or [])


class _Initiating(RuntimeError):
    """The failure a public operation raised first."""


@pytest.mark.parametrize("py310", [True, False], ids=["py310-seam", "native"])
def test_failed_restore_after_failure_is_recorded_without_add_note(monkeypatch, py310):
    if py310:
        monkeypatch.setattr(lammps_module, "_ADD_NOTE", None)
    engine = LammpsEngine(SimpleNamespace(), comm=None)

    def failing_restore(positions=None):
        raise RuntimeError("secondary restore failure")

    monkeypatch.setattr(engine, "ensure_full_system", failing_restore)
    original = _Initiating("original partn_search failure")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        engine._restore_after_failure(np.zeros((2, 3)), original, "partn_search")
    record = _notes(original)
    assert "restoring the full system afterwards failed" in record
    assert "secondary restore failure" in record
    assert "original partn_search failure" in record


class _CleanupFailingEndpoint(CommandEndpoint):
    """Primary minimisation failure, then the cleanup ``run 0`` fails too."""

    def __init__(self, positions):
        super().__init__(positions, failure="minimize")
        self.failed = False

    def command(self, command):
        if command.startswith("minimize"):
            self.failed = True
        if self.failed and command.startswith("run"):
            raise RuntimeError("secondary cleanup failure")
        super().command(command)


@pytest.mark.parametrize("py310", [True, False], ids=["py310-seam", "native"])
def test_endpoint_cleanup_failure_is_recorded_on_the_primary_error(monkeypatch, py310):
    if py310:
        monkeypatch.setattr(lammps_module, "_ADD_NOTE", None)
    first, _, _ = event()
    rows = np.array([3, 0, 1])
    positions = first[rows].copy()
    engine = LammpsEngine(
        SimpleNamespace(min_style="cg", minimize="1e-8 1e-8 100 1000"), comm=None
    )
    endpoint = _CleanupFailingEndpoint(positions)
    engine.lmp = endpoint
    engine._is_orthorhombic = True
    engine.full_system = FullSystem(
        tuple(np.array(TYPES)[rows]), ("Fe", "Ni"), (56.0, 60.0), CELL.copy(), PBC
    )
    proposed = positions.copy()
    proposed[1, 0] += 0.1
    with pytest.raises(RuntimeError, match="primary injected minimization") as raised:
        engine.minimize_with_results(
            positions=proposed,
            config=config(),
            types=tuple(np.array(TYPES)[rows]),
            constraints=payload().crop(tuple(rows)),
        )
    record = _notes(raised.value)
    assert "Endpoint cleanup failures" in record
    assert "secondary cleanup failure" in record
    assert engine._cleared_since_init
