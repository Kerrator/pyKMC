"""Plumbing tests for the HTST engine operations on the plugin architecture.

Covers the three layers that make ``get_forces``, ``dynamical_matrix_eskm`` and
``compute_event_prefactors`` manager-reachable without MPI: (1) the Worker op
registry built from the engine (extension methods are collected through
``Engine.__dir__`` / ``Engine.__getattr__``), (2) the Manager's auto-generated
submit wrappers (via a fake session resolved by the real worker thread), and
(3) the per-event fan-out of :class:`EventPrefactorPool`. The real engine
methods themselves are covered serially in ``test_engine_prefactors.py`` /
``test_profiling.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("lammps")
pytest.importorskip("mpi4py")

from pykmc.engine.base import Engine  # noqa: E402
from pykmc.htst.lammps_extension import HtstLammpsExtension  # noqa: E402
from pykmc.htst.pool import EventPrefactorPool  # noqa: E402
from pykmc.manager.manager import Manager  # noqa: E402
from pykmc.manager.worker import build_registry  # noqa: E402

_FORCES_SENTINEL = object()
_HESSIAN_SENTINEL = object()


class _StubEngine(Engine):
    """Engine with every abstract method stubbed (no LAMMPS instance)."""

    name = "htst_plumbing_stub"

    def __init__(self) -> None:
        super().__init__()

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def initialize_parameters(self) -> None:
        pass

    def initialize_system(
        self, types: Any, positions: Any, cell: Any, pbc: Any
    ) -> None:
        pass

    def initialize_potential(self) -> None:
        pass

    def get_positions(self) -> np.ndarray | None:
        return None

    def set_positions(self, positions: np.ndarray) -> None:
        pass

    def get_total_energy(self, positions: Any = None, recompute: bool = True) -> None:
        return None

    def get_potential_energy(
        self, positions: Any = None, recompute: bool = True
    ) -> None:
        return None

    def minimize(self, positions: Any = None) -> None:
        pass

    def minimize_with_results(self, positions: Any = None) -> None:
        return None


class _FakeSession:
    """Record every ``call``; answer with a canned result per op name."""

    def __init__(self, results: dict[str, object]) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._results = results

    def call(self, op_name: str, **kwargs: Any) -> object:
        self.calls.append((op_name, dict(kwargs)))
        return self._results[op_name]

    def shutdown(self) -> None:
        pass


def test_engine_registers_htst_ops() -> None:
    """The Worker registry built from the engine dispatches the three HTST ops."""
    engine = _StubEngine()
    HtstLammpsExtension(engine)
    registry = build_registry(engine)
    assert "get_forces" in registry
    assert "dynamical_matrix_eskm" in registry
    assert "compute_event_prefactors" in registry
    assert "_premin_surroundings" not in registry  # private helpers are not ops


def test_manager_get_forces_roundtrips() -> None:
    """Manager.get_forces submits the op with its kwargs and resolves the result."""
    fake = _FakeSession({"get_forces": _FORCES_SENTINEL})
    manager = Manager(local_sessions=[fake])
    manager.start()
    try:
        future = manager.get_forces(positions="POS")
        assert future.result(timeout=5) is _FORCES_SENTINEL
        assert fake.calls == [("get_forces", {"positions": "POS"})]
    finally:
        manager.shutdown()


def test_manager_dynamical_matrix_roundtrips() -> None:
    """Manager.dynamical_matrix_eskm forwards positions/free_indices/dx and resolves."""
    fake = _FakeSession({"dynamical_matrix_eskm": _HESSIAN_SENTINEL})
    manager = Manager(local_sessions=[fake])
    manager.start()
    try:
        future = manager.dynamical_matrix_eskm(
            positions="POS", free_indices=[1, 2], dx=0.02
        )
        assert future.result(timeout=5) is _HESSIAN_SENTINEL
        assert fake.calls == [
            (
                "dynamical_matrix_eskm",
                {"positions": "POS", "free_indices": [1, 2], "dx": 0.02},
            )
        ]
    finally:
        manager.shutdown()


def test_pool_fans_out_one_job_per_event() -> None:
    """EventPrefactorPool submits one compute_event_prefactors job per event."""
    fake = _FakeSession({"compute_event_prefactors": "PRE"})
    manager = Manager(local_sessions=[fake])
    manager.start()
    try:
        events = [
            {"central_atom_idx": 1, "cell": "c"},
            {"central_atom_idx": 2, "cell": "c"},
        ]
        futures = EventPrefactorPool(manager).compute_event_prefactors("CFG", events)
        assert [f.result(timeout=5) for f in futures] == ["PRE", "PRE"]
        assert fake.calls == [
            (
                "compute_event_prefactors",
                {"config": "CFG", "central_atom_idx": 1, "cell": "c"},
            ),
            (
                "compute_event_prefactors",
                {"config": "CFG", "central_atom_idx": 2, "cell": "c"},
            ),
        ]
    finally:
        manager.shutdown()


def test_pool_with_no_events_submits_nothing() -> None:
    """An empty batch costs no job and returns an empty list of futures."""
    fake = _FakeSession({})
    manager = Manager(local_sessions=[fake])
    manager.start()
    try:
        assert EventPrefactorPool(manager).compute_event_prefactors("CFG", []) == []
        assert fake.calls == []
    finally:
        manager.shutdown()
