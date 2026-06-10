"""Plumbing tests for the standalone HTST engine operations.

Covers the three layers that make ``get_forces`` and ``dynamical_matrix_eskm``
manager-reachable without MPI: (1) the engine operations map, (2) the session
send/recv methods (via a fake messenger), and (3) the Manager single-job
wrappers (via a fake session resolved by the real worker thread). The real
engine functions themselves are already covered serially in
``test_engine_prefactors.py`` / ``test_profiling.py``.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

pytest.importorskip("lammps")
pytest.importorskip("mpi4py")

from pykmc.enginemanager.lmpi.engines.mpi_api_engine import MpiApiEngine  # noqa: E402
from pykmc.enginemanager.lmpi.pool.manager import Manager  # noqa: E402
from pykmc.enginemanager.lmpi.sessions.mpi_api_sessions import MpiApiSession  # noqa: E402

_FORCES_SENTINEL = object()
_HESSIAN_SENTINEL = object()


class _FakeMessenger:
    """Record sends; answer status on tag 0 and a canned result on tag 1."""

    def __init__(self, result: object) -> None:
        self.sent: list[dict[str, Any]] = []
        self._result = result

    def send(self, msg: dict[str, Any], dest: int, tag: int) -> None:
        self.sent.append(msg)

    def recv(self, source: int, tag: int) -> dict[str, Any]:
        if tag == 0:
            return {"type": "status", "value": {"alive": True, "busy": False}}
        return {"type": "result", "value": self._result}


def test_engine_registers_htst_ops() -> None:
    """The ops map dispatches get_forces and dynamical_matrix_eskm."""
    engine = MpiApiEngine(None, None, 1, None, None, 0)
    assert "get_forces" in engine._operations_map
    assert "dynamical_matrix_eskm" in engine._operations_map


def test_session_get_forces_roundtrips() -> None:
    """Session sends the op message and returns the engine result."""
    messenger = _FakeMessenger(_FORCES_SENTINEL)
    session = MpiApiSession(messenger, engine_ranks=[1], session_id=1)

    out = session.get_forces(positions="POS")

    assert out is _FORCES_SENTINEL
    sent = messenger.sent[0]
    assert sent["type"] == "get_forces"
    assert sent["value"] == {"positions": "POS"}


def test_session_dynamical_matrix_roundtrips() -> None:
    """Session sends positions/free_indices/dx and returns the Hessian."""
    messenger = _FakeMessenger(_HESSIAN_SENTINEL)
    session = MpiApiSession(messenger, engine_ranks=[1], session_id=1)

    out = session.dynamical_matrix_eskm(positions="POS", free_indices=[1, 2], dx=0.02)

    assert out is _HESSIAN_SENTINEL
    sent = messenger.sent[0]
    assert sent["type"] == "dynamical_matrix_eskm"
    assert sent["value"] == {"positions": "POS", "free_indices": [1, 2], "dx": 0.02}


class _FakeSession:
    """Expose the two session methods; record kwargs; return sentinels."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_forces(self, positions: Optional[object] = None) -> object:
        self.calls.append(("get_forces", {"positions": positions}))
        return _FORCES_SENTINEL

    def dynamical_matrix_eskm(
        self, positions: object, free_indices: Optional[object] = None, dx: float = 1e-2
    ) -> object:
        self.calls.append(
            ("dynamical_matrix_eskm", {"positions": positions, "free_indices": free_indices, "dx": dx})
        )
        return _HESSIAN_SENTINEL


def test_manager_compute_forces_forwards() -> None:
    """Manager.compute_forces submits a job the worker resolves to the result."""
    fake = _FakeSession()
    manager = Manager(sessions=[fake])

    future = manager.compute_forces(positions="POS")

    assert future.result(timeout=5) is _FORCES_SENTINEL
    assert fake.calls == [("get_forces", {"positions": "POS"})]


def test_manager_compute_dynamical_matrix_forwards() -> None:
    """Manager.compute_dynamical_matrix forwards all kwargs and resolves."""
    fake = _FakeSession()
    manager = Manager(sessions=[fake])

    future = manager.compute_dynamical_matrix(positions="POS", free_indices=[3], dx=0.05)

    assert future.result(timeout=5) is _HESSIAN_SENTINEL
    assert fake.calls == [
        ("dynamical_matrix_eskm", {"positions": "POS", "free_indices": [3], "dx": 0.05})
    ]
