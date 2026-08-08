"""Tests for the rank-0 ``Manager`` driving an MPI pool of LAMMPS engines.

These tests need a real MPI launch -- ``ManagerFactory`` places the sessions on
COMM_WORLD rank 0 and the engines on ranks ``[1, world_size)``::

    mpirun -n 8 python -m pytest tests/manager/lmpi/test_manager.py

Under fewer ranks than a test needs they skip instead of raising the factory's
``ValueError("Not enough MPI ranks to allocate sessions")``.

Two mode rules drive the shape of every test below:

* The pool boots in GLOBAL mode (``MpiApiEngine.__init__`` calls ``use_global``),
  where only the *global* master rank reads incoming messages. Talking to the
  local sessions before ``use_local()`` therefore addresses engine ranks that are
  not listening, and rank 0 blocks forever in ``receive_status()``.
* ``initialize_sessions`` leaves the pool in GLOBAL mode, so per-session work
  needs an explicit ``use_local()`` afterwards.
"""

import pytest
from mpi4py import MPI
from pytest_lazy_fixtures import lf

from pykmc import Config, System
from pykmc.enginemanager.lmpi.pool import Manager, ManagerFactory


def _launch_pool(n_sessions: int) -> "Manager | None":
    """Launch a pool of ``n_sessions`` engine-rank-only sessions plus a global one.

    Parameters
    ----------
    n_sessions : int
        Number of local sessions to carve out of the engine ranks.

    Returns
    -------
    Manager or None
        The manager on COMM_WORLD rank 0, ``None`` on every engine rank (they
        only return once their engine loop has been shut down).

    """
    # engine ranks start at 1 (engine_use_rank_0=False), so the world needs one
    # extra rank for the rank-0 driver.
    needed = n_sessions + 1
    if MPI.COMM_WORLD.Get_size() < needed:
        pytest.skip(
            f"needs mpirun -n {needed} (n_sessions={n_sessions} + rank-0 driver)"
        )
    factory = ManagerFactory(n_sessions=n_sessions, use_rank_0=False, has_global=True)
    return factory.launch()


class TestManager:
    """End-to-end checks of the manager/session/engine message plumbing."""

    def test_initialize_manager(self) -> None:
        """Commands reach both the local session pool and the global session."""
        manager = _launch_pool(n_sessions=2)
        if manager is None:
            return  # Engine processes stop here
        # ------------ SESSION CODE (rank 0) ------------
        # broadcast_command talks to the LOCAL sessions; the pool boots global.
        manager.use_local()
        manager.broadcast_command("units metal")
        manager.broadcast_command("log flush")

        manager.use_global()
        manager.global_session.command("dimension 3")
        manager.global_session.command("log flush")

        manager.close_all()

    @pytest.mark.parametrize(
        "system, config",
        [(lf("system_single_type_fcc"), lf("config_system_single_type"))],
    )
    def test_minimize_manager(self, system: System, config: Config) -> None:
        """A minimize runs on the local pool and on the global session."""
        manager = _launch_pool(n_sessions=config.control.n_sessions)
        if manager is None:
            return  # Engine processes stop here
        # ------------ SESSION CODE (rank 0) ------------
        # initialize_sessions initializes the local pool AND the global session.
        manager.initialize_sessions(config, system)

        manager.use_local()
        future = manager.minimize(config)
        future.result()

        manager.use_global()
        manager.global_minimize(config)

        manager.close_all()
