"""MPI-only tests for the LAMMPS engine/session API and the rank-0 session pool.

Every test here needs a real MPI launch -- sessions live on COMM_WORLD rank 0 and
the engines on ranks ``[1, world_size)``::

    mpirun -n 8 python -m pytest tests/test_lammps_engine_api_mpi.py

Tests skip (rather than hang or raise) when the world is too small.

Two families are covered:

* *direct* tests wire one ``MpiApiEngine`` by hand and drive it through a single
  ``MpiApiSession`` -- the lowest layer of the protocol (tags 0/1/2);
* *manager* tests go through ``ManagerFactory``/``Manager``, the layer the KMC
  driver actually uses.

Both families must respect the engine's GLOBAL/LOCAL mode: only the master rank
of the *active* communicator reads incoming messages, so addressing a session
whose mode is not active leaves rank 0 blocked in ``receive_status()`` forever.
"""

import os

import numpy as np
import pytest
from mpi4py import MPI
from pytest_lazy_fixtures import lf

from pykmc import Config, System
from pykmc.enginemanager.lmpi.engines import MpiApiEngine
from pykmc.enginemanager.lmpi.pool import Manager, ManagerFactory
from pykmc.enginemanager.lmpi.sessions import MpiApiSession
from pykmc.enginemanager.messenger import MpiMessenger

_SYSTEM_AND_CONFIG = pytest.mark.parametrize(
    "system, config",
    [(lf("system_single_type_fcc"), lf("config_system_single_type"))],
)

# Total energy of the relaxed 256-atom Ni fcc fixture (decomposition-invariant).
_MINIMIZED_ENERGY = -1139.1999963495148


def _require_ranks(needed: int) -> None:
    """Skip collectively when the world is smaller than ``needed`` ranks."""
    if MPI.COMM_WORLD.Get_size() < needed:
        pytest.skip(f"needs mpirun -n {needed}")


def _launch_direct_session() -> "MpiApiSession | None":
    """Wire one dual-mode engine over ranks ``[1, size)`` and return its session.

    ``MpiApiEngine`` owns a *local* and a *global* LAMMPS instance, each with its
    own communicator and messenger, so both must be supplied here exactly as
    ``ManagerFactory.launch`` supplies them. The returned session talks to the
    global instance, which is the mode the engine boots in.

    Returns
    -------
    MpiApiSession or None
        The session on rank 0; ``None`` on the engine ranks, which only return
        once the engine loop has been shut down.

    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    engine_ranks = list(range(1, size))
    color = 1 if rank in engine_ranks else MPI.UNDEFINED

    local_engine_comm = comm.Split(color=color, key=rank)
    global_engine_comm = comm.Split(color=color, key=rank)
    local_messenger = MpiMessenger(comm=comm)
    global_messenger = MpiMessenger(comm=comm)

    if rank in engine_ranks:
        engine = MpiApiEngine(
            local_messenger=local_messenger,
            local_engine_comm=local_engine_comm,
            local_engine_id=1,
            global_messenger=global_messenger,
            global_engine_comm=global_engine_comm,
            global_engine_id=0,
        )
        engine.start()
        return None

    return MpiApiSession(
        messenger=global_messenger,
        engine_ranks=engine_ranks,
        session_id=0,
    )


def _launch_pool(config: Config, system: System, n_sessions: int) -> "Manager | None":
    """Launch and initialize an engine-rank-only pool of ``n_sessions`` sessions.

    Parameters
    ----------
    config : Config
        Parsed test configuration; ``control.n_sessions`` /
        ``control.engine_use_rank_0`` are pinned to the launched layout.
    system : System
        System every engine is initialized with.
    n_sessions : int
        Number of local sessions to carve out of the engine ranks.

    Returns
    -------
    Manager or None
        The manager on rank 0, ``None`` on the engine ranks. The pool is left in
        GLOBAL mode, exactly as ``initialize_sessions`` leaves it.

    """
    # engine_use_rank_0=True is deprecated and deadlocks on this branch lineage:
    # rank 0 would block in run_engine_loop and never build the sessions.
    config.control.n_sessions = n_sessions
    config.control.engine_use_rank_0 = False
    factory = ManagerFactory(n_sessions=n_sessions, use_rank_0=False, has_global=True)
    manager = factory.launch()
    if manager is None:
        return None
    manager.initialize_sessions(config, system)
    return manager


class TestLammpsApiMpiEngine:
    """Direct engine/session protocol tests (no Manager in the loop)."""

    def test_send_commends_from_session(self) -> None:
        """Plain LAMMPS commands sent through a session reach the log."""
        _require_ranks(2)
        session = _launch_direct_session()
        if session is None:
            return  # Engine processes stop here

        # ------------ SESSION CODE (rank 0) ------------
        session.command("units metal")
        session.command("dimension 3")
        session.command("log flush")
        session.close(wait_status=True)

        logfile = os.path.join(os.getcwd(), "lammps.log.0")
        with open(logfile) as handle:
            log_text = handle.read()
        assert "units metal" in log_text

    @_SYSTEM_AND_CONFIG
    def test_initialize_session(self, system: System, config: Config) -> None:
        """The initialize_* trio builds the box and potential on the engine."""
        _require_ranks(2)
        session = _launch_direct_session()
        if session is None:
            return  # Engine processes stop here

        # ------------ SESSION CODE (rank 0) ------------
        session.initialize_parameters()
        session.initialize_system(system)
        session.initialize_potential(config)
        session.command("log flush")
        session.close(wait_status=True)

        logfile = os.path.join(os.getcwd(), "lammps.log.0")
        with open(logfile) as handle:
            log_text = handle.read()
        assert "units metal" in log_text
        assert "atom_style atomic" in log_text
        assert "dimension 3" in log_text
        assert "boundary p p p" in log_text
        assert "atom_modify sort 0 0.0" in log_text
        assert "region box" in log_text
        assert "create_box" in log_text

    @_SYSTEM_AND_CONFIG
    def test_minimize(self, system: System, config: Config) -> None:
        """A minimize round-trips through the session without desyncing."""
        _require_ranks(2)
        session = _launch_direct_session()
        if session is None:
            return  # Engine processes stop here

        # ------------ SESSION CODE (rank 0) ------------
        session.initialize_parameters()
        session.initialize_system(system)
        session.initialize_potential(config)
        session.minimize(config)
        session.command("log flush")
        session.close(wait_status=True)

        logfile = os.path.join(os.getcwd(), "lammps.log.0")
        with open(logfile) as handle:
            log_text = handle.read()
        assert "units metal" in log_text
        assert "atom_style atomic" in log_text
        assert "region box" in log_text
        assert "create_box" in log_text

    @_SYSTEM_AND_CONFIG
    def test_get_total_energy(self, system: System, config: Config) -> None:
        """The minimized total energy comes back to rank 0 unchanged."""
        _require_ranks(2)
        session = _launch_direct_session()
        if session is None:
            return  # Engine processes stop here

        # ------------ SESSION CODE (rank 0) ------------
        session.initialize_parameters()
        session.initialize_system(system)
        session.initialize_potential(config)
        session.minimize(config)
        energy = session.get_total_energy()
        session.command("log flush")
        session.close(wait_status=True)

        assert round(energy, 3) == round(_MINIMIZED_ENERGY, 3)

    @_SYSTEM_AND_CONFIG
    def test_get_positions(self, system: System, config: Config) -> None:
        """Positions gather back to rank 0 with the full system shape."""
        _require_ranks(2)
        session = _launch_direct_session()
        if session is None:
            return  # Engine processes stop here

        # ------------ SESSION CODE (rank 0) ------------
        session.initialize_parameters()
        session.initialize_system(system)
        session.initialize_potential(config)
        session.minimize(config)
        positions = session.get_positions()
        session.command("log flush")
        session.close(wait_status=True)

        assert np.asarray(positions).shape == system.positions.shape

    @_SYSTEM_AND_CONFIG
    def test_set_positions(self, system: System, config: Config) -> None:
        """A scattered position update survives the following gather."""
        _require_ranks(2)
        session = _launch_direct_session()
        if session is None:
            return  # Engine processes stop here

        # ------------ SESSION CODE (rank 0) ------------
        session.initialize_parameters()
        session.initialize_system(system)
        session.initialize_potential(config)
        positions = session.get_positions()
        positions[0][0], positions[0][1], positions[0][2] = 0.1, 0.2, 0.3
        session.set_positions(positions)
        positions = session.get_positions()
        session.command("log flush")
        session.close(wait_status=True)

        assert positions[0][0] == 0.1
        assert positions[0][1] == 0.2
        assert positions[0][2] == 0.3


class TestManagerPool:
    """Session-pool tests driven through ``ManagerFactory``/``Manager``."""

    @_SYSTEM_AND_CONFIG
    def test_initialize_manager(self, system: System, config: Config) -> None:
        """The pool initializes every local session and the global one."""
        _require_ranks(2)
        manager = _launch_pool(config, system, n_sessions=1)
        if manager is None:
            return  # Engine processes stop here
        # ------------ SESSION CODE (rank 0) ------------
        manager.close_all()

    @_SYSTEM_AND_CONFIG
    def test_minimize_manager(self, system: System, config: Config) -> None:
        """A minimize job runs on the local pool."""
        _require_ranks(2)
        manager = _launch_pool(config, system, n_sessions=1)
        if manager is None:
            return  # Engine processes stop here
        # ------------ SESSION CODE (rank 0) ------------
        manager.use_local()  # initialize_sessions leaves the pool in GLOBAL mode
        manager.minimize(config).result()
        manager.close_all()

    @_SYSTEM_AND_CONFIG
    def test_minimize_with_results_manager(
        self, system: System, config: Config
    ) -> None:
        """minimize_with_results returns the relaxed geometry and its energy."""
        _require_ranks(2)
        manager = _launch_pool(config, system, n_sessions=1)
        if manager is None:
            return  # Engine processes stop here
        # ------------ SESSION CODE (rank 0) ------------
        manager.use_local()
        positions, total_energy = manager.minimize_with_results(config).result()
        manager.close_all()

        assert np.asarray(positions).shape == system.positions.shape
        assert round(total_energy, 3) == round(_MINIMIZED_ENERGY, 3)

    @_SYSTEM_AND_CONFIG
    def test_partn_manager(
        self, system: System, config: Config, capfd: "pytest.CaptureFixture[str]"
    ) -> None:
        """Refinements fan out over the local pARTn pool and every future resolves.

        pARTn runs on the LOCAL pool with one rank per session: a multi-rank
        session leaves ranks out of the close-time barrier and ``close_all``
        hangs. ``capfd.disabled()`` is required because pARTn's Fortran output
        deadlocks pytest's fd capture under MPI.
        """
        _require_ranks(8)
        with capfd.disabled():
            manager = _launch_pool(config, system, n_sessions=7)
            if manager is None:
                return  # Engine processes stop here
            # ------------ SESSION CODE (rank 0) ------------
            manager.use_local()
            futures = [manager.partn_refine(config, atom) for atom in 20 * [0]]
            results = [future.result() for future in futures]
            manager.close_all()

        assert len(results) == 20
        assert all(result is not None for result in results)

    @_SYSTEM_AND_CONFIG
    def test_partn_search_manager(
        self, system: System, config: Config, capfd: "pytest.CaptureFixture[str]"
    ) -> None:
        """A forward pARTn search round-trips a Result through the local pool."""
        _require_ranks(8)
        with capfd.disabled():
            manager = _launch_pool(config, system, n_sessions=7)
            if manager is None:
                return  # Engine processes stop here
            # ------------ SESSION CODE (rank 0) ------------
            manager.use_local()
            results = [future.result() for future in manager.partn_search(config, [0])]
            manager.close_all()

        assert len(results) == 1
        assert results[0] is not None

    @_SYSTEM_AND_CONFIG
    def test_compute_forces_and_dynamical_matrix_manager(
        self, system: System, config: Config
    ) -> None:
        """Forces + eskm Hessian round-trip through the session pool (local mode).

        Uses the n_sessions=7 / engine_use_rank_0=False layout (mpirun -n 8):
        ``get_forces`` only returns on engine rank 0, so anything Hessian- or
        forces-shaped needs one rank per session.
        """
        _require_ranks(8)
        manager = _launch_pool(config, system, n_sessions=7)
        if manager is None:
            return  # Engine processes stop here
        # ------------ SESSION CODE (rank 0) ------------
        manager.use_local()

        forces = manager.compute_forces(positions=system.positions.copy()).result()
        assert forces.shape == (system.positions.shape[0], 3)
        assert np.isfinite(forces).all()

        free = [0, 1]
        hessian = manager.compute_dynamical_matrix(
            positions=system.positions.copy(), free_indices=free, dx=0.01
        ).result()
        assert hessian.shape == (3 * len(free), 3 * len(free))
        assert np.isfinite(hessian).all()
        assert np.allclose(hessian, hessian.T)  # symmetrized by the op

        manager.close_all()
