"""Forces + eskm Hessian round-trip through the MPI session pool (mpirun -n 8).

The HTST extension's methods must reach the Manager op registry through the
plugin mechanism alone (``EngineManagerFactory(engine_extensions=[...])``); this
is the oracle's ``test_compute_forces_and_dynamical_matrix_manager`` on the
plugin architecture. Skips unless launched with exactly 8 ranks
(7 single-rank workers + the rank-0 driver).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip("lammps")
pytest.importorskip("mpi4py")

from mpi4py import MPI  # noqa: E402

from pykmc.factory import EngineManagerFactory  # noqa: E402
from pykmc.htst.lammps_extension import HtstLammpsExtension  # noqa: E402


@dataclass
class _LammpsConfigNi:
    pair_style: str = "lj/cut 6.0"
    pair_coeff: str = "* * 0.52 2.274"
    min_style: str = "cg"
    minimize: str = "1e-6 1e-8 1000 10000"
    frz_min: str = "1e-4 1e-6 100 1000"
    verbosity: int = 0


def test_compute_forces_and_dynamical_matrix_manager(ni_orthorhombic) -> None:
    """Forces + eskm Hessian round-trip through the session pool (local mode)."""
    if MPI.COMM_WORLD.Get_size() != 8:
        pytest.skip("needs mpirun -n 8 (7 single-rank workers + rank-0 driver)")
    system = ni_orthorhombic
    MPI.COMM_WORLD.Barrier()
    manager = EngineManagerFactory(
        engine_style="lammps",
        engine_config=_LammpsConfigNi(),
        n_workers=7,
        comm=MPI.COMM_WORLD,
        engine_extensions=[HtstLammpsExtension],
    ).launch()
    if manager is None:
        return  # worker ranks stop here
    # ------------ DRIVER CODE (rank 0) ------------
    try:
        manager.broadcast("start")
        manager.broadcast("initialize_parameters")
        manager.broadcast(
            "initialize_system",
            types=system.types,
            positions=system.positions,
            cell=system.cell,
            pbc=system.pbc,
        )
        manager.broadcast("initialize_potential")

        f = manager.get_forces(positions=system.positions.copy())
        forces = f.result()
        assert forces.shape == (system.positions.shape[0], 3)
        assert np.isfinite(forces).all()

        free = [0, 1]
        g = manager.dynamical_matrix_eskm(
            positions=system.positions.copy(), free_indices=free, dx=0.01
        )
        hessian = g.result()
        assert hessian.shape == (3 * len(free), 3 * len(free))
        assert np.isfinite(hessian).all()
        assert np.allclose(hessian, hessian.T)  # symmetrized by the op
    finally:
        manager.shutdown()
