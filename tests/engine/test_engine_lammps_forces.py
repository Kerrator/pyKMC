"""Force calls on LammpsEngine: finite difference, energy, MPI atom order.

Serial: pytest tests/engine/test_engine_lammps_forces.py
MPI:    mpirun -n 4 pytest tests/engine/test_engine_lammps_forces.py
"""

import os
from dataclasses import dataclass, field

import numpy as np
import pytest

lammps = pytest.importorskip("lammps")

from pykmc.engine import Engine  # noqa: E402

POTENTIAL = os.path.join(
    os.path.dirname(__file__), "..", "data", "Ni_v6_2.0_LKBeland2016.eam"
)

# Central-difference step (Å) and the agreement forces must reach (eV/Å).
# With this EAM potential, h = 1e-4 Å leaves errors near 5e-8 eV/Å, so the
# tolerance keeps a factor of 20. Forces left in the LAMMPS frame miss by
# about 3 eV/Å on the non-orthorhombic cell.
FD_STEP = 1e-4
FD_ATOL = 1e-6
# Leader-rank values against a COMM_SELF engine: the two differ only in the
# order contributions are summed across subdomains.
MPI_FORCE_ATOL = 1e-9
MPI_ENERGY_ATOL = 1e-8


@pytest.fixture(scope="module")
def eam_config():
    from mpi4py import MPI

    lmp = lammps.lammps(comm=MPI.COMM_SELF, cmdargs=["-log", "none", "-screen", "none"])
    has_eam = lmp.has_style("pair", "eam/alloy")
    lmp.close()
    if not has_eam:
        pytest.skip("LAMMPS built without pair_style eam/alloy (MANYBODY)")

    @dataclass
    class LammpsConfig:
        pair_style: str = "eam/alloy"
        pair_coeff: list[str] = field(default_factory=lambda: [f"* * {POTENTIAL} Ni"])
        min_style: str = "cg"
        minimize: str = "1e-6 1e-8 1000 10000"
        frz_min: str = "1e-4 1e-6 100 1000"
        verbosity: int = 0

    return LammpsConfig()


@pytest.fixture(params=["ni_orthorhombic", "ni_triclinic"])
def force_system(request):
    return request.getfixturevalue(request.param)


def _make_engine(config, comm, system) -> Engine:
    engine = Engine.create("lammps", config=config, comm=comm)
    engine.start()
    engine.initialize_parameters()
    engine.initialize_system(
        types=system.types,
        positions=system.positions,
        cell=system.cell,
        pbc=system.pbc,
    )
    engine.initialize_potential()
    return engine


def _off_site(system, seed: int = 7) -> np.ndarray:
    """Atoms moved off their sites, the same array on every rank."""
    rng = np.random.default_rng(seed)
    return system.positions + rng.uniform(-0.1, 0.1, size=system.positions.shape)


class TestLammpsForcesSerial:
    @pytest.fixture(autouse=True)
    def require_serial(self):
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_size() > 1:
            pytest.skip("serial tests must run without mpirun")

    def test_shapes_energy_and_held_positions(self, eam_config, force_system):
        """Both calls agree, the energy is the potential energy, positions are optional."""
        engine = _make_engine(eam_config, None, force_system)
        positions = _off_site(force_system)
        natoms = len(positions)

        forces = engine.get_forces(positions=positions)
        energy, forces_ef = engine.get_energy_and_forces(positions=positions)
        assert forces.shape == (natoms, 3)
        assert forces.dtype == np.float64
        assert isinstance(energy, float)
        np.testing.assert_array_equal(forces_ef, forces)
        assert energy == pytest.approx(
            engine.get_potential_energy(positions=positions), abs=1e-10
        )

        engine.set_positions(positions)
        np.testing.assert_array_equal(engine.get_forces(), forces)
        engine.close()

    def test_forces_match_finite_difference(self, eam_config, force_system):
        """Forces are minus the central difference of the engine's energy."""
        engine = _make_engine(eam_config, None, force_system)
        if not force_system.cell.orthorhombic:
            # LAMMPS must work in a rotated frame here, or the test proves nothing.
            assert not np.allclose(engine.Q, np.eye(3))
        positions = _off_site(force_system)
        forces = engine.get_forces(positions=positions)

        fd = np.empty_like(positions)
        for i in range(len(positions)):
            for k in range(3):
                plus = positions.copy()
                minus = positions.copy()
                plus[i, k] += FD_STEP
                minus[i, k] -= FD_STEP
                fd[i, k] = -(
                    engine.get_potential_energy(positions=plus)
                    - engine.get_potential_energy(positions=minus)
                ) / (2 * FD_STEP)
        engine.close()
        np.testing.assert_allclose(forces, fd, rtol=0, atol=FD_ATOL)


@pytest.mark.mpi
class TestLammpsForcesMPI:
    """
    Run with: mpirun -n 4 pytest tests/engine/test_engine_lammps_forces.py

    The leader rank's values are checked against a COMM_SELF engine built on
    that rank, atom for atom. A serial test cannot catch an atom-order leak.
    """

    @pytest.fixture(autouse=True)
    def require_mpi(self):
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_size() == 1:
            pytest.skip("requires mpirun -n N")
        yield
        MPI.COMM_WORLD.Barrier()

    def test_leader_matches_comm_self_engine(self, eam_config, force_system):
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        positions = _off_site(force_system)
        natoms = len(positions)

        engine = _make_engine(eam_config, comm, force_system)
        forces = engine.get_forces(positions=positions)
        energy_and_forces = engine.get_energy_and_forces(positions=positions)
        owned = comm.gather(engine.lmp.extract_setting("nlocal"), root=0)
        engine.close()

        if comm.Get_rank() != 0:
            assert forces is None
            assert energy_and_forces is None
            return

        # Atoms are spread over the ranks, so gathering order is exercised.
        assert sum(owned) == natoms
        assert max(owned) < natoms

        reference = _make_engine(eam_config, MPI.COMM_SELF, force_system)
        ref_energy, ref_forces = reference.get_energy_and_forces(positions=positions)
        reference.close()

        energy, forces_ef = energy_and_forces
        assert forces.shape == (natoms, 3)
        np.testing.assert_allclose(forces, ref_forces, rtol=0, atol=MPI_FORCE_ATOL)
        np.testing.assert_allclose(forces_ef, ref_forces, rtol=0, atol=MPI_FORCE_ATOL)
        assert energy == pytest.approx(ref_energy, abs=MPI_ENERGY_ATOL)
