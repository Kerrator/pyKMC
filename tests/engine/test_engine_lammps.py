import pytest

pytest.importorskip("lammps")

from pykmc.engine import Engine, EngineExtension
from pykmc.system import System
from .test_engine_contract import EngineContractTests
from dataclasses import dataclass
from typing import Any


@pytest.fixture(params=["ni_orthorhombic", "ni_triclinic", "ni_slab"])
def system(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session")
def lammps_config_Ni():
    @dataclass
    class LammpsConfig:
        pair_style: str = "lj/cut 6.0"
        pair_coeff: str = "* * 0.52 2.274"
        min_style: str = "cg"
        minimize: str = "1e-6 1e-8 1000 10000"
        frz_min: str = "1e-4 1e-6 100 1000"
        verbosity: int = 0

    return LammpsConfig()


class TestLammpsEngineSerial(EngineContractTests):
    @pytest.fixture(autouse=True)
    def require_serial(self):
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_size() > 1:
            pytest.skip("serial tests must run without mpirun")

    @pytest.fixture(autouse=True)
    def setup(self, lammps_config_Ni, system):
        self.config = lammps_config_Ni
        self.system = system

    def make_engine(self):
        return Engine.create("lammps", config=self.config, comm=None)

    def make_test_extension(self, engine) -> EngineExtension:
        return _ComputeKineticEnergy(engine=engine)

    def make_conflicting_extension(self, engine) -> EngineExtension:
        return _ConflictingExtension(engine=engine)

    def test_kinetic_energy_returns_float(self):
        """_ComputeKineticEnergy.get_kinetic_energy() returns a float."""
        engine = self.make_engine()
        engine.start()
        self.initialize(engine)
        _ComputeKineticEnergy(engine=engine)
        ke = engine.get_kinetic_energy()
        assert isinstance(ke, float)
        engine.close()


@pytest.mark.mpi
class TestLammpsEngineMPI(EngineContractTests):
    """
    Run with: mpirun -n 4 pytest tests/engine/test_engine_lammps.py

    All ranks execute each test collectively.
    Scalar assertions are restricted to rank 0 via is_rank0.
    """

    @pytest.fixture(autouse=True)
    def setup(self, lammps_config_Ni, system):
        from mpi4py import MPI

        self.config = lammps_config_Ni
        self.system = system
        self.comm = MPI.COMM_WORLD
        yield
        MPI.COMM_WORLD.Barrier()

    @pytest.fixture(autouse=True)
    def require_mpi(self):
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_size() == 1:
            pytest.skip("requires mpirun -n N")

    @property
    def is_rank0(self) -> bool:
        from mpi4py import MPI

        return self.comm.Get_rank() == 0

    def make_engine(self) -> Engine:
        return Engine.create("lammps", config=self.config, comm=self.comm)

    def make_test_extension(self, engine) -> EngineExtension:
        return _ComputeKineticEnergy(engine=engine)

    def make_conflicting_extension(self, engine) -> EngineExtension:
        return _ConflictingExtension(engine=engine)

    def test_kinetic_energy_returns_float(self):
        """_ComputeKineticEnergy.get_kinetic_energy() returns a float on rank 0."""
        engine = self.make_engine()
        engine.start()
        self.initialize(engine)
        _ComputeKineticEnergy(engine=engine)
        ke = engine.get_kinetic_energy()
        if self.is_rank0:
            assert isinstance(ke, float)
        engine.close()


class _ComputeKineticEnergy(EngineExtension):
    """Test extension: adds get_kinetic_energy() via a LAMMPS compute."""

    def __init__(self, engine):
        super().__init__(engine)

    def get_kinetic_energy(self) -> float | None:
        compute_id = "test_ke"
        reduce_id = "test_ke_reduce"
        if not self.engine._has_compute(compute_id):
            self.engine.lmp.command(f"compute {compute_id} all ke/atom")
            self.engine.lmp.command(
                f"compute {reduce_id} all reduce sum c_{compute_id}"
            )
        self.engine.lmp.command("run 0 post no")
        result = self.engine.lmp.extract_compute(reduce_id, 0, 0)
        if self.engine._is_rank0:
            return float(result)
        return None


class _ConflictingExtension(EngineExtension):
    """Test extension: same method name as _ComputeKineticEnergy to test conflict detection."""

    def __init__(self, engine):
        super().__init__(engine)

    def get_kinetic_energy(self) -> None:
        pass


def _max_mic(positions: Any, target: Any, lengths: Any) -> float:
    """Largest per-axis minimum-image deviation of ``positions`` from ``target``."""
    import numpy as np

    d = np.asarray(positions, dtype=float) - np.asarray(target, dtype=float)
    d -= lengths * np.round(d / lengths)
    return float(np.max(np.abs(d)))


@pytest.mark.mpi
class TestLammpsScatterMigrationMPI:
    """A scatter may move an atom by any distance without losing it.

    ``scatter_atoms`` writes coordinates on the rank that currently owns each
    atom. At the next setup the brick ``Comm::exchange`` offers an atom that
    left its sub-box only to the two neighbouring sub-boxes along each axis,
    so an atom moved two or more sub-boxes away is dropped ("Lost atoms").
    With one to three sub-boxes along an axis every sub-box neighbours every
    other one and nothing can be lost, so these tests split x over all ranks
    and need at least four of them.
    """

    @pytest.fixture(autouse=True)
    def require_four_ranks(self) -> None:
        """Skip below four ranks: fewer sub-boxes per axis cannot lose an atom."""
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_size() < 4:
            pytest.skip("needs >= 4 ranks along one axis (mpirun -n 4)")

    @staticmethod
    def _engine(config: Any, system: System) -> tuple[Any, Any]:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        engine = Engine.create("lammps", config=config, comm=comm)
        engine.start()
        engine.initialize_parameters()
        engine.lmp.command(f"processors {comm.Get_size()} 1 1")
        engine.initialize_system(
            types=system.types,
            positions=system.positions,
            cell=system.cell,
            pbc=system.pbc,
        )
        engine.initialize_potential()
        engine.lmp.command("run 0 post no")  # establish ownership
        return engine, comm

    @staticmethod
    def _owned(engine: Any, comm: Any) -> int:
        from mpi4py import MPI

        return comm.allreduce(int(engine.lmp.extract_global("nlocal")), op=MPI.SUM)

    @staticmethod
    def _max_mic_deviation(engine: Any, comm: Any, target: Any, lengths: Any) -> float:
        """Broadcast rank 0's largest minimum-image deviation from ``target``."""
        got = engine.get_positions()
        value = _max_mic(got, target, lengths) if comm.Get_rank() == 0 else None
        return comm.bcast(value, root=0)

    def test_scatter_migration_swap_across_two_slabs(
        self, lammps_config_Ni: Any, ni_orthorhombic: System
    ) -> None:
        """Swapping two atoms two sub-boxes apart keeps every atom."""
        import numpy as np

        engine, comm = self._engine(lammps_config_Ni, ni_orthorhombic)
        positions = np.array(ni_orthorhombic.positions, dtype=float)
        lengths = np.diag(np.asarray(ni_orthorhombic.cell, dtype=float))
        n = comm.Get_size()
        width = lengths[0] / n
        slab = np.floor(positions[:, 0] / width).astype(int) % n
        row = np.flatnonzero(np.all(np.abs(positions[:, 1:]) < 1e-9, axis=1))
        a = int(row[np.argmin(positions[row, 0])])  # the atom at the origin
        far = [int(i) for i in row if 2 <= (slab[i] - slab[a]) % n <= n - 2]
        assert far, "no atom two sub-boxes away on the x row through the origin"
        b = far[0]
        swapped = positions.copy()
        swapped[[a, b]] = positions[[b, a]]
        natoms = int(engine.lmp.get_natoms())

        try:
            engine.set_positions(swapped)
            engine.lmp.command("run 0 post no")  # "Lost atoms" without migration

            assert int(engine.lmp.get_natoms()) == natoms
            assert self._owned(engine, comm) == natoms
            # Every rank asserts the same broadcast value: a rank-0-only
            # failure would leave the other ranks in the next collective.
            assert self._max_mic_deviation(engine, comm, swapped, lengths) < 1e-12
        finally:
            engine.close()

    def test_scatter_migration_constrained_sequence_across_slabs(
        self, lammps_config_Ni: Any, ni_orthorhombic: System
    ) -> None:
        """Constrained minimisations scatter every request over the entry state.

        The endpoint transaction restores the entry coordinates after each
        call, so rigid translations of 0.75 slab per call are scattered 1.5
        slab away from the native coordinates at the second call.
        """
        import numpy as np

        from pykmc.physics import ResolvedConstraints

        engine, comm = self._engine(lammps_config_Ni, ni_orthorhombic)
        entry = comm.bcast(engine.get_positions(), root=0)
        types = list(ni_orthorhombic.types)
        cell = np.asarray(ni_orthorhombic.cell, dtype=float)
        lengths = np.diag(cell)
        payload = ResolvedConstraints.resolve(
            entry,
            types,
            None,
            list(range(len(types))),
            cell=cell,
            pbc=tuple(bool(p) for p in ni_orthorhombic.pbc),
        )
        natoms = int(engine.lmp.get_natoms())
        width = lengths[0] / comm.Get_size()
        try:
            for k in (1, 2, 3):
                request = entry.copy()
                request[:, 0] += k * 0.75 * width
                out = engine.minimize_with_results(
                    positions=request, types=types, constraints=payload
                )
                assert int(engine.lmp.get_natoms()) == natoms
                assert self._owned(engine, comm) == natoms
                result = out[0] if comm.Get_rank() == 0 else None
                dev = comm.bcast(
                    None if result is None else _max_mic(result, request, lengths),
                    root=0,
                )
                assert dev < 1e-6
        finally:
            engine.close()
