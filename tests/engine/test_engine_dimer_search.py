"""The dimer search worker operation on a dedicated serial LAMMPS instance.

Serial: pytest tests/engine/test_engine_dimer_search.py
MPI:    mpirun -n 3 pytest tests/engine/test_engine_dimer_search.py
        (one worker of two ranks; any larger -n gives one wider worker)

The system is an fcc Ni vacancy, 4x4x4 cubic cells minus one atom, shifted by
0.3 Å so that no atom sits on a cell face. The central atom is a nearest
neighbour of the vacancy across the x and y boundaries, so its hop into the
vacancy crosses both. With this potential that hop is 1.185 eV.
"""

import os
from dataclasses import dataclass, field

import numpy as np
import pytest

lammps = pytest.importorskip("lammps")

from ase.cell import Cell  # noqa: E402
from mpi4py import MPI  # noqa: E402

from pykmc.dimer import DimerResult  # noqa: E402
from pykmc.engine import Engine  # noqa: E402
from pykmc.engine.dimer_search import DimerSearchExtension  # noqa: E402
from pykmc.engine.lammps import LammpsEngine  # noqa: E402

POTENTIAL = os.path.join(
    os.path.dirname(__file__), "..", "data", "Ni_v6_2.0_LKBeland2016.eam"
)
SHIFT = 0.3
# The local start draws on the central atom only, so the search hops it.
OPTIONS = {"start": "local", "start_radius": 1.0}
HOP_BARRIER = 1.185
# eventsearch.delr_thr default: a minimum this close to the start is the start.
DELR_THR = 0.5
# Atoms beyond this distance of the central atom are frozen in the mask tests.
FROZEN_BEYOND = 7.0


@dataclass
class _LammpsConfig:
    pair_style: str = "eam/alloy"
    pair_coeff: list[str] = field(default_factory=lambda: [f"* * {POTENTIAL} Ni"])
    min_style: str = "cg"
    minimize: str = "0 1e-6 10000 100000"
    frz_min: str = "1e-4 1e-6 100 1000"
    verbosity: int = 0


@dataclass
class _Cell:
    positions: np.ndarray
    types: list[str]
    cell: Cell
    pbc: np.ndarray
    central: int


def _minimum_image(d: np.ndarray, cell: np.ndarray) -> np.ndarray:
    lengths = np.diag(cell)
    return d - np.round(d / lengths) * lengths


@pytest.fixture(scope="module")
def eam_config():
    lmp = lammps.lammps(comm=MPI.COMM_SELF, cmdargs=["-log", "none", "-screen", "none"])
    has_eam = lmp.has_style("pair", "eam/alloy")
    lmp.close()
    if not has_eam:
        pytest.skip("LAMMPS built without pair_style eam/alloy (MANYBODY)")
    return _LammpsConfig()


@pytest.fixture(scope="module")
def vacancy(eam_config) -> _Cell:
    """The minimised Ni vacancy cell, the same on every rank."""
    from ase.build import bulk

    atoms = bulk("Ni", crystalstructure="fcc", a=3.524, cubic=True).repeat([4, 4, 4])
    cell = atoms.get_cell()
    positions = np.delete(atoms.get_positions() + SHIFT, 0, axis=0)
    types = ["Ni"] * len(positions)
    pbc = np.array([True, True, True])
    engine = _make_engine(eam_config, MPI.COMM_SELF, positions, types, cell, pbc)
    minimum, _ = engine.minimize_with_results()
    engine.close()
    lengths = np.diag(cell)
    target = np.array([lengths[0] - 1.762 + SHIFT, lengths[1] - 1.762 + SHIFT, SHIFT])
    central = int(np.argmin(np.linalg.norm(minimum - target, axis=1)))
    return _Cell(minimum, types, cell, pbc, central)


def _make_engine(config, comm, positions, types, cell, pbc, extension=None) -> Engine:
    engine = Engine.create("lammps", config=config, comm=comm)
    if extension is not None:
        extension(engine)
    engine.start()
    engine.initialize_parameters()
    engine.initialize_system(types=types, positions=positions, cell=cell, pbc=pbc)
    engine.initialize_potential()
    return engine


def _close(engine: Engine) -> None:
    """Close the engine and the extension's instance, if it was built."""
    instance = engine._extensions["DimerSearchExtension"]._force_engine
    if instance is not None:
        instance.close()
    engine.close()


def _search(engine: Engine, system: _Cell, seed: int = 1, **kwargs) -> DimerResult:
    request = dict(
        positions=system.positions,
        types=system.types,
        cell=system.cell,
        pbc=system.pbc,
        movable=np.arange(len(system.positions)),
        central_atom=system.central,
        seed=seed,
        options=OPTIONS,
    )
    request.update(kwargs)
    return engine.dimer_search(**request)


def _assert_same(a: DimerResult, b: DimerResult, atol: float) -> None:
    """Every field equal, arrays and floats to ``atol``."""
    for name, value in a._asdict().items():
        other = getattr(b, name)
        if isinstance(value, np.ndarray):
            np.testing.assert_allclose(value, other, rtol=0, atol=atol, err_msg=name)
        elif isinstance(value, float):
            assert value == pytest.approx(other, abs=atol), name
        else:
            assert value == other, name


def _frozen_movable(cell: _Cell) -> np.ndarray:
    offsets = _minimum_image(cell.positions - cell.positions[cell.central], cell.cell)
    return np.flatnonzero(np.linalg.norm(offsets, axis=1) <= FROZEN_BEYOND)


class TestDimerSearchSerial:
    @pytest.fixture(autouse=True)
    def require_serial(self):
        if MPI.COMM_WORLD.Get_size() > 1:
            pytest.skip("serial tests must run without mpirun")

    @pytest.fixture
    def engine(self, eam_config, vacancy):
        engine = _make_engine(
            eam_config,
            None,
            vacancy.positions,
            vacancy.types,
            vacancy.cell,
            vacancy.pbc,
            DimerSearchExtension,
        )
        yield engine
        _close(engine)

    def test_one_search_returns_the_raw_output(self, engine, vacancy):
        """The op finds the hop and returns every field of the core's result."""
        result = _search(engine, vacancy, seed=1)
        n_atoms = len(vacancy.positions)

        assert isinstance(result, DimerResult)
        assert result.success and result.reason is None
        assert result.seed == 1
        assert result.e_saddle - result.e_start == pytest.approx(HOP_BARRIER, abs=0.01)
        assert result.curvature < 0
        assert result.delr1 < DELR_THR
        for name in ("saddle_positions", "min1_positions", "min2_positions", "mode"):
            assert getattr(result, name).shape == (n_atoms, 3), name
        for name in ("e_start", "e_saddle", "e_min1", "e_min2", "delr2"):
            assert np.isfinite(getattr(result, name)), name
        assert 0 < result.force_calls_saddle < result.force_calls_total

    def test_instance_is_built_on_first_use_and_reused(self, engine, vacancy):
        """One serial instance, built by the first search, reused by the next."""
        extension = engine._extensions["DimerSearchExtension"]
        assert extension._force_engine is None

        first = _search(engine, vacancy, seed=2)
        instance = extension._force_engine
        assert instance is not None and instance is not engine
        assert instance.comm is MPI.COMM_SELF
        assert instance.lmp.extract_setting("world_size") == 1

        again = _search(engine, vacancy, seed=2)
        assert extension._force_engine is instance
        # A reused instance repeats the fresh one's search exactly.
        _assert_same(first, again, atol=0.0)

    def test_reused_instance_matches_a_fresh_one(self, eam_config, engine, vacancy):
        """After other searches, the same seed gives what a new instance gives."""
        _search(engine, vacancy, seed=5)
        reused = _search(engine, vacancy, seed=3)
        fresh_engine = _make_engine(
            eam_config,
            None,
            vacancy.positions,
            vacancy.types,
            vacancy.cell,
            vacancy.pbc,
            DimerSearchExtension,
        )
        fresh = _search(fresh_engine, vacancy, seed=3)
        _close(fresh_engine)
        _assert_same(reused, fresh, atol=0.0)

    def test_worker_engine_is_left_as_it_was(self, engine, vacancy):
        """Positions and energy of the worker's own engine do not change."""
        moved = vacancy.positions.copy()
        moved[vacancy.central] += [0.05, -0.03, 0.02]
        engine.set_positions(moved)
        positions_before = engine.get_positions()
        energy_before = engine.get_potential_energy()

        _search(engine, vacancy)

        np.testing.assert_array_equal(engine.get_positions(), positions_before)
        assert engine.get_potential_energy(recompute=False) == energy_before
        assert engine.get_potential_energy() == energy_before

    def test_frozen_atoms_stay_and_lammps_reports_zero_force(
        self, engine, vacancy, monkeypatch
    ):
        """Frozen atoms never move, and every force LAMMPS returns is zero on them."""
        movable = _frozen_movable(vacancy)
        frozen = np.setdiff1d(np.arange(len(vacancy.positions)), movable)
        assert 0 < len(movable) < len(vacancy.positions)

        reported = []
        original = LammpsEngine.get_energy_and_forces

        def record(self, positions=None):
            energy, forces = original(self, positions)
            if self is not engine:
                reported.append(forces[frozen].copy())
            return energy, forces

        monkeypatch.setattr(LammpsEngine, "get_energy_and_forces", record)
        result = _search(engine, vacancy, movable=movable)
        monkeypatch.undo()

        assert result.success
        assert len(reported) == result.force_calls_total
        assert all(np.all(forces == 0.0) for forces in reported)
        for name in ("saddle_positions", "min1_positions", "min2_positions"):
            positions = getattr(result, name)
            np.testing.assert_array_equal(
                positions[frozen], vacancy.positions[frozen], err_msg=name
            )
        np.testing.assert_array_equal(result.mode[frozen], 0.0)

        # The fix and group are gone, and without them frozen atoms next to the
        # saddle feel a force: the zeros above came from the fix.
        instance = engine._extensions["DimerSearchExtension"]._force_engine
        assert not instance.lmp.has_id("fix", "f_dimer_frozen")
        assert not instance.lmp.has_id("group", "g_dimer_frozen")
        _, forces = instance.get_energy_and_forces(result.saddle_positions)
        assert np.abs(forces[frozen]).max() > 1e-3

    def test_atom_order_and_periodic_wrapping(self, engine, vacancy, monkeypatch):
        """Results are in pyKMC atom order and wrapped on the periodic axes only."""
        lengths = np.diag(vacancy.cell)
        seen = []
        original = LammpsEngine.get_energy_and_forces

        def record(self, positions=None):
            if self is not engine:
                seen.append(positions.max(axis=0))
            return original(self, positions)

        monkeypatch.setattr(LammpsEngine, "get_energy_and_forces", record)
        result = _search(engine, vacancy, seed=1)
        monkeypatch.undo()

        # The search crossed the x and y boundaries in its own coordinates...
        assert np.max(seen, axis=0)[0] > lengths[0]
        assert np.max(seen, axis=0)[1] > lengths[1]
        # ...and its results are wrapped back into the cell.
        for name in ("saddle_positions", "min1_positions", "min2_positions"):
            positions = getattr(result, name)
            assert np.all(positions >= 0) and np.all(positions < lengths), name
        # The central atom hopped into the vacancy at (0.3, 0.3, 0.3).
        hop = _minimum_image(result.min2_positions - vacancy.positions, vacancy.cell)
        assert int(np.argmax(np.linalg.norm(hop, axis=1))) == vacancy.central
        np.testing.assert_allclose(
            result.min2_positions[vacancy.central], [SHIFT] * 3, atol=0.05
        )

        # The same search on shuffled atoms returns the same result, shuffled.
        order = np.random.default_rng(11).permutation(len(vacancy.positions))
        rank = np.argsort(order)
        shuffled = _search(
            engine,
            _Cell(
                vacancy.positions[order],
                [vacancy.types[i] for i in order],
                vacancy.cell,
                vacancy.pbc,
                int(rank[vacancy.central]),
            ),
            seed=1,
        )
        assert shuffled.success
        for name in ("saddle_positions", "min1_positions", "min2_positions", "mode"):
            np.testing.assert_allclose(
                getattr(shuffled, name)[rank],
                getattr(result, name),
                rtol=0,
                atol=1e-6,
                err_msg=name,
            )
        assert shuffled.e_saddle == pytest.approx(result.e_saddle, abs=1e-8)

    def test_new_system_resets_the_instance(self, engine, vacancy):
        """Another cell and periodicity reset the same instance with clear."""
        before = _search(engine, vacancy, seed=4)
        extension = engine._extensions["DimerSearchExtension"]
        instance = extension._force_engine

        # A slab: the bulk cell opened along z, periodic in x and y only.
        slab = np.array(vacancy.cell)
        slab[2, 2] += 10.0
        result = _search(
            engine,
            vacancy,
            cell=slab,
            pbc=np.array([True, True, False]),
            options={**OPTIONS, "max_translations": 3},
        )
        assert result.reason == "max_iterations"
        assert extension._force_engine is instance
        boxlo, boxhi, *_, periodicity, _ = instance.lmp.extract_box()
        assert list(periodicity) == [1, 1, 0]
        assert boxhi[2] - boxlo[2] == pytest.approx(slab[2, 2])

        # Back on the bulk cell, the search is the one done before.
        _assert_same(_search(engine, vacancy, seed=4), before, atol=0.0)
        assert extension._force_engine is instance

    def test_failed_search_returns_a_reason_and_force_calls(self, engine, vacancy):
        """A search that does not converge reports why and how many calls it took."""
        result = _search(engine, vacancy, options={**OPTIONS, "max_translations": 2})
        assert not result.success
        assert result.reason == "max_iterations"
        assert result.saddle_positions is None
        assert result.force_calls_saddle == result.force_calls_total > 0

    def test_bad_requests_raise(self, engine, vacancy):
        """A malformed request is a programming error, not a failed search."""
        with pytest.raises(ValueError, match="movable"):
            _search(engine, vacancy, movable=[0, len(vacancy.positions)])
        with pytest.raises(ValueError, match="movable"):
            _search(engine, vacancy, movable=[-1])
        with pytest.raises(ValueError, match="integer"):
            _search(engine, vacancy, movable=np.ones(len(vacancy.positions), bool))
        with pytest.raises(ValueError, match="start_radius"):
            _search(engine, vacancy, options={"start": "local"})

    def test_lammps_error_is_not_caught(self, engine, vacancy):
        """A LAMMPS-fatal error raises out of the op; the next search rebuilds."""
        _search(engine, vacancy, options={**OPTIONS, "max_translations": 1})
        extension = engine._extensions["DimerSearchExtension"]
        broken = vacancy.positions.copy()
        broken[vacancy.central] = np.nan
        with pytest.raises(Exception, match="Non-numeric atom coords"):
            _search(engine, vacancy, positions=broken, movable=_frozen_movable(vacancy))
        assert extension._force_engine is None

        result = _search(engine, vacancy, seed=1)
        assert result.success
        instance = extension._force_engine
        assert not instance.lmp.has_id("group", "g_dimer_frozen")


class _ProbedDimerSearch(DimerSearchExtension):
    """The extension plus one op that reports each worker rank's instance."""

    def dimer_probe(self) -> list | None:
        instance = self._force_engine
        if instance is None:
            state = None
        else:
            if not hasattr(self, "_first_instance"):
                self._first_instance = instance
            state = {
                "serial": instance.comm is MPI.COMM_SELF
                and instance.lmp.extract_setting("world_size") == 1,
                "first": instance is self._first_instance,
            }
        return self.engine.comm.gather(state, root=0)


@pytest.mark.mpi
class TestDimerSearchWorker:
    """
    Run with: mpirun -n 3 pytest tests/engine/test_engine_dimer_search.py

    One worker spanning every rank but the manager's, driven through its own
    session.
    """

    @pytest.fixture(autouse=True)
    def require_mpi(self):
        if MPI.COMM_WORLD.Get_size() < 3:
            pytest.skip("requires mpirun -n 3 or more")
        yield
        MPI.COMM_WORLD.Barrier()

    def test_worker_op_through_its_own_session(self, eam_config, vacancy):
        from pykmc.factory import EngineManagerFactory

        world = MPI.COMM_WORLD
        manager = EngineManagerFactory(
            engine_style="lammps",
            engine_config=eam_config,
            n_workers=1,
            comm=world,
            engine_extensions=[_ProbedDimerSearch],
        ).launch()
        if manager is None:  # worker ranks return here after shutdown
            return
        try:
            manager.broadcast("start")
            manager.broadcast("initialize_parameters")
            manager.broadcast(
                "initialize_system",
                types=vacancy.types,
                positions=vacancy.positions,
                cell=vacancy.cell,
                pbc=vacancy.pbc,
            )
            manager.broadcast("initialize_potential")

            ranks = world.Get_size() - 1
            assert manager.dimer_probe().result() == [None] * ranks

            positions_before = manager.get_positions().result()
            energy_before = manager.get_potential_energy().result()
            request = dict(
                positions=vacancy.positions,
                types=vacancy.types,
                cell=vacancy.cell,
                pbc=vacancy.pbc,
                movable=_frozen_movable(vacancy),
                central_atom=vacancy.central,
                seed=6,
                options=OPTIONS,
            )

            first = manager.dimer_search(**request).result()
            assert first.success
            probe = manager.dimer_probe().result()
            assert probe[0] == {"serial": True, "first": True}
            assert probe[1:] == [None] * (ranks - 1)

            again = manager.dimer_search(**request).result()
            assert manager.dimer_probe().result()[0] == {"serial": True, "first": True}
            _assert_same(first, again, atol=0.0)

            np.testing.assert_array_equal(
                manager.get_positions().result(), positions_before
            )
            assert manager.get_potential_energy().result() == energy_before
        finally:
            manager.shutdown()

        # The worker's search equals one run on a serial engine here.
        reference_engine = _make_engine(
            eam_config,
            MPI.COMM_SELF,
            vacancy.positions,
            vacancy.types,
            vacancy.cell,
            vacancy.pbc,
            DimerSearchExtension,
        )
        reference = reference_engine.dimer_search(**request)
        _close(reference_engine)
        _assert_same(first, reference, atol=0.0)
