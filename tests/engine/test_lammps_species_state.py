"""Species/type/mass rule and full-system ownership of ``LammpsEngine``.

Covers the engine side of the active-volume contract: the single
``species_map`` / ``types_to_int`` rule, the remembered ``full_system``,
``system_is_cropped`` / ``ensure_full_system``, the restore of
``partn_search`` / ``partn_refine`` under active volume on every exit path
(and the original exception surviving a failed restore), explicit rejection
of unsupported geometry/constraint combinations before any state changes,
the finite/shape guard in ``set_positions`` and on the active-volume path,
and ``full_system.masses`` following the potential.

Serial tests skip under ``mpirun``; the MPI class skips without it and is
meant for ``mpirun -n 4``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("lammps")

from pykmc.activevolume import active_volume as av  # noqa: E402
from pykmc.engine import lammps as lammps_module  # noqa: E402
from pykmc.engine.lammps import (  # noqa: E402
    FullSystem,
    LammpsEngine,
    species_map,
    types_to_int,
)
from pykmc.result import ErrorType  # noqa: E402
from pykmc.system import System  # noqa: E402


@dataclass
class _LammpsCfg:
    """``lj/cut`` sets no masses, so every mass LAMMPS holds came from pyKMC."""

    pair_style: str = "lj/cut 6.0"
    pair_coeff: str = "* * 0.52 2.274"
    min_style: str = "cg"
    minimize: str = "1e-6 1e-8 1000 10000"
    frz_min: str = "1e-4 1e-6 100 1000"
    verbosity: int = 0


def _av_config(
    lammps_cfg: _LammpsCfg,
    active_volume: bool = True,
    frozen_atoms: object | None = None,
) -> SimpleNamespace:
    """Minimal config namespace for ``partn_search`` / ``partn_refine``.

    Only the attributes read before pARTn is instantiated are provided; the
    tests inject a failure or a rejection before that point.
    """
    return SimpleNamespace(
        control=SimpleNamespace(active_volume=active_volume),
        activevolume=SimpleNamespace(ract=6.0, rmov=3.0, AV_debug=False),
        lammps=lammps_cfg,
        eventsearch=SimpleNamespace(delr_thr=0.1),
        frozen_atoms=frozen_atoms,
    )


class _InjectedFailure(RuntimeError):
    """Raised by the fake pARTn factory to simulate a failure after the crop."""


def _failing_partn() -> SimpleNamespace:
    """Return a ``pypARTn`` stand-in whose ``artn(...)`` raises ``_InjectedFailure``."""

    def artn(engine: str) -> None:
        raise _InjectedFailure("injected after the active-volume crop")

    return SimpleNamespace(artn=artn)


def _initialize(engine: LammpsEngine, system: object) -> None:
    """Load ``system`` (a conftest ``System``) into ``engine``."""
    engine.initialize_parameters()
    engine.initialize_system(
        types=system.types,
        positions=system.positions,
        cell=system.cell,
        pbc=system.pbc,
    )
    engine.initialize_potential()


def _gathered_types(lmp: object) -> np.ndarray:
    """Integer types of the live instance, in LAMMPS id order."""
    return np.ctypeslib.as_array(lmp.gather_atoms("type", 0, 1)).copy()


def test_species_map_is_alphabetical_with_ase_masses() -> None:
    """``species_map`` returns plain tuples in alphabetical order."""
    from ase.data import atomic_masses, atomic_numbers

    species, masses = species_map(np.array(["Ni", "Cr", "Ni", "Fe"]))
    assert species == ("Cr", "Fe", "Ni")
    assert masses == tuple(float(atomic_masses[atomic_numbers[s]]) for s in species)
    assert all(type(s) is str for s in species)
    assert all(type(m) is float for m in masses)


def test_types_to_int_uses_full_species_order() -> None:
    """A subset of the species keeps the full map's integer types."""
    species, _ = species_map(["Ni", "Cr", "Fe"])
    got = types_to_int(np.array(["Ni", "Ni", "Fe"]), species)
    assert got.dtype == np.int32
    assert got.tolist() == [3, 3, 2]


def test_species_helpers_reject_bad_input() -> None:
    """Unknown symbols and empty systems are ``ValueError``s, not ``KeyError``s."""
    with pytest.raises(ValueError, match="unknown chemical symbol"):
        species_map(["Ni", "Xx"])
    with pytest.raises(ValueError, match="at least one atom"):
        species_map([])
    with pytest.raises(ValueError, match="not in the species map"):
        types_to_int(["Cr"], ("Fe", "Ni"))


class TestLammpsSpeciesStateSerial:
    """Serial real-LAMMPS engine-state tests."""

    @pytest.fixture(autouse=True)
    def require_serial(self) -> None:
        """Skip under ``mpirun``."""
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_size() > 1:
            pytest.skip("serial tests must run without mpirun")

    @pytest.fixture
    def engine(self) -> LammpsEngine:
        """Yield a started serial ``lj/cut`` engine, closed after the test."""
        engine = LammpsEngine(config=_LammpsCfg(), comm=None)
        engine.start()
        yield engine
        engine.close()

    def test_full_system_recorded_at_initialize(
        self, engine: LammpsEngine, system_binary_fcc: System
    ) -> None:
        """``initialize_system`` records species, masses, cell and pbc."""
        system = system_binary_fcc
        cell_in = np.array(system.cell, copy=True)
        _initialize(engine, system)
        fs = engine.full_system
        assert isinstance(fs, FullSystem)
        assert fs.species == ("Fe", "Ni")
        species, masses = species_map(system.types)
        assert fs.masses == masses
        assert fs.pbc == (True, True, True)
        assert fs.natoms == len(system.types)
        assert fs.types == tuple(system.types)
        np.testing.assert_array_equal(fs.cell, cell_in)
        # The recorded cell is a private copy.
        system.cell[0, 0] += 1.0
        np.testing.assert_array_equal(fs.cell, cell_in)
        system.cell[0, 0] -= 1.0
        assert not engine.system_is_cropped

    def test_live_types_and_masses_follow_the_rule(
        self, engine: LammpsEngine, system_binary_fcc: System
    ) -> None:
        """LAMMPS types/masses in the live instance match the helpers."""
        system = system_binary_fcc
        _initialize(engine, system)
        species, masses = species_map(system.types)
        assert engine.lmp.extract_global("ntypes") == len(species)
        np.testing.assert_array_equal(
            _gathered_types(engine.lmp), types_to_int(system.types, species)
        )
        live_mass = engine.lmp.extract_atom("mass")
        for i, mass in enumerate(masses):
            assert live_mass[i + 1] == pytest.approx(mass)

    def test_ensure_full_system_before_initialize_raises(
        self, engine: LammpsEngine
    ) -> None:
        """No remembered system: ``ensure_full_system`` is a ``RuntimeError``."""
        assert not engine.system_is_cropped
        with pytest.raises(RuntimeError, match="initialize_system has not been called"):
            engine.ensure_full_system()

    def test_ensure_full_system_is_noop_when_intact(
        self, engine: LammpsEngine, ni_orthorhombic: System
    ) -> None:
        """An intact engine is not rebuilt and its positions are not touched."""
        system = ni_orthorhombic
        _initialize(engine, system)
        moved = system.positions.copy()
        moved[0, 0] += 0.3
        engine.set_positions(moved)
        assert engine.ensure_full_system(system.positions) is False
        np.testing.assert_allclose(engine.get_positions(), moved, atol=1e-10)

    def test_crop_then_restore_reproduces_fresh_energy(
        self, engine: LammpsEngine, ni_orthorhombic: System
    ) -> None:
        """AV crop, ``ensure_full_system``, energy equals a fresh engine to 1e-8."""
        system = ni_orthorhombic
        _initialize(engine, system)
        config = _av_config(engine.config)
        cell = np.array(system.cell)

        av.partn_search_AV(engine, config, 0, system.positions, cell, system.types)
        assert engine.system_is_cropped
        assert engine.lmp.get_natoms() < len(system.types)

        assert engine.ensure_full_system(system.positions) is True
        assert not engine.system_is_cropped
        assert engine.lmp.get_natoms() == len(system.types)
        assert engine.ensure_full_system(system.positions) is False
        e_restored = engine.get_total_energy()

        fresh = LammpsEngine(config=_LammpsCfg(), comm=None)
        fresh.start()
        try:
            _initialize(fresh, system)
            e_fresh = fresh.get_total_energy()
        finally:
            fresh.close()
        assert e_restored == pytest.approx(e_fresh, abs=1e-8)
        # The restored engine is fully usable for a later full-system operation.
        rng = np.random.default_rng(1)
        perturbed = system.positions + rng.uniform(-0.05, 0.05, system.positions.shape)
        new_positions, e_min = engine.minimize_with_results(positions=perturbed)
        assert new_positions.shape == system.positions.shape
        assert e_min < engine.get_total_energy(positions=perturbed)

    def test_ensure_full_system_cropped_requires_positions(
        self, engine: LammpsEngine, ni_orthorhombic: System
    ) -> None:
        """A cropped engine cannot be rebuilt without full-system positions."""
        system = ni_orthorhombic
        _initialize(engine, system)
        av.partn_search_AV(
            engine,
            _av_config(engine.config),
            0,
            system.positions,
            np.array(system.cell),
            system.types,
        )
        with pytest.raises(ValueError, match="positions are required"):
            engine.ensure_full_system()
        with pytest.raises(ValueError, match="expected"):
            engine.ensure_full_system(system.positions[:-1])
        assert engine.ensure_full_system(system.positions) is True

    def test_command_clear_marks_the_engine_cropped(
        self, engine: LammpsEngine, ni_orthorhombic: System
    ) -> None:
        """Any ``clear`` through ``command`` is detected even before atoms differ."""
        system = ni_orthorhombic
        _initialize(engine, system)
        e_fresh = engine.get_total_energy()
        engine.command("clear")
        assert engine.system_is_cropped
        assert engine.ensure_full_system(system.positions) is True
        assert engine.get_total_energy() == pytest.approx(e_fresh, abs=1e-8)

    def test_partn_search_failure_after_crop_restores_engine(
        self,
        engine: LammpsEngine,
        ni_orthorhombic: System,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failure raised after the crop still restores the full system."""
        system = ni_orthorhombic
        _initialize(engine, system)
        e_fresh = engine.get_total_energy()
        monkeypatch.setattr(lammps_module, "pypARTn", _failing_partn())
        config = _av_config(engine.config)
        with pytest.raises(_InjectedFailure):
            engine.partn_search(
                config, 0, system.positions.copy(), np.array(system.cell), system.types
            )
        assert engine.lmp is not None, "a Python failure must not close the engine"
        assert not engine.system_is_cropped
        assert engine.lmp.get_natoms() == len(system.types)
        assert engine.get_total_energy(positions=system.positions) == pytest.approx(
            e_fresh, abs=1e-8
        )
        # stdout was handed back (the search redirects it during pARTn).
        print("stdout is restored")

    def test_partn_refine_invalid_saddle_returns_err_and_restores(
        self,
        engine: LammpsEngine,
        ni_orthorhombic: System,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A saddle atom missing from the crop yields ``Err(REFINEMENT_INVALID_MINIMA)``."""
        system = ni_orthorhombic
        _initialize(engine, system)
        e_fresh = engine.get_total_energy()
        monkeypatch.setattr(lammps_module, "pypARTn", _failing_partn())
        config = _av_config(engine.config)
        cell = np.array(system.cell)
        _, d = av.find_mic(system.positions - system.positions[0], cell, pbc=True)
        far_atom = int(np.argmax(d))  # outside ract, hence outside the crop
        result = engine.partn_refine(
            config,
            0,
            positions=system.positions.copy(),
            cell=cell,
            types=system.types,
            saddle_idx=np.array([far_atom]),
            saddle_positions=system.positions[[far_atom]],
        )
        assert not result.is_ok()
        assert result.err_value().type is ErrorType.REFINEMENT_INVALID_MINIMA
        assert "active-volume map" in result.err_value().message
        assert not engine.system_is_cropped
        assert engine.get_total_energy(positions=system.positions) == pytest.approx(
            e_fresh, abs=1e-8
        )

    def test_partn_search_rejects_triclinic_before_any_clear(
        self, engine: LammpsEngine, ni_triclinic: System
    ) -> None:
        """A non-orthorhombic cell under AV raises before the engine is touched."""
        system = ni_triclinic
        _initialize(engine, system)
        e_fresh = engine.get_total_energy()
        config = _av_config(engine.config)
        with pytest.raises(ValueError, match="orthorhombic"):
            engine.partn_search(
                config, 0, system.positions.copy(), np.array(system.cell), system.types
            )
        assert not engine._cleared_since_init
        assert not engine.system_is_cropped
        assert engine.get_total_energy(recompute=False) == pytest.approx(e_fresh)

    def test_partn_search_rejects_frozen_atoms_with_active_volume(
        self, engine: LammpsEngine, ni_orthorhombic: System
    ) -> None:
        """``frozen_atoms`` + active volume is refused before any state change."""
        system = ni_orthorhombic
        _initialize(engine, system)
        config = _av_config(engine.config, frozen_atoms=object())
        with pytest.raises(ValueError, match="frozen_atoms"):
            engine.partn_search(
                config, 0, system.positions.copy(), np.array(system.cell), system.types
            )
        assert not engine.system_is_cropped

    def test_partn_search_requires_full_system_inputs_under_av(
        self, engine: LammpsEngine, ni_orthorhombic: System
    ) -> None:
        """Missing positions/cell/types under AV is a ``ValueError`` before the crop."""
        system = ni_orthorhombic
        _initialize(engine, system)
        config = _av_config(engine.config)
        with pytest.raises(ValueError, match="requires full-system positions"):
            engine.partn_search(config, 0, positions=None, cell=None, types=None)
        assert not engine.system_is_cropped

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_set_positions_rejects_non_finite(
        self, engine: LammpsEngine, ni_orthorhombic: System, bad: float
    ) -> None:
        """A NaN/inf coordinate raises ``ValueError`` and the engine stays usable."""
        system = ni_orthorhombic
        _initialize(engine, system)
        e_fresh = engine.get_total_energy()
        poisoned = system.positions.copy()
        poisoned[3, 1] = bad
        with pytest.raises(ValueError, match="non-finite"):
            engine.set_positions(poisoned)
        with pytest.raises(ValueError, match="non-finite"):
            engine.get_total_energy(positions=poisoned)
        assert engine.lmp is not None
        assert engine.get_total_energy(positions=system.positions) == pytest.approx(
            e_fresh, abs=1e-8
        )

    def test_set_positions_rejects_shape_mismatch(
        self, engine: LammpsEngine, ni_orthorhombic: System
    ) -> None:
        """A too-short or too-long array is refused instead of a silent scatter.

        Regression for the review of the first S4 pass: ``scatter_atoms``
        silently truncated a full-system array on a cropped instance and read
        past the buffer (NaN coordinates) for a short array on a full one.
        """
        system = ni_orthorhombic
        _initialize(engine, system)
        e_fresh = engine.get_total_energy()
        n = len(system.types)
        with pytest.raises(ValueError, match=f"expected \\({n}, 3\\)"):
            engine.set_positions(system.positions[:10])
        with pytest.raises(ValueError, match=f"expected \\({n}, 3\\)"):
            engine.set_positions(np.vstack([system.positions, system.positions[:1]]))
        with pytest.raises(ValueError, match=f"expected \\({n}, 3\\)"):
            engine.set_positions(system.positions.ravel())
        # Positions were never touched.
        assert np.isfinite(engine.get_positions()).all()
        assert engine.get_total_energy() == pytest.approx(e_fresh, abs=1e-8)
        # On a cropped instance a full-system array is refused as well.
        cell = np.array(system.cell)
        av.partn_search_AV(
            engine, _av_config(engine.config), 0, system.positions, cell, system.types
        )
        n_crop = int(engine.lmp.get_natoms())
        assert n_crop < n
        with pytest.raises(ValueError, match=f"expected \\({n_crop}, 3\\)"):
            engine.set_positions(system.positions)
        assert engine.ensure_full_system(system.positions) is True

    def test_partn_refine_nan_saddle_raises_and_restores(
        self,
        engine: LammpsEngine,
        ni_orthorhombic: System,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A NaN saddle position under AV is a ``ValueError`` before any LAMMPS call.

        Regression for the review of the first S4 pass: the guard lived only in
        ``LammpsEngine.set_positions`` and the AV module's own ``set_positions``
        scattered the NaN, giving LAMMPS's per-rank "Non-numeric atom coords"
        error (a hang on a multi-rank engine).
        """
        system = ni_orthorhombic
        _initialize(engine, system)
        e_fresh = engine.get_total_energy()
        monkeypatch.setattr(lammps_module, "pypARTn", _failing_partn())
        config = _av_config(engine.config)
        saddle = system.positions[[1]].copy()
        saddle[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            engine.partn_refine(
                config,
                0,
                positions=system.positions.copy(),
                cell=np.array(system.cell),
                types=system.types,
                saddle_idx=np.array([1]),
                saddle_positions=saddle,
            )
        # The guard fired before the crop: nothing was cleared or rebuilt.
        assert not engine._cleared_since_init
        assert not engine.system_is_cropped
        assert engine.get_total_energy(positions=system.positions) == pytest.approx(
            e_fresh, abs=1e-8
        )

    def test_partn_search_nan_positions_raise_before_crop(
        self, engine: LammpsEngine, ni_orthorhombic: System
    ) -> None:
        """A NaN full-system coordinate under AV raises before ``clear``."""
        system = ni_orthorhombic
        _initialize(engine, system)
        e_fresh = engine.get_total_energy()
        poisoned = system.positions.copy()
        poisoned[2, 2] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            engine.partn_search(
                _av_config(engine.config),
                0,
                poisoned,
                np.array(system.cell),
                system.types,
            )
        assert not engine._cleared_since_init
        assert not engine.system_is_cropped
        assert engine.get_total_energy(recompute=False) == pytest.approx(e_fresh)

    def test_restore_failure_does_not_mask_the_original_exception(
        self,
        engine: LammpsEngine,
        ni_orthorhombic: System,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the restore fails after a search failure, the search failure is raised.

        The restore error is reported as a ``RuntimeWarning`` naming both, and
        ``system_is_cropped`` tells the caller the engine is still a crop.
        """
        system = ni_orthorhombic
        _initialize(engine, system)
        monkeypatch.setattr(lammps_module, "pypARTn", _failing_partn())
        real_ensure = engine.ensure_full_system

        def broken_restore(positions: np.ndarray | None = None) -> bool:
            raise KeyError("restore exploded")

        monkeypatch.setattr(engine, "ensure_full_system", broken_restore)
        with pytest.warns(RuntimeWarning, match="restore exploded") as record:
            with pytest.raises(_InjectedFailure, match="injected after"):
                engine.partn_search(
                    _av_config(engine.config),
                    0,
                    system.positions.copy(),
                    np.array(system.cell),
                    system.types,
                )
        assert any("_InjectedFailure" in str(w.message) for w in record)
        assert engine.system_is_cropped
        # Without a primary failure a restore error propagates on its own
        # (a stubbed search that "succeeds" leaves only the restore to fail).
        monkeypatch.setattr(engine, "_partn_search_impl", lambda *a, **k: None)
        with pytest.raises(KeyError, match="restore exploded"):
            engine.partn_search(
                _av_config(engine.config),
                0,
                system.positions.copy(),
                np.array(system.cell),
                system.types,
            )
        assert real_ensure(system.positions) is True
        assert not engine.system_is_cropped

    def test_full_system_masses_follow_the_potential(self) -> None:
        """``full_system.masses`` are the live per-type masses after ``pair_coeff``."""
        from pathlib import Path

        eam = (
            Path(__file__).resolve().parents[2]
            / "examples"
            / "NiCr_fcc_447at_slab_monovacancy"
            / "Bonny_2013_NiFeCr.eam"
        )
        if not eam.is_file():
            pytest.skip(f"{eam.name} not present (examples/ not shipped)")
        from ase.build import bulk

        atoms = bulk("Ni", crystalstructure="fcc", a=3.524, cubic=True).repeat(2)
        types = atoms.get_chemical_symbols()
        types[0] = "Cr"
        cfg = _LammpsCfg(pair_style="eam/alloy", pair_coeff=f"* * {eam} Cr Ni")
        engine = LammpsEngine(config=cfg, comm=None)
        engine.start()
        try:
            engine.initialize_parameters()
            engine.initialize_system(
                types=types,
                positions=atoms.get_positions(),
                cell=atoms.get_cell(),
                pbc=[True] * 3,
            )
            ase_masses = species_map(types)[1]
            assert engine.full_system.masses == ase_masses  # before pair_coeff
            engine.initialize_potential()
            ptr = engine.lmp.extract_atom("mass")
            live = (float(ptr[1]), float(ptr[2]))  # copy: the pointer dies at clear
            assert engine.full_system.masses == live
            assert engine.full_system.masses != ase_masses  # setfl values differ
            assert engine.full_system.masses == pytest.approx(ase_masses, rel=1e-3)
            # The restore replays them and lands on the same live values.
            positions = atoms.get_positions()
            e_fresh = engine.get_total_energy()
            engine.command("clear")
            assert engine.ensure_full_system(positions) is True
            assert engine.full_system.masses == live
            assert engine.get_total_energy() == pytest.approx(e_fresh, abs=1e-8)
        finally:
            engine.close()


@pytest.mark.mpi
class TestLammpsSpeciesStateMPI:
    """Run with ``mpirun -n 4``; every rank executes each test collectively."""

    @pytest.fixture(autouse=True)
    def require_mpi(self) -> None:
        """Skip without ``mpirun``."""
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_size() == 1:
            pytest.skip("requires mpirun -n N")
        self.comm = MPI.COMM_WORLD
        yield
        MPI.COMM_WORLD.Barrier()

    @property
    def is_rank0(self) -> bool:
        """True on the rank that owns extraction results."""
        return self.comm.Get_rank() == 0

    @pytest.fixture
    def engine(self) -> LammpsEngine:
        """Yield a started 4-rank ``lj/cut`` engine, closed after the test."""
        engine = LammpsEngine(config=_LammpsCfg(), comm=self.comm)
        engine.start()
        yield engine
        engine.close()

    def test_crop_then_restore_reproduces_fresh_energy(
        self, engine: LammpsEngine, ni_orthorhombic: System
    ) -> None:
        """Collective crop + restore; rank 0 compares with a fresh engine."""
        system = ni_orthorhombic
        _initialize(engine, system)
        e_fresh = engine.get_total_energy()
        av.partn_search_AV(
            engine,
            _av_config(engine.config),
            0,
            system.positions,
            np.array(system.cell),
            system.types,
        )
        assert engine.system_is_cropped
        assert engine.ensure_full_system(system.positions) is True
        assert not engine.system_is_cropped
        e_restored = engine.get_total_energy()
        if self.is_rank0:
            assert e_restored == pytest.approx(e_fresh, abs=1e-8)
        else:
            assert e_restored is None

    def test_set_positions_nan_raises_on_every_rank(
        self, engine: LammpsEngine, ni_orthorhombic: System
    ) -> None:
        """Every rank raises symmetrically and the engine remains usable."""
        system = ni_orthorhombic
        _initialize(engine, system)
        e_fresh = engine.get_total_energy()
        poisoned = system.positions.copy()
        poisoned[5, 2] = np.nan
        raised = False
        try:
            engine.set_positions(poisoned)
        except ValueError as exc:
            raised = "non-finite" in str(exc)
        assert raised
        n_raised = self.comm.allreduce(int(raised))
        assert n_raised == self.comm.Get_size()
        e_after = engine.get_total_energy(positions=system.positions)
        if self.is_rank0:
            assert e_after == pytest.approx(e_fresh, abs=1e-8)
        else:
            assert e_after is None

    def _assert_restored(
        self, engine: LammpsEngine, system: System, e_fresh: float | None
    ) -> None:
        """Every rank sees an intact full system; rank 0 checks the energy."""
        assert not engine.system_is_cropped
        assert int(engine.lmp.get_natoms()) == len(system.types)
        e_after = engine.get_total_energy(positions=system.positions)
        if self.is_rank0:
            assert e_after == pytest.approx(e_fresh, abs=1e-8)
        else:
            assert e_after is None

    def test_partn_search_failure_after_crop_restores_on_every_rank(
        self,
        engine: LammpsEngine,
        ni_orthorhombic: System,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failure after the collective crop is raised and restored on all ranks."""
        system = ni_orthorhombic
        _initialize(engine, system)
        e_fresh = engine.get_total_energy()
        monkeypatch.setattr(lammps_module, "pypARTn", _failing_partn())
        raised = False
        try:
            engine.partn_search(
                _av_config(engine.config),
                0,
                system.positions.copy(),
                np.array(system.cell),
                system.types,
            )
        except _InjectedFailure:
            raised = True
        assert self.comm.allreduce(int(raised)) == self.comm.Get_size()
        assert engine.lmp is not None
        self._assert_restored(engine, system, e_fresh)

    def test_partn_refine_invalid_saddle_is_symmetric_across_ranks(
        self,
        engine: LammpsEngine,
        ni_orthorhombic: System,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A saddle atom missing from the crop: Err on rank 0, None elsewhere, restored."""
        system = ni_orthorhombic
        _initialize(engine, system)
        e_fresh = engine.get_total_energy()
        monkeypatch.setattr(lammps_module, "pypARTn", _failing_partn())
        cell = np.array(system.cell)
        _, d = av.find_mic(system.positions - system.positions[0], cell, pbc=True)
        far_atom = int(np.argmax(d))
        result = engine.partn_refine(
            _av_config(engine.config),
            0,
            positions=system.positions.copy(),
            cell=cell,
            types=system.types,
            saddle_idx=np.array([far_atom]),
            saddle_positions=system.positions[[far_atom]],
        )
        if self.is_rank0:
            is_err = (
                result is not None
                and not result.is_ok()
                and result.err_value().type is ErrorType.REFINEMENT_INVALID_MINIMA
            )
        else:
            is_err = result is None
        # Every rank must reach the same verdict; a lone False on any rank
        # (an Err on a non-root rank, or a None on root) fails the allreduce.
        assert self.comm.allreduce(int(is_err)) == self.comm.Get_size()
        self._assert_restored(engine, system, e_fresh)

    def test_partn_refine_nan_saddle_raises_on_every_rank(
        self,
        engine: LammpsEngine,
        ni_orthorhombic: System,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A NaN saddle position raises symmetrically before any collective call.

        Regression for the review of the first S4 pass: the unguarded AV
        ``set_positions`` let LAMMPS raise on the owning rank only and the
        4-rank engine hung.
        """
        system = ni_orthorhombic
        _initialize(engine, system)
        e_fresh = engine.get_total_energy()
        monkeypatch.setattr(lammps_module, "pypARTn", _failing_partn())
        saddle = system.positions[[1]].copy()
        saddle[0, 0] = np.nan
        raised = False
        try:
            engine.partn_refine(
                _av_config(engine.config),
                0,
                positions=system.positions.copy(),
                cell=np.array(system.cell),
                types=system.types,
                saddle_idx=np.array([1]),
                saddle_positions=saddle,
            )
        except ValueError as exc:
            raised = "non-finite" in str(exc)
        assert self.comm.allreduce(int(raised)) == self.comm.Get_size()
        assert not engine._cleared_since_init
        self._assert_restored(engine, system, e_fresh)

    def test_partn_search_rejects_triclinic_on_every_rank(
        self, engine: LammpsEngine, ni_triclinic: System
    ) -> None:
        """The triclinic rejection under AV fires on every rank before any clear."""
        system = ni_triclinic
        _initialize(engine, system)
        e_fresh = engine.get_total_energy()
        raised = False
        try:
            engine.partn_search(
                _av_config(engine.config),
                0,
                system.positions.copy(),
                np.array(system.cell),
                system.types,
            )
        except ValueError as exc:
            raised = "orthorhombic" in str(exc)
        assert self.comm.allreduce(int(raised)) == self.comm.Get_size()
        assert not engine._cleared_since_init
        self._assert_restored(engine, system, e_fresh)
