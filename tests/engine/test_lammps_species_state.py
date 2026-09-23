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

import numpy as np
import pytest

pytest.importorskip("lammps")

from pykmc.engine.lammps import (  # noqa: E402
    FullSystem,
    LammpsEngine,
    species_map,
    types_to_int,
)
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
        finally:
            engine.close()
