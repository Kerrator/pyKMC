"""Tests for the multi-species, pbc-aware active-volume ``reset()`` and crop.

``reset()`` used to hard-code ``create_box 1 box``, ``boundary p p p`` and
emit no ``mass``: every active-volume (AV) crop rebuilt the LAMMPS instance as
a one-type, mass-less, fully periodic box. A multi-element ``pair_coeff``
(``eam/alloy ... Cr Ni``) was rejected inside ``reset()``, pair styles that do
not set masses themselves (``lj/cut``, ``sw``, ``mlip``) aborted at the crop's
first rebalance with "Not all per-type masses are set", and a slab crop was
silently periodic in the vacuum direction.

The fake-engine tests pin the emitted command stream and need no native
LAMMPS (``map_types`` delegates to ``pykmc.engine.lammps``, so the tests that
call it need the ``lammps`` Python package importable; nothing here needs
``pypARTn``). The real-LAMMPS tests (serial, no pARTn search) drive the actual
``partn_search_AV`` / ``partn_refine_AV`` entry points on Ni, NiCr and Si
cells and check the engine state, not only the command intent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase.build import bulk
from ase.cell import Cell
from ase.data import atomic_masses, atomic_numbers

from pykmc.activevolume import active_volume as av

_ROOT = Path(__file__).resolve().parents[1]
_CELL = np.diag([20.0, 20.0, 50.0])
_EAM_FILE = (
    _ROOT / "examples" / "NiCr_fcc_447at_slab_monovacancy" / "Bonny_2013_NiFeCr.eam"
)
_SI_SW = _ROOT / "examples" / "Si_vac" / "Si.sw"


def _mass(symbol: str) -> float:
    """ASE standard atomic mass of ``symbol`` (what the engine gives LAMMPS)."""
    return float(atomic_masses[atomic_numbers[symbol]])


# ---------------------------------------------------------------------------
# Fake engine (command stream only)
# ---------------------------------------------------------------------------


class _FakeLammps:
    """Enough of the ``lammps`` object for the crop helpers to run."""

    def __init__(self) -> None:
        self.natoms = 0
        self.last_scatter: np.ndarray | None = None

    def create_atoms(self, n: int, ids: object, types: object, x: object) -> None:
        """Record the atom count."""
        self.natoms = int(n)

    def get_natoms(self) -> int:
        """Atom count of the live (fake) instance."""
        return self.natoms

    def scatter_atoms(self, name: str, dtype: int, count: int, data: object) -> None:
        """Record the scattered positions as an ``(n, 3)`` array."""
        self.last_scatter = np.array(list(data), dtype=float).reshape(-1, 3)

    def extract_compute(self, *args: object) -> float:
        """Return a dummy energy."""
        return -1.0


class _FakeEngine:
    """Record every LAMMPS command string; optionally fail on a prefix."""

    def __init__(
        self,
        fail_on: str | None = None,
        full_system: object | None = None,
    ) -> None:
        self.commands: list[str] = []
        self.fail_on = fail_on
        self.lmp = _FakeLammps()
        self.rank = 0
        if full_system is not None:
            self.full_system = full_system

    def command(self, cmd: str) -> None:
        """Store ``cmd`` verbatim, raising first if it matches ``fail_on``."""
        self.commands.append(cmd)
        if self.fail_on is not None and cmd.startswith(self.fail_on):
            raise RuntimeError(f"injected failure on {cmd!r}")


@dataclass
class _EamCfg:
    """LAMMPS config shim for the fake engine (strings are never parsed)."""

    pair_style: str = "eam/alloy"
    pair_coeff: str = "* * Bonny_2013_NiFeCr.eam Cr Ni"
    min_style: str = "cg"
    minimize: str = "1e-6 1e-8 1000 10000"
    frz_min: str = "1e-4 1e-6 100 1000"


@dataclass
class _AVCfg:
    """Active-volume radii sized for a 4x4x4 FCC cell (14.1 Angstrom)."""

    ract: float = 6.0
    rmov: float = 3.0
    AV_debug: bool = False


@dataclass
class _ResetCfg:
    """Config shim carrying what ``reset()`` and the crop helpers read."""

    lammps: _EamCfg = field(default_factory=_EamCfg)
    activevolume: _AVCfg = field(default_factory=_AVCfg)


def _index_of(commands: list[str], prefix: str) -> int:
    """Return the index of the first command that starts with ``prefix``."""
    return next(i for i, c in enumerate(commands) if c.startswith(prefix))


def _nicr_map() -> dict[str, av.TypeEntry]:
    """Hand-built two-species map (no engine import needed)."""
    return {
        "Cr": {"ref": 1, "mass": _mass("Cr")},
        "Ni": {"ref": 2, "mass": _mass("Ni")},
    }


def test_map_types_matches_engine_rule() -> None:
    """``map_types`` follows the engine's alphabetical ``sorted(set(types))`` rule."""
    pytest.importorskip("lammps")
    int_types, map_type = av.map_types(["Ni", "Cr", "Ni", "Fe"])
    assert list(map_type) == ["Cr", "Fe", "Ni"]
    assert [map_type[k]["ref"] for k in map_type] == [1, 2, 3]
    assert map_type["Ni"]["mass"] == _mass("Ni")
    assert int_types.tolist() == [3, 1, 3, 2]


def test_map_types_delegates_to_engine_helpers() -> None:
    """``map_types`` and the engine's ``species_map``/``types_to_int`` agree."""
    pytest.importorskip("lammps")
    from pykmc.engine.lammps import species_map, types_to_int

    types = np.array(["Ni", "Cr", "Ni", "Fe", "Cr"])
    int_types, map_type = av.map_types(types)
    species, masses = species_map(types)
    assert tuple(map_type) == species
    assert tuple(e["mass"] for e in map_type.values()) == masses
    np.testing.assert_array_equal(int_types, types_to_int(types, species))


def test_reset_sets_one_mass_per_species() -> None:
    """``reset()`` emits ``create_box N`` and one ``mass`` per species (ASE values)."""
    engine = _FakeEngine()
    av.reset(engine, _ResetCfg(), _CELL, _nicr_map())
    assert "create_box 2 box" in engine.commands
    masses = [c for c in engine.commands if c.startswith("mass ")]
    assert masses == [f"mass 1 {_mass('Cr')}", f"mass 2 {_mass('Ni')}"]


def test_reset_orders_box_and_masses_before_potential() -> None:
    """``create_box N`` and ``mass`` precede ``pair_style``/``pair_coeff``."""
    engine = _FakeEngine()
    av.reset(engine, _ResetCfg(), _CELL, _nicr_map())
    i_box = engine.commands.index("create_box 2 box")
    i_mass = _index_of(engine.commands, "mass ")
    i_style = _index_of(engine.commands, "pair_style")
    i_coeff = _index_of(engine.commands, "pair_coeff")
    assert i_box < i_mass < i_style < i_coeff


def test_reset_propagates_explicit_pbc() -> None:
    """An explicit ``pbc`` becomes the crop's ``boundary``."""
    engine = _FakeEngine()
    av.reset(engine, _ResetCfg(), _CELL, _nicr_map(), pbc=(True, True, False))
    assert "boundary p p f" in engine.commands
    assert "boundary p p p" not in engine.commands


def test_reset_reads_pbc_from_engine_full_system() -> None:
    """Without ``pbc``, ``reset()`` uses the engine's remembered full system."""
    engine = _FakeEngine(full_system=SimpleNamespace(pbc=(False, True, True)))
    av.reset(engine, _ResetCfg(), _CELL, _nicr_map())
    assert "boundary f p p" in engine.commands


def test_reset_defaults_to_periodic_without_full_system() -> None:
    """An engine with no remembered system keeps the historical ``p p p``."""
    engine = _FakeEngine()
    av.reset(engine, _ResetCfg(), _CELL, _nicr_map())
    assert "boundary p p p" in engine.commands


def test_reset_rejects_non_orthorhombic_cell_before_clear() -> None:
    """A triclinic cell raises ``ValueError`` and no command (no ``clear``) is sent."""
    engine = _FakeEngine()
    cell = np.array(bulk("Ni", "fcc", a=3.524, cubic=False).get_cell())
    with pytest.raises(ValueError, match="orthorhombic"):
        av.reset(engine, _ResetCfg(), cell, _nicr_map())
    assert engine.commands == []


def test_require_orthorhombic_cell_accepts_cell_object() -> None:
    """``require_orthorhombic_cell`` accepts arrays and ``ase.cell.Cell``."""
    av.require_orthorhombic_cell(_CELL, "t")
    av.require_orthorhombic_cell(Cell(_CELL), "t")
    with pytest.raises(ValueError, match="orthorhombic"):
        av.require_orthorhombic_cell(
            np.array([[10, 0, 0], [5, 10, 0], [0, 0, 10]]), "t"
        )


def test_make_av_cleans_up_when_first_run_fails() -> None:
    """A failing ``run 0`` in ``make_AV`` removes the buffer fix and group."""
    engine = _FakeEngine(fail_on="run 0 post no")
    with pytest.raises(RuntimeError, match="injected"):
        av.make_AV(engine, np.array([0, 1, 2]), np.array([2]))
    i_run = _index_of(engine.commands, "run 0 post no")
    assert engine.commands[i_run + 1 :] == ["unfix f_buffer", "group buffer delete"]


def test_redefine_atoms_unfixes_when_run_fails() -> None:
    """``fix 1`` from ``redefine_atoms`` is removed when ``run 0`` fails."""
    engine = _FakeEngine(fail_on="run 0")
    with pytest.raises(RuntimeError, match="injected"):
        av.redefine_atoms(engine, np.zeros((3, 3)), [1, 1, 1])
    assert engine.commands[-2:] == ["run 0", "unfix 1"]


def test_get_potential_energy_uncomputes_on_failure() -> None:
    """The scratch ``c1`` compute is removed when its ``run 0`` fails."""
    engine = _FakeEngine(fail_on="run 0")
    with pytest.raises(RuntimeError, match="injected"):
        av.get_potential_energy(engine)
    assert engine.commands[-3:] == ["compute c1 all pe", "run 0", "uncompute c1"]


def _fcc_ni_cell(cr_stride: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Build a 4x4x4 FCC Ni cell; every ``cr_stride``-th atom is Cr (0 = none)."""
    atoms = bulk("Ni", crystalstructure="fcc", a=3.524, cubic=True)
    atoms = atoms.repeat([4, 4, 4])
    types = atoms.get_chemical_symbols()
    if cr_stride:
        for i in range(0, len(types), cr_stride):
            types[i] = "Cr"
    return types, atoms.get_positions(), np.array(atoms.get_cell())


def test_partn_refine_av_cleans_core_group_when_minimize_fails() -> None:
    """A failing core minimisation still removes ``f_core``, ``core`` and ``fix 1``."""
    pytest.importorskip("lammps")
    types, positions, cell = _fcc_ni_cell(5)
    engine = _FakeEngine(fail_on="minimize")
    with pytest.raises(RuntimeError, match="injected"):
        av.partn_refine_AV(
            engine,
            _ResetCfg(),
            0,
            positions,
            cell,
            types,
            np.array([0]),
            positions[[0]] + 0.1,
        )
    cmds = engine.commands
    assert "create_box 2 box" in cmds
    i_fix1 = max(i for i, c in enumerate(cmds) if c == "fix 1 all setforce 0.0 0.0 0.0")
    assert "unfix 1" in cmds[i_fix1:]
    i_min = _index_of(cmds, "minimize")
    assert cmds[i_min + 1 :] == ["unfix f_core", "group core delete"]


def _outer_shell_atom(
    positions: np.ndarray, cell: np.ndarray, ract: float
) -> tuple[int, np.ndarray]:
    """Return the crop member farthest from atom 0 and its 0.05 A outward saddle."""
    _, d = av.find_mic(positions - positions[0], cell, pbc=True)
    in_crop = np.where(d <= ract)[0]
    edge = int(in_crop[np.argmax(d[in_crop])])
    u = positions[edge] - positions[0]
    u = u / np.linalg.norm(u)
    return edge, (positions[edge] + 0.05 * u)[None, :]


def test_partn_refine_av_places_saddle_atom_displaced_past_ract() -> None:
    """An in-crop shell atom relaxed 0.05 A past ``ract`` is placed, not rejected.

    Regression for the review of the first S4 pass: a saddle-distance check
    (``> ract``) that the base never had turned this ordinary shell relaxation
    into ``Err(REFINEMENT_INVALID_MINIMA)``, which the basin path treats as
    fatal. Only membership in the crop is checked (the base's ``.item()``
    crash case), and the saddle position is scattered as given.
    """
    pytest.importorskip("lammps")
    types, positions, cell = _fcc_ni_cell(0)
    cfg = _ResetCfg(activevolume=_AVCfg(ract=5.6, rmov=3.0))
    edge, saddle = _outer_shell_atom(positions, cell, cfg.activevolume.ract)
    _, d = av.find_mic(saddle[0] - positions[0], cell, pbc=True)
    assert d > cfg.activevolume.ract, "fixture: the saddle must lie past ract"
    engine = _FakeEngine()
    _, atom_map, _ = av.partn_refine_AV(
        engine, cfg, 0, positions.copy(), cell, types, np.array([edge]), saddle
    )
    crop_index = int(np.where(atom_map == edge)[0][0])
    np.testing.assert_allclose(engine.lmp.last_scatter[crop_index], saddle[0])
    assert f"group core id {crop_index + 1}" in engine.commands
    assert isinstance(av.ActiveVolumeSaddleError("x"), ValueError)


def test_partn_refine_av_rejects_nan_saddle_before_any_command() -> None:
    """A NaN saddle coordinate is a ``ValueError`` before ``clear`` is sent."""
    pytest.importorskip("lammps")
    types, positions, cell = _fcc_ni_cell(0)
    engine = _FakeEngine()
    bad = positions[[0]].copy()
    bad[0, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        av.partn_refine_AV(
            engine, _ResetCfg(), 0, positions, cell, types, np.array([0]), bad
        )
    assert engine.commands == []
    with pytest.raises(ValueError, match="one row per saddle_idx"):
        av.partn_refine_AV(
            engine,
            _ResetCfg(),
            0,
            positions,
            cell,
            types,
            np.array([0, 1]),
            positions[[0]],
        )
    assert engine.commands == []


def test_partn_search_av_rejects_nan_positions_before_any_command() -> None:
    """A NaN full-system coordinate is a ``ValueError`` before ``clear`` is sent."""
    pytest.importorskip("lammps")
    types, positions, cell = _fcc_ni_cell(0)
    engine = _FakeEngine()
    bad = positions.copy()
    bad[7, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        av.partn_search_AV(engine, _ResetCfg(), 0, bad, cell, types)
    assert engine.commands == []


def test_av_set_positions_guards_shape_and_finiteness() -> None:
    """The module-level ``set_positions`` refuses NaN and size-mismatched arrays."""
    pytest.importorskip("lammps")
    engine = _FakeEngine()
    engine.lmp.natoms = 4
    with pytest.raises(ValueError, match="expected \\(4, 3\\)"):
        av.set_positions(engine, np.zeros((5, 3)))
    with pytest.raises(ValueError, match="expected \\(4, 3\\)"):
        av.set_positions(engine, np.zeros((3, 3)))
    poisoned = np.zeros((4, 3))
    poisoned[2, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        av.set_positions(engine, poisoned)
    assert engine.lmp.last_scatter is None
    av.set_positions(engine, np.ones((4, 3)))
    np.testing.assert_array_equal(engine.lmp.last_scatter, np.ones((4, 3)))


def _slab_positions(gap: float) -> tuple[np.ndarray, np.ndarray]:
    """Two FCC Ni(100)-like layers with ``gap`` A of vacuum along z."""
    types, positions, cell = _fcc_ni_cell(0)
    keep = positions[:, 2] < 3.6  # two (100) layers of the 4x4x4 cell
    positions = positions[keep]
    thickness = positions[:, 2].max() - positions[:, 2].min()
    cell = cell.copy()
    cell[2, 2] = thickness + gap
    return positions, cell


def test_define_av_respects_nonperiodic_axis_in_thin_cell() -> None:
    """No fictitious image connects opposite faces of a nonperiodic thin axis."""
    cfg = _ResetCfg()
    cfg.activevolume.ract = 1.3
    cfg.activevolume.rmov = 0.8
    positions = np.array([[5.0, 5.0, 2.2], [5.0, 5.0, 0.2], [5.5, 5.0, 2.2]])
    cell = np.diag([10.0, 10.0, 2.4])
    _, nonperiodic, _ = av.define_AV(cfg, 0, positions, cell, pbc=(True, True, False))
    _, periodic, _ = av.define_AV(cfg, 0, positions, cell, pbc=(True, True, True))
    np.testing.assert_array_equal(nonperiodic, [0, 2])
    np.testing.assert_array_equal(periodic, [0, 1, 2])


def test_define_av_is_silent_for_thick_vacuum_or_periodic_axes() -> None:
    """No warning with enough vacuum, on a periodic axis, or without ``pbc``."""
    import warnings

    cfg = _ResetCfg()  # ract = 6
    thin_positions, thin_cell = _slab_positions(gap=4.0)
    thick_positions, thick_cell = _slab_positions(gap=10.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        av.define_AV(cfg, 0, thick_positions, thick_cell, pbc=(True, True, False))
        av.define_AV(cfg, 0, thin_positions, thin_cell, pbc=(True, True, True))
        av.define_AV(cfg, 0, thin_positions, thin_cell)


def test_partn_search_av_passes_engine_pbc_to_define_av() -> None:
    """The crop uses the same actual-PBC membership as the source resolver."""
    pytest.importorskip("lammps")
    cfg = _ResetCfg()
    cfg.activevolume.ract = 1.3
    cfg.activevolume.rmov = 0.8
    positions = np.array([[5.0, 5.0, 2.2], [5.0, 5.0, 0.2], [5.5, 5.0, 2.2]])
    cell = np.diag([10.0, 10.0, 2.4])
    engine = _FakeEngine(full_system=SimpleNamespace(pbc=(True, True, False)))
    atom_map, _ = av.partn_search_AV(engine, cfg, 0, positions, cell, ["Ni"] * 3)
    np.testing.assert_array_equal(atom_map, [0, 2])
    assert "boundary p p f" in engine.commands


def test_partn_refine_av_rejects_saddle_atom_missing_from_crop() -> None:
    """A saddle atom not in the crop is a saddle error, not a numpy error."""
    pytest.importorskip("lammps")
    types, positions, cell = _fcc_ni_cell(0)
    engine = _FakeEngine()
    _, d = av.find_mic(positions - positions[0], cell, pbc=True)
    outside = int(np.argmax(d))  # farthest (minimum-image) atom: not in the crop
    with pytest.raises(av.ActiveVolumeSaddleError, match="matched 0 atoms"):
        av.partn_refine_AV(
            engine,
            _ResetCfg(),
            0,
            positions,
            cell,
            types,
            np.array([outside]),
            positions[[outside]],
        )


# ---------------------------------------------------------------------------
# Real LAMMPS (serial)
# ---------------------------------------------------------------------------


@dataclass
class _EngineCfg:
    """Engine config; the ``lj/cut`` default sets no masses, so ``reset()`` has to."""

    pair_style: str = "lj/cut 6.0"
    pair_coeff: str = "* * 0.52 2.274"
    min_style: str = "cg"
    minimize: str = "1e-6 1e-8 1000 10000"
    frz_min: str = "1e-4 1e-6 100 1000"
    verbosity: int = 0


@dataclass
class _SearchCfg:
    """Config shim for ``partn_search_AV``."""

    lammps: _EngineCfg = field(default_factory=_EngineCfg)
    activevolume: _AVCfg = field(default_factory=_AVCfg)


def _engine_cfg(potential: str) -> _EngineCfg:
    """Return the engine config for ``potential`` (``lj/cut``, ``eam/alloy``, ``sw``)."""
    if potential == "eam/alloy":
        if not _EAM_FILE.is_file():
            pytest.skip(f"{_EAM_FILE.name} not present (examples/ not shipped)")
        return _EngineCfg(pair_style="eam/alloy", pair_coeff=f"* * {_EAM_FILE} Cr Ni")
    if potential == "sw":
        if not _SI_SW.is_file():
            pytest.skip(f"{_SI_SW.name} not present (examples/ not shipped)")
        return _EngineCfg(pair_style="sw", pair_coeff=f"* * {_SI_SW} Si")
    return _EngineCfg()


def _require_serial() -> None:
    """Skip under ``mpirun``: these tests drive one serial LAMMPS instance."""
    pytest.importorskip("lammps")
    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size() > 1:
        pytest.skip("serial test: run without mpirun")


def _diamond_si_cell() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Build a 3x3x3 diamond Si cell (216 atoms, 16.3 Angstrom box)."""
    atoms = bulk("Si", crystalstructure="diamond", a=5.431, cubic=True)
    atoms = atoms.repeat([3, 3, 3])
    return (
        atoms.get_chemical_symbols(),
        atoms.get_positions(),
        np.array(atoms.get_cell()),
    )


def _cr_cluster_cell(radius: float) -> tuple[list[str], np.ndarray, np.ndarray]:
    """4x4x4 FCC Ni with every atom within ``radius`` of atom 0 turned into Cr."""
    types, positions, cell = _fcc_ni_cell(0)
    _, d = av.find_mic(positions - positions[0], cell, pbc=True)
    for i in np.where(d <= radius)[0]:
        types[int(i)] = "Cr"
    return types, positions, cell


def _started_engine(config: _EngineCfg) -> object:
    """Return a started ``LammpsEngine`` (serial)."""
    from pykmc.engine.lammps import LammpsEngine

    engine = LammpsEngine(config=config, comm=None)
    engine.start()
    return engine


def _initialize(
    engine: object, types: list[str], positions: np.ndarray, cell: np.ndarray
) -> None:
    """Load the full system into ``engine``."""
    engine.initialize_parameters()
    engine.initialize_system(
        types=types, positions=positions, cell=Cell(cell), pbc=[True] * 3
    )
    engine.initialize_potential()


def _gathered_types(lmp: object) -> np.ndarray:
    """Integer types of the live instance, in LAMMPS id order."""
    return np.ctypeslib.as_array(lmp.gather_atoms("type", 0, 1)).copy()


def test_map_types_matches_lammps_engine_types() -> None:
    """``map_types`` reproduces the integer types and masses the main engine assigns."""
    _require_serial()
    types, positions, cell = _fcc_ni_cell(5)
    engine = _started_engine(_EngineCfg())
    try:
        _initialize(engine, types, positions, cell)
        int_types, map_type = av.map_types(types)
        assert engine.lmp.extract_global("ntypes") == len(map_type)
        np.testing.assert_array_equal(_gathered_types(engine.lmp), int_types)
        live_mass = engine.lmp.extract_atom("mass")
        for entry in map_type.values():
            assert live_mass[entry["ref"]] == pytest.approx(entry["mass"])
    finally:
        engine.close()


@pytest.mark.parametrize(
    ("cell_kind", "n_species", "potential"),
    [
        ("Ni", 1, "lj/cut"),
        ("NiCr", 2, "lj/cut"),
        ("NiCr", 2, "eam/alloy"),
        ("Si", 1, "sw"),
    ],
    ids=["Ni-lj", "NiCr-lj", "NiCr-eam", "Si-sw"],
)
def test_partn_search_AV_crop_holds_every_species(
    cell_kind: str, n_species: int, potential: str
) -> None:
    """The AV crop is built with the main engine's integer types and masses.

    ``partn_search_AV`` is the real entry point up to (not including) the
    pARTn search, so no plugin load or MPI launch is needed. The ``sw`` case
    is the SW-Si failure: the pair style sets no masses, so the crop's first
    rebalance aborts unless ``reset()`` emitted them. Afterwards
    ``ensure_full_system`` must give back a full engine with the fresh energy.
    """
    _require_serial()
    if cell_kind == "Si":
        types, positions, cell = _diamond_si_cell()
    else:
        types, positions, cell = _fcc_ni_cell(5 if cell_kind == "NiCr" else 0)
    config = _SearchCfg(lammps=_engine_cfg(potential))
    engine = _started_engine(config.lammps)
    try:
        _initialize(engine, types, positions, cell)
        e_full = engine.get_total_energy()
        full_types = _gathered_types(engine.lmp)
        assert not engine.system_is_cropped

        atom_map, central_id = av.partn_search_AV(
            engine, config, 0, positions, cell, types
        )
        lmp = engine.lmp
        assert engine.system_is_cropped
        int_types, map_type = av.map_types(types)
        expected = int_types[atom_map]
        assert set(expected.tolist()) == set(range(1, n_species + 1))

        assert lmp.extract_global("ntypes") == n_species
        assert lmp.get_natoms() == len(atom_map)
        np.testing.assert_array_equal(_gathered_types(lmp), expected)
        np.testing.assert_array_equal(_gathered_types(lmp), full_types[atom_map])
        assert atom_map[central_id[0] - 1] == 0
        # eam/alloy re-sets masses from the setfl file (potential-provided
        # masses win); every other style keeps the ASE values reset() emitted.
        rel = 1e-3 if potential == "eam/alloy" else 1e-12
        live_mass = lmp.extract_atom("mass")
        for entry in map_type.values():
            assert live_mass[entry["ref"]] == pytest.approx(entry["mass"], rel=rel)
        # A `run 0` on the crop succeeds (masses set) with a finite energy/atom.
        lmp.command("run 0 post no")
        assert np.isfinite(lmp.get_thermo("pe") / lmp.get_natoms())

        assert engine.ensure_full_system(positions) is True
        assert not engine.system_is_cropped
        assert lmp.get_natoms() == len(types)
        assert engine.get_total_energy() == pytest.approx(e_full, abs=1e-8)
    finally:
        engine.close()


@pytest.mark.parametrize("potential", ["lj/cut", "eam/alloy"], ids=["lj", "eam"])
def test_crop_with_only_second_species_keeps_full_type_map(potential: str) -> None:
    """A crop holding only Cr atoms still has two types, Ni's mass and Cr = type 1."""
    _require_serial()
    types, positions, cell = _cr_cluster_cell(radius=6.0)
    config = _SearchCfg(lammps=_engine_cfg(potential))
    engine = _started_engine(config.lammps)
    try:
        _initialize(engine, types, positions, cell)
        full_types = _gathered_types(engine.lmp)
        atom_map, _ = av.partn_search_AV(engine, config, 0, positions, cell, types)
        crop_symbols = {types[i] for i in atom_map}
        assert crop_symbols == {"Cr"}, "fixture must yield a Cr-only crop"

        lmp = engine.lmp
        assert lmp.extract_global("ntypes") == 2
        got = _gathered_types(lmp)
        assert set(got.tolist()) == {1}
        np.testing.assert_array_equal(got, full_types[atom_map])
        rel = 1e-3 if potential == "eam/alloy" else 1e-12  # setfl masses win
        live_mass = lmp.extract_atom("mass")
        assert live_mass[1] == pytest.approx(_mass("Cr"), rel=rel)
        assert live_mass[2] == pytest.approx(_mass("Ni"), rel=rel)
        assert engine.full_system.species == ("Cr", "Ni")
        # full_system.masses follow the live instance after pair_coeff (the
        # setfl values for eam/alloy, the ASE values for lj/cut).
        assert engine.full_system.masses == (live_mass[1], live_mass[2])
        assert engine.full_system.masses == pytest.approx(
            (_mass("Cr"), _mass("Ni")), rel=rel
        )
    finally:
        engine.close()


def test_crop_boundary_follows_slab_pbc() -> None:
    """A crop of a slab keeps the non-periodic axis non-periodic and restores."""
    _require_serial()
    from ase.build import surface

    atoms = surface("Ni", (1, 0, 0), layers=4, vacuum=5.0).repeat([4, 4, 1])
    types = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    cell = np.array(atoms.get_cell())
    config = _SearchCfg()
    engine = _started_engine(config.lammps)
    try:
        engine.initialize_parameters()
        engine.initialize_system(
            types=types, positions=positions, cell=atoms.get_cell(), pbc=atoms.get_pbc()
        )
        engine.initialize_potential()
        assert engine.full_system.pbc == (True, True, False)
        e_full = engine.get_total_energy()
        av.partn_search_AV(engine, config, 0, positions, cell, types)
        assert list(engine.lmp.extract_global("periodicity")) == [1, 1, 0]
        assert engine.ensure_full_system(positions) is True
        assert list(engine.lmp.extract_global("periodicity")) == [1, 1, 0]
        assert engine.get_total_energy() == pytest.approx(e_full, abs=1e-8)
    finally:
        engine.close()


def test_partn_refine_av_real_places_shell_atom_displaced_past_ract() -> None:
    """Real LAMMPS: a 5NN shell atom relaxed 0.05 A past ``ract`` is refined.

    Same input as the review probe (FCC Ni 4x4x4, ``ract=5.6``, ``rmov=3.0``,
    outer shell atom displaced radially outward): the crop is built, the
    saddle position is placed and held by ``f_core`` through the core
    minimisation, and the engine restores afterwards.
    """
    _require_serial()
    types, positions, cell = _fcc_ni_cell(0)
    config = _SearchCfg(activevolume=_AVCfg(ract=5.6, rmov=3.0))
    edge, saddle = _outer_shell_atom(positions, cell, config.activevolume.ract)
    engine = _started_engine(config.lammps)
    try:
        _initialize(engine, types, positions, cell)
        e_full = engine.get_total_energy()
        e_init, atom_map, central_id = av.partn_refine_AV(
            engine, config, 0, positions.copy(), cell, types, np.array([edge]), saddle
        )
        assert np.isfinite(e_init)
        assert atom_map[central_id[0] - 1] == 0
        crop_index = int(np.where(atom_map == edge)[0][0])
        crop_positions = av.get_positions(engine)
        np.testing.assert_allclose(crop_positions[crop_index], saddle[0], atol=1e-10)
        assert engine.ensure_full_system(positions) is True
        assert engine.get_total_energy() == pytest.approx(e_full, abs=1e-8)
    finally:
        engine.close()
