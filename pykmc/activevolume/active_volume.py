"""Active-volume crop of the LAMMPS engine for pARTn search and refinement.

The active volume (AV) rebuilds the *shared* LAMMPS instance as a spherical
crop around a central atom (``define_AV``); ``LammpsEngine.partn_search`` /
``partn_refine`` restore the full system afterwards (see the engine's
"State ownership" notes). Three index spaces meet here:

- **global index**: position in the full-system arrays (0-based);
- **crop index**: position in ``atom_map`` (0-based); ``atom_map[i]`` is the
  global index of crop atom ``i``;
- **LAMMPS id**: ``crop index + 1`` inside the cropped instance.

Species/type/mass identity comes from the initialized engine's full-system
descriptor. The crop retains its ordered species slots and authoritative
masses, including species absent from the crop or even the source atom list.
Standalone helpers without a descriptor use the engine's default alphabetical
``species_map`` rule.

Supported geometry: orthorhombic cells only (``reset`` raises before touching
the instance otherwise). ``define_AV`` selects atoms with minimum-image
distances along the source's actual periodic axes. Search, refinement and
resolved endpoint constraints use that same boundary policy; a nonperiodic
axis never captures atoms across a fictitious periodic image.

Every position array handed to LAMMPS from here (``partn_search_AV``,
``partn_refine_AV``, ``redefine_atoms``, ``set_positions``) goes through the
engine's finite/shape guard first (``pykmc.engine.lammps._require_positions``)
so a NaN coordinate raises ``ValueError`` symmetrically on every rank before
any collective call instead of desynchronising a multi-rank instance.
"""

from __future__ import annotations

from typing import Protocol, TypedDict

import numpy as np
import ctypes
from ase.cell import Cell
from ase.geometry import find_mic
from ..physics import overlay_tolerance, validate_event_constraints, _indices


class ActiveVolumeSaddleError(ValueError):
    """A saddle atom is missing from (or duplicated in) the active-volume crop."""


def _check_positions(
    positions: np.ndarray, op_name: str, natoms: int | None = None
) -> np.ndarray:
    """Apply the engine's finite/shape guard (imported lazily, like ``map_types``)."""
    from ..engine.lammps import _require_positions

    return _require_positions(positions, op_name, natoms=natoms)


def require_orthorhombic_cell(cell: np.ndarray, op_name: str) -> None:
    """Raise ``ValueError`` if ``cell`` is not orthorhombic.

    The active-volume crop builds a ``region box block`` and selects atoms
    with cell-diagonal bounds, so a triclinic cell must be rejected *before*
    any ``clear`` rebuilds the instance.

    Parameters
    ----------
    cell : np.ndarray
        (3, 3) cell (or an ``ase.cell.Cell``).
    op_name : str
        Operation name used in the error message.

    """
    if not Cell.new(np.asarray(cell, dtype=float)).orthorhombic:
        raise ValueError(
            f"{op_name}: the active-volume crop supports orthorhombic cells only "
            "(the crop box is a `region box block`); got a non-orthorhombic cell"
        )


def define_AV(
    config: object,
    central_atom_idx: int,
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: tuple[bool, bool, bool] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the active-volume members around ``central_atom_idx``.

    Members use minimum-image distances only along the supplied periodic axes.
    Standalone callers omitting ``pbc`` retain the fully periodic default.
    """
    # Defining parameters
    # Radius of whole active volume in Ang
    r_a = config.activevolume.ract  # Ensure AV is larger than topology analysis
    # Defines the radius of atoms that can move.
    r_m = config.activevolume.rmov

    # NEED TO ADD WARNING IF R_A<R_M

    axes = np.asarray((True, True, True) if pbc is None else pbc)
    if axes.ndim == 0:
        axes = np.repeat(axes, 3)
    if axes.shape != (3,) or axes.dtype.kind != "b":
        raise ValueError("active volume requires three boolean PBC axes")

    center = positions[central_atom_idx]

    inner_movable_idx = []
    buffer_idx = []
    total_active_idx = []
    non_active_idx = []

    for i, pos in enumerate(positions):
        diff = pos - center
        diff_mic, distance = find_mic(diff, cell, pbc=axes)
        if np.abs(distance) <= r_m:
            inner_movable_idx.append(i)
            total_active_idx.append(i)  # Inner is also part of total active
        elif (
            np.abs(distance) > r_m and np.abs(distance) <= r_a
        ):  # Can change to make it between r_m and r_a
            buffer_idx.append(i)
            total_active_idx.append(i)
        else:
            non_active_idx.append(i)

    buffer_idx = np.array(sorted(buffer_idx))
    av_idx = np.array(sorted(total_active_idx))

    av_positions = positions[av_idx]

    # print(len(av_idx)," atoms in AV,", len(av_idx)-len(buffer_idx), "movable atoms")

    return av_positions, av_idx, buffer_idx


def _quiet(engine: _CommandEngine, cmd: str) -> None:
    """Run a cleanup command, swallowing its error so the original propagates."""
    try:
        engine.command(cmd)
    except Exception:  # noqa: BLE001 - secondary cleanup failure
        pass


def make_AV(engine, av_indices, buffer_indices):
    # Define the buffer group based on the new LAMMPS IDs
    # We need to find which index in 'av_indices' corresponds to 'buffer_indices'
    engine_buffer_ids = []
    buffer_set = set(buffer_indices)
    for i, original_id in enumerate(av_indices):
        if original_id in buffer_set:
            engine_buffer_ids.append(i + 1)  # LAMMPS IDs are 1-based

    # The buffer group and its setforce fix must outlive this call (they hold
    # the shell frozen during the pARTn run); the full-system restore removes
    # them. They are only torn down here if the first run fails.
    try:
        if engine_buffer_ids:
            engine.command(f"group buffer id {' '.join(map(str, engine_buffer_ids))}")
            engine.command("fix f_buffer buffer setforce 0.0 0.0 0.0")
        else:
            engine.command("group buffer empty")
            engine.command("fix f_buffer buffer setforce 0.0 0.0 0.0")
            print("No buffer atoms defined")

        engine.command("run 0 post no")
    except Exception:
        _quiet(engine, "unfix f_buffer")
        _quiet(engine, "group buffer delete")
        raise


class TypeEntry(TypedDict):
    """One LAMMPS atom type: its 1-based index and its mass."""

    ref: int
    mass: float


class _CommandEngine(Protocol):
    """Minimal engine surface the active-volume box rebuild drives."""

    def command(self, cmd: str) -> None:
        """Run one LAMMPS command string."""
        ...


def map_types(
    types: list[str] | np.ndarray,
    *,
    species: tuple[str, ...] | None = None,
    masses: tuple[float, ...] | None = None,
) -> tuple[np.ndarray, dict[str, TypeEntry]]:
    """Map element symbols to LAMMPS integer types.

    Supply the full-system ``species`` and ``masses`` together to retain
    authoritative type order, absent slots and isotope masses. Validation
    shares the engine's explicit-map rules and happens before any native
    mutation. Without an explicit map, use alphabetical species and ASE
    masses from the full source ``types`` for standalone helper callers.
    Returns per-atom integer types and a ``{symbol: {"ref", "mass"}}`` map.
    """
    from ..engine.lammps import (
        _validate_species_override,
        species_map,
        types_to_int,
    )

    if (species is None) != (masses is None):
        raise ValueError("map_types: species and masses must be given together")
    if species is None:
        species, masses = species_map(types)
    else:
        species, masses = _validate_species_override(species, masses)
    map_type: dict[str, TypeEntry] = {
        symbol: {"ref": i + 1, "mass": mass}
        for i, (symbol, mass) in enumerate(zip(species, masses, strict=True))
    }
    int_types = types_to_int(types, species)
    return int_types, map_type


def _engine_pbc(engine: object) -> tuple[bool, bool, bool]:
    """Return the pbc remembered by ``engine`` or fully periodic if it has none."""
    full = getattr(engine, "full_system", None)
    pbc = getattr(full, "pbc", None)
    if pbc is None:
        return (True, True, True)
    return (bool(pbc[0]), bool(pbc[1]), bool(pbc[2]))


def reset(
    engine: _CommandEngine,
    config: object,
    cell: np.ndarray,
    map_type: dict[str, TypeEntry],
    pbc: tuple[bool, bool, bool] | None = None,
    *,
    preserve_masses: bool = False,
) -> None:
    """Clear the LAMMPS instance and rebuild an empty box for the active volume.

    ``map_type`` is the ``{symbol: {"ref", "mass"}}`` map ``map_types`` built
    from the full-system types. The box gets one LAMMPS atom type per species
    and one ``mass`` per type, both before ``pair_coeff``: a multi-element
    ``pair_coeff`` (``eam/alloy``, ``eam/fs``) needs the full type count to
    exist, and pair styles that do not set masses themselves (``lj/cut``,
    ``mlip``, ``sw``) need them before the first ``run 0``.

    ``pbc`` is the crop's boundary; when omitted it is read from the engine's
    remembered full system (``engine.full_system.pbc``) and falls back to
    fully periodic for engines without one. A non-orthorhombic ``cell`` raises
    ``ValueError`` before anything is cleared.

    With ``preserve_masses=True``, reapply the supplied authoritative masses
    after ``pair_coeff`` too: EAM potentials may overwrite masses when read.
    The real crop entry points use this when a full-system map is available.
    """
    require_orthorhombic_cell(cell, "active_volume.reset")
    if pbc is None:
        pbc = _engine_pbc(engine)
    engine.command("clear")
    initialize_parameters(engine, pbc)
    # Create cell
    xhi, yhi, zhi = cell[0][0], cell[1, 1], cell[2, 2]
    engine.command("region box block 0.0 {} 0.0 {} 0.0 {}".format(xhi, yhi, zhi))
    engine.command("create_box {} box".format(len(map_type)))
    for entry in map_type.values():
        engine.command("mass {} {}".format(entry["ref"], entry["mass"]))
    initialize_potential(engine, config)
    if preserve_masses:
        for entry in map_type.values():
            engine.command("mass {} {}".format(entry["ref"], entry["mass"]))


def clear(engine):
    """
    Clears lammps instance
    """
    engine.command("clear")


def redefine_atoms(engine, positions, type=None) -> None:
    """
    Check to see if current lammps system has enough atoms
    If not, deletes all atoms then redefines them
    """
    if type is None:
        type = [1] * len(positions)
    positions = _check_positions(
        positions, "active_volume.redefine_atoms", natoms=len(positions)
    )
    new_positions = positions.flatten().astype(np.float64)
    ids = np.arange(1, len(positions) + 1, dtype=np.int32)
    engine.lmp.create_atoms(len(positions), ids, type, x=new_positions)
    engine.command("comm_style tiled")
    engine.command("balance 1.1 rcb")
    engine.command("neigh_modify every 1 delay 0 check yes")
    engine.command("fix 1 all setforce 0.0 0.0 0.0")
    try:
        engine.command("run 0")
    finally:
        _quiet(engine, "unfix 1")


def partn_search_AV(
    engine,
    config,
    central_atom_idx: int,
    positions,
    cell,
    type,
    constraints=None,
    user_constraints=None,
    *,
    validated: bool = False,
) -> [np.array, int]:
    """Crop the engine to the active volume around ``central_atom_idx``.

    ``validated`` declares that the caller (the engine's public wrapper, or
    ``partn_refine_AV``) already ran ``validate_event_constraints`` on this
    same ``constraints`` payload against this source, cell and PBC; the
    helper then trusts it instead of repeating the O(N) validation. A direct
    call without a validated payload always validates.
    """
    # Guard the full-system positions before the crop clears anything: a NaN
    # raises here on every rank, with the engine still intact.
    positions = _check_positions(
        positions, "active_volume.partn_search_AV", natoms=len(type)
    )
    full = getattr(engine, "full_system", None)
    species = getattr(full, "species", None)
    masses = getattr(full, "masses", None)
    int_types, map_type = map_types(type, species=species, masses=masses)
    pbc = _engine_pbc(engine)
    if constraints is None or not validated:
        constraints = validate_event_constraints(
            config,
            positions,
            type,
            cell,
            pbc,
            central_atom_idx,
            constraints,
            user_constraints=user_constraints,
            active_volume=True,
        )
    av_positions, av_idx, _ = define_AV(
        config, central_atom_idx, positions, cell, pbc=pbc
    )
    reset(
        engine,
        config,
        cell,
        map_type=map_type,
        pbc=pbc,
        preserve_masses=species is not None,
    )
    atom_map = np.array(av_idx, dtype=int)
    av_type = int_types[atom_map]

    redefine_atoms(engine, av_positions, av_type)
    # This group owns the whole execution restriction, including user atoms
    # inside rmov. Native IDs are translated from full-source rows only here.
    make_AV(engine, av_idx, constraints.local_fixed_indices)
    return atom_map, np.array(np.where(atom_map == central_atom_idx)[0] + 1)


def partn_refine_AV(
    engine,
    config,
    central_atom_idx: int,
    positions,
    cell,
    type,
    saddle_idx,
    saddle_positions,
    constraints=None,
    user_constraints=None,
    *,
    validated: bool = False,
) -> [float, np.array, int]:
    """
    Receive the system with the central atom index, define an active volume around this atom, then update the positions
    with those for the saddle.

    This was added in order to get the activation energy for an event, as the traditional method does not work for
    Active Volumes.

    Raises ``ActiveVolumeSaddleError`` when a saddle atom is not in the crop
    (the engine reports it as ``Err(REFINEMENT_INVALID_MINIMA)``); this is
    the base's ``.item()`` crash case. A saddle position is placed wherever
    the caller put it, as the base did: there is no distance check against
    ``ract`` (an in-crop shell atom relaxed slightly past ``ract`` is a
    legitimate saddle geometry). Non-finite ``saddle_positions`` raise
    ``ValueError`` before any LAMMPS command. ``validated`` has the same
    meaning as in ``partn_search_AV``.
    """
    saddle_idx = np.array(_indices(saddle_idx, upper=len(positions)), dtype=int)
    saddle_positions = _check_positions(
        saddle_positions, "active_volume.partn_refine_AV (saddle_positions)"
    )
    if saddle_positions.shape != (saddle_idx.size, 3):
        raise ValueError(
            "active_volume.partn_refine_AV: saddle_positions have shape "
            f"{saddle_positions.shape}, expected {(saddle_idx.size, 3)} "
            "(one row per saddle_idx entry)"
        )

    if constraints is None or not validated:
        constraints = validate_event_constraints(
            config,
            positions,
            type,
            cell,
            _engine_pbc(engine),
            central_atom_idx,
            constraints,
            user_constraints=user_constraints,
            active_volume=True,
        )
    # Only user-declared fixed atoms are a coordinate contract: validate them at
    # the PSR tolerance and re-clamp their rows. The AV shell is a crop
    # restriction held by ``f_buffer``/``f_core``, so a shell atom is placed
    # wherever the caller put it (contracts 7f policy 5).
    proposed = np.array(positions, copy=True)
    proposed[saddle_idx] = saddle_positions
    constraints.validate_positions(
        proposed, tolerance=overlay_tolerance(config), user_only=True
    )
    saddle_positions = constraints.protect_positions(proposed, user_only=True)[
        saddle_idx
    ]
    atom_map, central_lammps_id = partn_search_AV(
        engine,
        config,
        central_atom_idx,
        positions,
        cell,
        type,
        constraints=constraints,
        user_constraints=user_constraints,
        validated=True,
    )
    av_positions = positions[atom_map]

    if config.activevolume.AV_debug:
        E_before = get_potential_energy(engine)
        engine.command("min_style {}".format(config.lammps.min_style))
        engine.command("minimize 1.0e-6 1.0e-8 10 10")
        E_init = get_potential_energy(engine)
        print("Before minimization: ", E_before, "After minimization: ", E_init)
        print("% Difference:", abs((E_before - E_init) / E_init * 100), "%")
    else:
        E_init = get_potential_energy(engine)

    core_idx = []
    core_ids = []
    for i, atom_idx in enumerate(saddle_idx):
        matches = np.where(atom_map == atom_idx)[0]
        if matches.size != 1:
            raise ActiveVolumeSaddleError(
                f"saddle atom {atom_idx} matched {matches.size} atoms in the "
                "active-volume map (expected exactly 1)"
            )
        index = int(matches[0])  # index in atom map where this value is true
        av_positions[index] = saddle_positions[i]
        core_idx.append(index)  # Atom id
        core_ids.append(index + 1)
    set_positions(engine, av_positions)

    engine.command("fix 1 all setforce 0.0 0.0 0.0")
    try:
        engine.command("run 0")
    finally:
        _quiet(engine, "unfix 1")

    # Want to minimize initially to speed up refinement process
    engine.command(f"group core id {' '.join(map(str, core_ids))}")
    try:
        engine.command("fix f_core core setforce 0.0 0.0 0.0")
        try:
            engine.command("min_style {}".format(config.lammps.min_style))
            engine.command("minimize {}".format(config.lammps.frz_min))
        finally:
            _quiet(engine, "unfix f_core")
    finally:
        _quiet(engine, "group core delete")

    return E_init, atom_map, central_lammps_id


def position_results_AV(
    config, artn, atom_map, positions
) -> [np.array, np.array, np.array, int]:
    min1positions = artn.extract("tau_min1")
    min2positions = artn.extract("tau_min2")
    saddlepositions = artn.extract("tau_sad")

    # find atom that moves the most
    dist = (min1positions - saddlepositions) ** 2
    dist = dist.sum(axis=-1)
    dist = np.sqrt(dist)
    dist[dist > config.atomicenvironment.rcut] = (
        0
        # if atom moves more that rcutevent, consider that it crosses the cell (happens with lammps), so distance = 0 to not consider it as the one that moves the most
    )
    index_move = np.argmax(dist)

    index_move_mapped = atom_map[index_move]

    min1positions_mapped = positions.copy()
    min2positions_mapped = positions.copy()
    saddlepositions_mapped = positions.copy()

    for i, atom_idx in enumerate(atom_map):
        saddlepositions_mapped[atom_idx][0] = saddlepositions[i][0]
        saddlepositions_mapped[atom_idx][1] = saddlepositions[i][1]
        saddlepositions_mapped[atom_idx][2] = saddlepositions[i][2]

        min1positions_mapped[atom_idx][0] = min1positions[i][0]
        min1positions_mapped[atom_idx][1] = min1positions[i][1]
        min1positions_mapped[atom_idx][2] = min1positions[i][2]

        min2positions_mapped[atom_idx][0] = min2positions[i][0]
        min2positions_mapped[atom_idx][1] = min2positions[i][1]
        min2positions_mapped[atom_idx][2] = min2positions[i][2]

    return (
        min1positions_mapped,
        min2positions_mapped,
        saddlepositions_mapped,
        index_move_mapped,
    )


def initialize_parameters(
    engine: _CommandEngine, pbc: tuple[bool, bool, bool] = (True, True, True)
) -> None:
    """Emit the crop's global settings; ``boundary`` follows ``pbc``."""
    engine.command("units metal")
    engine.command("atom_style atomic")
    engine.command("dimension 3")
    engine.command("boundary " + " ".join("p" if p else "f" for p in pbc))
    engine.command("atom_modify map array")  # ! necessary for scatter atoms
    engine.command("atom_modify sort 0 0.0")  # ! necessary for partn


def initialize_system(engine, system):
    # system parameters
    natoms = len(system.types)
    cell = system.cell
    x = system.positions.flatten()  # Lammps format

    xhi, yhi, zhi = cell[0][0], cell[1, 1], cell[2, 2]

    ind = np.linspace(0, natoms - 1, natoms).astype(int)
    ind += 1  # Lammps id start at 1
    # Same species -> type/mass rule as the engine (see map_types).
    int_types, map_type = map_types(system.types)

    # lammps create system
    engine.command("region box block 0.0 {} 0.0 {} 0.0 {}".format(xhi, yhi, zhi))
    engine.command("create_box {} box".format(len(map_type)))
    engine.lmp.create_atoms(natoms, ind, int_types.tolist(), x)
    # Set masses
    for key in map_type.keys():
        engine.command("mass {} {}".format(map_type[key]["ref"], map_type[key]["mass"]))
    # Label atoms name to type :
    engine.command(
        "labelmap atom "
        + " ".join(f"{int(e['ref'])} {key}" for key, e in map_type.items())
    )


def initialize_potential(engine, config):
    pair_style = config.lammps.pair_style
    pair_coeff = config.lammps.pair_coeff
    engine.command("pair_style {}".format(pair_style))
    engine.command("pair_coeff {}".format(pair_coeff))


def minimize(engine, config, positions=None):
    if positions is not None:
        set_positions(engine=engine, positions=positions)
    engine.command("min_style {}".format(config.lammps.min_style))
    engine.command("minimize {}".format(config.lammps.minimize))


def get_total_energy(engine, positions=None):
    if positions is not None:
        set_positions(engine=engine, positions=positions)
    # Get total energy
    engine.command("run 0")
    result = engine.lmp.get_thermo("etotal")
    if engine.rank == 0:
        return result


def get_potential_energy(engine, positions=None):
    if positions is not None:
        set_positions(engine=engine, positions=positions)
    # get potential energy
    engine.command("compute c1 all pe")
    try:
        engine.command("run 0")
        result = engine.lmp.extract_compute("c1", 0, 0)
    finally:
        _quiet(engine, "uncompute c1")
    return result


def get_positions(engine):
    result = engine.lmp.gather_atoms("x", 1, 3)
    if engine.rank == 0:
        # convert ctype positions into a numpy array
        result = np.ctypeslib.as_array(result)
        result = np.reshape(result, (-1, 3))
        return result


def set_positions(engine, positions):
    # Real engines own the periodic-image and frame conversion boundary. AV
    # crops retain the source cell, so refinement uses that same scatter path.
    # Command-only helper engines retain the legacy guarded protocol below.
    scatter = getattr(engine, "set_positions", None)
    if scatter is not None:
        scatter(positions)
        return
    # Same symmetric finite/shape guard as LammpsEngine.set_positions.
    positions = _check_positions(
        positions, "active_volume.set_positions", natoms=int(engine.lmp.get_natoms())
    )
    positions = positions.flatten().astype(np.float64)
    positions = np.ascontiguousarray(positions)
    c_array = (ctypes.c_double * len(positions))(*positions)
    engine.lmp.scatter_atoms("x", 1, 3, c_array)
