"""Force queries for one dimer search on its private serial LAMMPS engine."""

from __future__ import annotations

import ctypes
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np
from ase.cell import Cell

from ..dimer import _minimum_image, _wrap

if TYPE_CHECKING:
    from .lammps import LammpsEngine

ForceQuery = Callable[[np.ndarray], tuple[float, np.ndarray]]
_NEIGHBOR_SKIN = 0.5  # Å


class _EAMForceEvaluator:
    """Evaluate prescribed positions on the private serial engine for one search."""

    def __init__(self, engine: LammpsEngine, cell: Cell, pbc: np.ndarray) -> None:
        self.engine = engine
        self.cell = np.asarray(cell)
        self.pbc = pbc
        self.fixed_axes = None if pbc.all() else np.linalg.inv(self.cell)[:, ~pbc]
        self.previous: np.ndarray | None = None

    def __call__(self, positions: np.ndarray) -> tuple[float, np.ndarray]:
        """Return the fresh EAM potential energy and forces at this geometry."""
        engine = self.engine
        delta = None if self.previous is None else positions - self.previous
        small_step = delta is not None and np.max(np.abs(delta)) <= _NEIGHBOR_SKIN / 2
        if small_step and self.fixed_axes is not None:
            # Let full setup detect a lost atom immediately at a fixed face.
            fixed = positions @ self.fixed_axes
            small_step = np.all((fixed >= 0) & (fixed < 1))
        if small_step:
            # With dt = 1 in metal units, nve/noforce translates every atom by
            # its prescribed increment. Verlet updates ghosts and checks the
            # accumulated displacement against half the skin before computing
            # the EAM energy and forces. Scattering x and skipping setup would
            # bypass this part of the integration loop.
            velocity = np.ascontiguousarray(
                engine._positions_to_lammps(delta), dtype=np.float64
            )
            engine.lmp.scatter_atoms(
                "v", 1, 3, velocity.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            )
            engine.command("run 1 pre no post no")
            values = engine.lmp.get_thermo("pe"), engine._gather_forces()
        else:
            # A new search may change the frozen group. Use full setup for it
            # and for large jumps, including a change of periodic image.
            values = engine.get_energy_and_forces(_wrap(positions, self.cell, self.pbc))
        # The minimizer can update its positions in place between calls.
        self.previous = positions.copy()
        return values


def configure_neighbor_reuse(engine: LammpsEngine) -> None:
    """Configure Verlet updates on the worker's private EAM instance."""
    engine.command(f"neighbor {_NEIGHBOR_SKIN} bin")
    engine.command("neigh_modify delay 0 every 1 check yes")
    engine.command("thermo_style custom step pe")
    engine.command("timestep 1")
    engine.command("fix f_dimer_displace all nve/noforce")


def _interaction_bound(engine: LammpsEngine, reuse: bool) -> float | None:
    """Read an upper bound for this initialized EAM instance's pair cutoff.

    In single-collection communication mode LAMMPS reports at least the
    largest active pair cutoff plus the neighbor skin. Only subtract a skin
    explicitly set by this worker. Native EAM exposes no cutoff via extract_pair.
    """
    comm = engine.comm
    if comm is not None and comm.Get_size() == 1:
        comm = None
    leader = comm is None or comm.Get_rank() == 0
    directory = None
    path = None
    bound = None
    try:
        if leader:
            try:
                directory = TemporaryDirectory()
                path = str(Path(directory.name) / "communication")
            except OSError:
                pass
        if comm is not None:
            path = comm.bcast(path, root=0)
        if path is not None:
            # All ranks run the same command; LAMMPS writes only on rank zero.
            success = True
            try:
                engine.command(f'info communication out overwrite "{path}"')
            except OSError:
                success = False
            if comm is not None:
                success = all(comm.allgather(success))
            if leader and success:
                bound = _read_interaction_bound(Path(path), reuse)
    finally:
        if directory is not None:
            try:
                directory.cleanup()
            except OSError:
                bound = None
    return comm.bcast(bound, root=0) if comm is not None else bound


def _read_interaction_bound(path: Path, reuse: bool) -> float | None:
    """Read the interaction bound from the communication report, on rank zero."""
    try:
        for line in path.read_text().splitlines():
            if line.startswith("Communication cutoff = "):
                limit = float(line.split("=")[1])
                if reuse:
                    limit -= _NEIGHBOR_SKIN
                if np.isfinite(limit) and limit > 0:
                    # Older LAMMPS reports can round to six significant digits.
                    return limit + 1e-5 * (limit + _NEIGHBOR_SKIN)
    except (OSError, ValueError):
        pass
    return None


def _crop_selection(
    start: np.ndarray,
    movable: np.ndarray,
    center: np.ndarray,
    lattice: np.ndarray,
    pbc: np.ndarray,
    cutoff: float,
) -> tuple[np.ndarray, Callable[[np.ndarray], bool]] | None:
    """Return kept atom indices and a reach check, or None if a crop saves little."""

    def radii(x: np.ndarray) -> np.ndarray:
        return np.linalg.norm(_minimum_image(x - center, lattice, pbc), axis=1)

    distance = radii(start)
    # The 0.5 margin lets atoms move a little outward before the full system is used.
    max_reach = float(distance[movable].max()) + 0.5
    # The embedding force needs the neighbors of each movable atom's
    # neighbors. Keep both shells around every position a movable atom may reach.
    kept = np.flatnonzero(distance <= np.nextafter(max_reach + 2 * cutoff, np.inf))
    # A nearly full crop adds mapping and checking work and saves little.
    if len(kept) > 0.9 * len(start):
        return None

    def within_reach(positions: np.ndarray) -> bool:
        return bool(np.all(radii(positions) <= max_reach))

    return kept, within_reach


def _cropped_query(
    engine: LammpsEngine,
    full: ForceQuery,
    types: Sequence[str] | np.ndarray,
    start: np.ndarray,
    full_energy: float,
    cell: Cell,
    pbc: np.ndarray,
    movable: np.ndarray,
    central_atom: int,
    reuse: bool,
    resources: ExitStack,
) -> ForceQuery:
    """Build a per-search local instance only if enough frozen atoms can be omitted."""
    from .lammps import LammpsEngine

    cutoff = _interaction_bound(engine, reuse)
    if cutoff is None:
        return full
    lattice = np.asarray(cell)
    selection = _crop_selection(
        start, movable, start[central_atom].copy(), lattice, pbc, cutoff
    )
    if selection is None:
        return full
    kept, within_reach = selection
    local = LammpsEngine(
        engine.config, engine.comm, engine_id=f"crop.{engine.engine_id}"
    )
    resources.callback(local.close)
    local.start()
    local.initialize_parameters()
    local.initialize_system(
        np.asarray(types)[kept],
        _wrap(start[kept], lattice, pbc),
        cell,
        pbc,
        species=sorted(set(types)),
    )
    local.initialize_potential()
    if reuse:
        configure_neighbor_reuse(local)
    local_evaluate = (
        _EAMForceEvaluator(local, cell, pbc) if reuse else local.get_energy_and_forces
    )
    local_energy, _ = local_evaluate(start[kept])
    offset = full_energy - local_energy
    if not np.isfinite(offset):
        return full
    frozen = np.flatnonzero(~movable)
    frozen_positions = start[frozen]
    mobile = np.flatnonzero(movable)
    local_mobile = np.flatnonzero(movable[kept])
    use_full = False

    def evaluate(positions: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal use_full
        if not use_full:
            use_full = not (
                np.array_equal(positions[frozen], frozen_positions)
                and within_reach(positions[mobile])
            )
        if not use_full:
            energy, local_forces = local_evaluate(positions[kept])
            mobile_forces = local_forces[local_mobile]
            if np.isfinite(energy) and np.isfinite(mobile_forces).all():
                forces = np.zeros_like(positions)
                forces[mobile] = mobile_forces
                return energy + offset, forces
            use_full = True
        return full(positions)

    return evaluate


def dimer_force_query(
    engine: LammpsEngine,
    types: Sequence[str] | np.ndarray,
    cell: Cell,
    pbc: np.ndarray,
    movable: np.ndarray,
    central_atom: int,
    reuse: bool,
    crop: bool,
    resources: ExitStack,
) -> ForceQuery:
    """Create a query in full pyKMC atom order for the lifetime of one search."""
    full = (
        _EAMForceEvaluator(engine, cell, pbc) if reuse else engine.get_energy_and_forces
    )
    if not crop or movable.all() or not movable.any():
        return full
    query = None

    def evaluate(positions: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal query
        if query is None:
            # The first dimer evaluation is always at its starting geometry.
            # Use its full-system result to anchor the absolute energy.
            energy, forces = full(positions)
            query = full
            if np.isfinite(energy) and np.isfinite(forces).all():
                query = _cropped_query(
                    engine,
                    full,
                    types,
                    positions,
                    energy,
                    cell,
                    pbc,
                    movable,
                    central_atom,
                    reuse,
                    resources,
                )
            return energy, forces
        return query(positions)

    return evaluate
