"""Atoms present and displaced in an HTST Hessian calculation."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from pykmc.region import Region, SphereParams


@dataclass(frozen=True)
class HessianRegion:
    """Atom sets of one Hessian calculation.

    Attributes
    ----------
    present : np.ndarray
        Sorted global indices of the atoms built into the scratch engine.
    displaced : np.ndarray
        Sorted positions within `present` of the atoms the Hessian displaces. Their
        global indices are `present[displaced]`. The other present atoms stay fixed.
    pbc : tuple[bool, bool, bool]
        Periodic boundary conditions of the scratch box.
    n_zero_modes : int
        3 (the translations) when every atom of the system is displaced, else 0.

    """

    present: np.ndarray
    displaced: np.ndarray
    pbc: tuple[bool, bool, bool]
    n_zero_modes: int


def _as_pbc(pbc: Any) -> tuple[bool, bool, bool]:
    return (bool(pbc[0]), bool(pbc[1]), bool(pbc[2]))


def build_region(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: Any,
    center_index: int,
    r_displaced: float,
    r_present: float,
) -> HessianRegion:
    """Select the atoms present and displaced in a Hessian calculation.

    Both sets are spheres around atom `center_index`, selected by the region
    module (minimum-image distances).

    Parameters
    ----------
    positions : np.ndarray
        Atomic positions, shape (N, 3).
    cell : np.ndarray
        Simulation cell (orthogonal).
    pbc : sequence of bool
        Periodic boundary conditions of the system, given to the scratch box.
    center_index : int
        Atom the distances are measured from.
    r_displaced : float
        Atoms within this distance of the centre are displaced.
    r_present : float
        Atoms within this distance of the centre are present.

    Returns
    -------
    HessianRegion
        The atom sets of the calculation.

    """

    def sphere(radius: float) -> np.ndarray:
        region = Region.create(
            "sphere", config=SphereParams(center_atom=center_index, radius=radius)
        )
        return np.asarray(region.select(positions, cell=cell), dtype=np.int64)

    present = sphere(r_present)
    displaced = np.searchsorted(present, sphere(r_displaced)).astype(np.int64)
    return HessianRegion(
        present, displaced, _as_pbc(pbc), 3 if displaced.size == len(positions) else 0
    )


def wrapped_on_open_axis(
    region: HessianRegion,
    geometries: Any,
    start_positions: np.ndarray,
    cell: np.ndarray,
    pbc: Any,
) -> bool:
    """Tell whether a geometry was wrapped across a non-periodic axis.

    The geometries of a reference event are recentred with a periodic wrap on every
    axis. Along a non-periodic axis, the present atoms that were wrapped sit one
    cell length away from the others, at positions the system never had.

    Parameters
    ----------
    region : HessianRegion
        The region of the calculation.
    geometries : sequence of np.ndarray
        Full-size geometries of the event (saddle and minima).
    start_positions : np.ndarray
        Positions of the system the event was searched from, shape (N, 3).
    cell : np.ndarray
        Simulation cell (orthogonal).
    pbc : sequence of bool
        Periodic boundary conditions of the system.

    """
    cell = np.asarray(cell)
    for axis in range(3):
        if pbc[axis]:
            continue
        for geometry in geometries:
            shift = (
                geometry[region.present, axis] - start_positions[region.present, axis]
            )
            if shift.max() - shift.min() > cell[axis, axis] / 2:
                return True
    return False


def frozen_overlap(region: HessianRegion, frozen_ids: np.ndarray) -> int:
    """Count the frozen atoms among the displaced atoms of `region`.

    Frozen atoms that are only present (never displaced) do not count.
    """
    return int(np.intersect1d(region.present[region.displaced], frozen_ids).size)
