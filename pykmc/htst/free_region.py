"""Minimum-image selection of the free (movable) atoms around an event centre.

The free region is a single continuous-radius sphere around one atom, evaluated
with minimum-image distances on periodic axes of an orthorhombic cell. It is
deliberately independent of the active-volume machinery: no engine, no config.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .request import HTSTEventRequest, HTSTRequestError, require_orthorhombic


def select_free_indices(
    positions: Any,
    center_index: int,
    radius: float,
    cell: Any,
    pbc: Any,
) -> np.ndarray:
    """Return the sorted global indices of atoms within ``radius`` of the centre.

    The cutoff is inclusive and the centre atom sits at distance zero, so the
    result always contains ``center_index``; a tiny radius yields exactly
    ``[center_index]``. Distances use the minimum-image convention on periodic
    axes. A sphere larger than half a periodic cell length sees one image per
    atom only, as with any minimum-image selection.

    Parameters
    ----------
    positions : array_like
        ``(N, 3)`` Cartesian positions in Å.
    center_index : int
        Global index of the centre atom.
    radius : float
        Inclusive cutoff radius in Å; must be finite and >= 0.
    cell : array_like
        ``(3, 3)`` orthorhombic cell.
    pbc : array_like
        Three booleans (Python or NumPy; integer ``0``/``1`` are also accepted),
        periodicity per axis. Anything else, including strings, is rejected.

    Returns
    -------
    np.ndarray
        Sorted ``int`` array of global atom indices.

    Raises
    ------
    HTSTGeometryError
        If the cell is not orthorhombic.
    HTSTRequestError
        If ``positions`` (shape or non-finite entries), ``center_index`` or
        ``pbc`` are malformed.
    ValueError
        If ``radius`` is negative or not finite.

    """
    try:
        pos = np.asarray(positions, dtype=float)
    except (TypeError, ValueError) as exc:
        raise HTSTRequestError(f"positions are not numeric: {exc}") from exc
    if pos.ndim != 2 or pos.shape[1] != 3 or pos.shape[0] < 1:
        raise HTSTRequestError(
            f"positions must have shape (N, 3) with N >= 1, got {pos.shape}"
        )
    if not np.all(np.isfinite(pos)):
        raise HTSTRequestError("positions contain non-finite entries")
    n_atoms = pos.shape[0]
    if isinstance(center_index, bool) or not isinstance(
        center_index, (int, np.integer)
    ):
        raise HTSTRequestError(
            f"center_index must be an int, got {type(center_index).__name__}"
        )
    center = int(center_index)
    if not 0 <= center < n_atoms:
        raise HTSTRequestError(
            f"center_index {center} out of range for {n_atoms} atoms"
        )
    if isinstance(radius, bool) or not isinstance(
        radius, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"radius must be a real number, got {radius!r}")
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError(f"radius must be finite and >= 0, got {radius!r}")
    cell_arr = require_orthorhombic(cell)
    pbc_arr = np.asarray(pbc).reshape(-1)
    if pbc_arr.shape != (3,):
        raise HTSTRequestError(f"pbc must have three entries, got {pbc_arr.shape}")
    if pbc_arr.dtype == bool:
        periodic = pbc_arr
    elif np.issubdtype(pbc_arr.dtype, np.integer) and np.all(np.isin(pbc_arr, (0, 1))):
        periodic = pbc_arr.astype(bool)
    else:
        raise HTSTRequestError(f"pbc must be three booleans, got {pbc!r}")

    delta = pos - pos[center]
    lengths = np.diag(cell_arr)
    for axis in range(3):
        if periodic[axis]:
            length = lengths[axis]
            if length <= 0.0:
                raise HTSTRequestError(
                    f"periodic axis {axis} needs a positive cell length, got {length!r}"
                )
            delta[:, axis] -= length * np.round(delta[:, axis] / length)
    dist = np.linalg.norm(delta, axis=1)
    # Positions are finite and radius >= 0, so the centre (distance 0) always passes.
    inside = np.flatnonzero(dist <= float(radius))
    return np.sort(inside.astype(int))


def common_free_indices(
    request: HTSTEventRequest, free_indices: Any = None
) -> np.ndarray:
    """Select one vibrational set from a validated request, excluding fixed rows.

    Explicit selections override the sphere, never the source constraints.
    Returned indices name request rows; constraint identities may be unrelated
    global IDs or a noncontiguous crop of the source ordering.
    """
    n_atoms = len(request.types)
    if free_indices is None:
        positions = (
            request.saddle_positions
            if request.settings.free_region_center == "saddle"
            else request.min1_positions
        )
        free = select_free_indices(
            positions,
            request.center_index,
            request.settings.free_radius,
            request.cell,
            request.pbc,
        )
    else:
        free = np.asarray(free_indices)
        if free.ndim != 1:
            raise ValueError(f"free_indices must be one-dimensional, got {free.shape}")
        if free.size and not np.issubdtype(free.dtype, np.integer):
            raise ValueError(f"free_indices must be integers, got dtype {free.dtype}")
        free = np.sort(free.astype(int))
        if free.size and (free[0] < 0 or free[-1] >= n_atoms):
            raise ValueError(f"free_indices out of range for {n_atoms} atoms: {free}")
        if free.size and len(np.unique(free)) != free.size:
            raise ValueError(f"free_indices contains duplicates: {free}")
    if request.constraints is not None:
        free = free[~np.isin(free, request.constraints.local_fixed_indices)]
    return free


def free_set_report(request: HTSTEventRequest) -> tuple[int, int, int, int]:
    """Count the free set of a validated request and what its constraints excluded.

    Returns ``(n_free, n_sphere, user_excluded, shell_excluded)``: the size of
    the common free set, the number of atoms within ``free_radius`` of the
    centre in the centring geometry, and how many of those the request's
    constraints excluded because they are user-fixed or because they belong to
    the active-volume shell (fixed rows that are not user-declared). The
    service logs these per request so a shrunken free set is never silent
    (contracts 7f policy 5 as amended by R14/N09).
    """
    positions = (
        request.saddle_positions
        if request.settings.free_region_center == "saddle"
        else request.min1_positions
    )
    sphere = select_free_indices(
        positions,
        request.center_index,
        request.settings.free_radius,
        request.cell,
        request.pbc,
    )
    free = common_free_indices(request, sphere)
    excluded = np.setdiff1d(sphere, free)
    if request.constraints is None or excluded.size == 0:
        return int(free.size), int(sphere.size), 0, 0
    user_rows = frozenset(request.constraints.user_view().local_fixed_indices)
    shell = int(sum(1 for i in excluded if int(i) not in user_rows))
    return int(free.size), int(sphere.size), int(excluded.size) - shell, shell


__all__ = ["common_free_indices", "free_set_report", "select_free_indices"]
