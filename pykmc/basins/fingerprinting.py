"""Structural fingerprints for fast basin-state deduplication.

A fingerprint is a short, translation/relabeling-invariant vector summarising a
configuration. Comparing fingerprints with a cheap Chebyshev (max element-wise)
tolerance rejects the vast majority of non-matching candidate states before the
expensive full structural comparison (point-set registration / nearest-neighbour
matching) runs. This is the deduplication pre-filter for basin exploration, but the
functions are deliberately free of any basin/engine state so they can be reused for
any "are these two configurations the same?" question.

Two fingerprint flavours are provided:

* :func:`com_fingerprint` -- sorted per-atom distances from a periodic-aware centre
  of mass. General-purpose fallback; length scales with the number of atoms.
* :func:`atoms_of_interest_fingerprint` -- a two-component fingerprint built only from
  undercoordinated ("defect") atoms, far shorter and more discriminating for point
  defects in a crystal. This is the recommended choice for surface/vacancy systems.

:func:`compute_fingerprint` dispatches between them from a :class:`~pykmc.config.Config`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from pykmc.config import Config


def circular_mean_position(
    positions: np.ndarray, box: np.ndarray, pbc: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Periodic-aware centre of mass via the circular mean.

    Maps each coordinate to an angle on a circle, computes the circular mean via
    ``atan2(mean(sin), mean(cos))``, and maps back to Cartesian. This is invariant
    under any periodic re-imaging of the same point cloud.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Atom positions.
    box : ndarray, shape (3,)
        Orthorhombic box lengths (diagonal of the cell).
    pbc : ndarray, shape (3,)
        Per-dimension periodic-boundary flags.

    Returns
    -------
    com : ndarray, shape (3,)
        Centre of mass, in ``[0, L)`` for periodic dimensions.
    resultant : ndarray, shape (3,)
        Resultant length per dimension (1.0 = perfectly conditioned, 0.0 =
        uniform/ill-defined). Non-periodic dimensions get 1.0.

    """
    com = np.empty(3, dtype=np.float64)
    resultant = np.ones(3, dtype=np.float64)
    for dim in range(3):
        if pbc[dim] and box[dim] > 0:
            theta = 2.0 * np.pi * positions[:, dim] / box[dim]
            mean_sin = np.mean(np.sin(theta))
            mean_cos = np.mean(np.cos(theta))
            resultant[dim] = np.sqrt(mean_sin**2 + mean_cos**2)
            angle = np.arctan2(mean_sin, mean_cos)
            com[dim] = angle * box[dim] / (2.0 * np.pi) % box[dim]
        else:
            com[dim] = np.mean(positions[:, dim])
    return com, resultant


def reference_atom_com(
    positions: np.ndarray, box: np.ndarray, pbc: np.ndarray, ref_idx: int = 0
) -> np.ndarray:
    """Centre of mass by unwrapping all atoms relative to a reference atom.

    Uses the minimum-image convention to unwrap positions relative to
    ``positions[ref_idx]``, then takes the arithmetic mean. Well-defined when the
    point cloud fits within half the box in each dimension.
    """
    ref = positions[ref_idx]
    diffs = positions - ref
    for dim in range(3):
        if pbc[dim] and box[dim] > 0:
            diffs[:, dim] -= np.round(diffs[:, dim] / box[dim]) * box[dim]
    return ref + diffs.mean(axis=0)


def com_fingerprint(positions: np.ndarray, cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
    """Sorted per-atom distances from the centre of mass (general fallback)."""
    box = np.diag(cell).astype(np.float64)
    pbc_array = np.asarray(pbc, dtype=bool) if pbc is not None else np.array([True, True, True])
    pos = np.array(positions, dtype=np.float64, copy=True)
    for dim in range(3):
        if pbc_array[dim] and box[dim] > 0:
            pos[:, dim] = np.mod(pos[:, dim], box[dim])
    com = reference_atom_com(pos, box, pbc_array, ref_idx=0)
    diffs = pos - com
    for dim in range(3):
        if pbc_array[dim] and box[dim] > 0:
            diffs[:, dim] -= np.round(diffs[:, dim] / box[dim]) * box[dim]
    return np.sort(np.linalg.norm(diffs, axis=1))


def atoms_of_interest_fingerprint(
    positions: np.ndarray, cell: np.ndarray, pbc: np.ndarray, rnei: float, coord_thr: int
) -> np.ndarray:
    """Two-component fingerprint built from undercoordinated atoms.

    Component 1 (defect-internal): sorted distances from the circular-mean centre of
    mass of the undercoordinated atoms to each undercoordinated atom. Captures the
    internal geometry of the defect cluster.

    Component 2 (defect-position): minimum-image distance from the defect centre of
    mass to the bulk centre of mass (reference-atom-unwrapped). Disambiguates states
    with identical defect geometry but a different defect position.

    The circular mean makes component 1 invariant under any periodic representation of
    the same physical state. Falls back to reference-atom unwrapping when the circular
    mean is ill-conditioned (resultant < 0.1). Returns an empty array when no atom is
    undercoordinated.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Atom positions.
    cell : ndarray, shape (3, 3)
        Simulation cell (orthorhombic; the diagonal is used).
    pbc : ndarray, shape (3,)
        Per-dimension periodic-boundary flags.
    rnei : float
        First-neighbour cutoff used to count neighbours.
    coord_thr : int
        Atoms with fewer than ``coord_thr`` neighbours are "atoms of interest".

    """
    pbc_array = np.asarray(pbc, dtype=bool) if pbc is not None else np.array([True, True, True])
    cell_diag = np.diag(cell).astype(np.float64)
    pos = np.array(positions, dtype=np.float64, copy=True)

    # Wrap positions for PBC
    for dim in range(3):
        if pbc_array[dim] and cell_diag[dim] > 0:
            pos[:, dim] = np.mod(pos[:, dim], cell_diag[dim])

    # Build tree and count neighbours
    if np.all(pbc_array):
        tree = cKDTree(pos, boxsize=cell_diag.tolist())
    else:
        tree = cKDTree(pos)

    neighbor_lists = tree.query_ball_point(pos, rnei)
    counts = np.array([len(n) - 1 for n in neighbor_lists], dtype=np.int32)

    # Find interesting atom indices (undercoordinated)
    interesting_mask = counts < coord_thr
    if not np.any(interesting_mask):
        return np.array([], dtype=np.float64)

    defect_pos = pos[interesting_mask]

    # Component 1: defect-internal distances via the circular-mean defect COM
    defect_com, resultant = circular_mean_position(defect_pos, cell_diag, pbc_array)
    if np.any(resultant[pbc_array] < 0.1):
        # Fallback: reference-atom unwrapping for an ill-conditioned circular mean
        defect_com = reference_atom_com(defect_pos, cell_diag, pbc_array, ref_idx=0)

    diffs = defect_pos - defect_com
    for dim in range(3):
        if pbc_array[dim] and cell_diag[dim] > 0:
            diffs[:, dim] -= np.round(diffs[:, dim] / cell_diag[dim]) * cell_diag[dim]
    sorted_defect_dists = np.sort(np.linalg.norm(diffs, axis=1))

    # Component 2: defect position relative to the bulk
    bulk_com = reference_atom_com(pos, cell_diag, pbc_array, ref_idx=0)
    bulk_defect_diff = defect_com - bulk_com
    for dim in range(3):
        if pbc_array[dim] and cell_diag[dim] > 0:
            bulk_defect_diff[dim] -= np.round(bulk_defect_diff[dim] / cell_diag[dim]) * cell_diag[dim]
    bulk_defect_dist = np.linalg.norm(bulk_defect_diff)

    return np.append(sorted_defect_dists, bulk_defect_dist)


def fingerprint_tolerance(config: Config) -> float:
    """Return the Chebyshev tolerance for the fingerprint pre-filter.

    Uses ``[BASIN] fingerprint_tolerance`` when set, else the 0.5 default suited to
    COM-distance fingerprints.
    """
    if config.basin is not None and config.basin.fingerprint_tolerance is not None:
        return config.basin.fingerprint_tolerance
    return 0.5


def _derived_coord_thr(config: Config) -> "int | None":
    """Return the atoms-of-interest threshold: explicit override, else style-derived.

    The style-derived fallback reads ``coordination_threshold`` via getattr because
    the coordination-based AtomicEnvironment styles ship on a separate branch; on a
    config without them this simply returns None (COM fallback).
    """
    if config.basin is not None and config.basin.fingerprint_coordination_thr is not None:
        return config.basin.fingerprint_coordination_thr
    coord_thr = getattr(config.atomicenvironment, "coordination_threshold", None)
    if (config.atomicenvironment.style in ("coordination", "coordination/graph")
            and coord_thr is not None):
        return coord_thr + 1
    return None


def compute_fingerprint(
    config: Config, positions: np.ndarray, cell: np.ndarray, pbc: np.ndarray
) -> "np.ndarray | None":
    """Compute a structural fingerprint, dispatching from ``[BASIN] fingerprint_mode``.

    - ``off``: return None — the caller must skip the pre-filter entirely (benchmark
      baseline measuring raw deduplication cost).
    - ``com``: force the full COM-distance fingerprint.
    - ``atoms_of_interest``: force the undercoordinated-atoms fingerprint; the threshold
      is ``fingerprint_coordination_thr`` or coordination_threshold + 1, and a
      ValueError is raised when neither is derivable.
    - ``auto`` (default): atoms-of-interest when a threshold is derivable (explicit
      override or a coordination-based AtomicEnvironment style), else COM-distance.
    """
    mode = config.basin.fingerprint_mode if config.basin is not None else "auto"

    if mode == "off":
        return None
    if mode == "com":
        return com_fingerprint(positions, cell, pbc)

    coord_thr = _derived_coord_thr(config)
    if mode == "atoms_of_interest":
        if coord_thr is None:
            raise ValueError(
                "fingerprint_mode = 'atoms_of_interest' needs fingerprint_coordination_thr "
                "or a coordination-based AtomicEnvironment style to derive the threshold."
            )
        return atoms_of_interest_fingerprint(
            positions, cell, pbc,
            rnei=config.atomicenvironment.rnei,
            coord_thr=coord_thr,
        )

    # auto
    if coord_thr is not None:
        return atoms_of_interest_fingerprint(
            positions, cell, pbc,
            rnei=config.atomicenvironment.rnei,
            coord_thr=coord_thr,
        )
    return com_fingerprint(positions, cell, pbc)
