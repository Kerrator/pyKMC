"""Module containing function to apply geometric transformations."""

__all__ = [
    "transform_positions",
    "translate",
    "push_towards",
    "compute_delr",
    "per_atom_displacement",
    "minimum_image_distance",
    "normalize_pbc",
    "wrap_positions",
    "minimum_image_displacement",
]
import ase.geometry
import numpy as np


def normalize_pbc(pbc: bool | np.ndarray) -> np.ndarray:
    """Return an owned three-axis boolean array, rejecting ambiguous shapes."""
    axes = np.asarray(pbc)
    if axes.ndim == 0 and axes.dtype.kind == "b":
        return np.full(3, bool(axes), dtype=bool)
    if axes.shape != (3,) or axes.dtype.kind != "b":
        raise ValueError("PBC must be a boolean scalar or three boolean axes")
    return axes.copy()


def wrap_positions(
    positions: np.ndarray, cell: np.ndarray, pbc: bool | np.ndarray = True
) -> np.ndarray:
    """Wrap periodic coordinates without changing open-axis displacements."""
    axes = normalize_pbc(pbc)
    values = np.array(positions, dtype=float, copy=True)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("positions must be a finite (N, 3) array")
    if not axes.any():
        return values
    # eps=0 keeps periodic orthorhombic coordinates in [0, L), without a
    # Cartesian clamp that would destroy negative nonperiodic coordinates.
    return ase.geometry.wrap_positions(values, cell=cell, pbc=axes, eps=0)


def minimum_image_displacement(
    displacement: np.ndarray, cell: np.ndarray | None, pbc: bool | np.ndarray = True
) -> np.ndarray:
    """Return displacement vectors using only the declared periodic axes."""
    axes = normalize_pbc(pbc)
    values = np.array(displacement, dtype=float, copy=True)
    if not np.isfinite(values).all() or values.shape[-1:] != (3,):
        raise ValueError("displacement must contain finite three-vectors")
    if cell is None or not axes.any():
        return values
    return ase.geometry.find_mic(values, cell=cell, pbc=axes)[0]


def transform_positions(
    positions: np.ndarray,
    transformation_matrix: np.ndarray,
    translation_matrix: np.ndarray,
    permutation_matrix: np.ndarray,
) -> np.ndarray:
    """Apply rotation, translation and permutation to all positions.

    Parameters
    ----------
    positions : np.ndarray
        positions to transform.
    transformation_matrix : np.ndarray
        transformation matrix (e.g. rotation).
    translation_matrix : np.ndarray
        translation matrix
    permutation_matrix : np.ndarray
        permutation matrix.

    Returns
    -------
    np.ndarray
        The transformed positions.

    """
    transform_positions = positions @ transformation_matrix.T + translation_matrix
    return transform_positions[permutation_matrix]


def translate(
    positions: np.ndarray,
    displacement: np.ndarray,
    cell: np.ndarray,
    pbc: bool | np.ndarray = True,
) -> np.ndarray:
    """Translate atomic positions by a displacement vector and apply periodic wrapping.

    Parameters
    ----------
    positions : np.ndarray
        Array of atomic positions with shape (N, 3), where N is the number of atoms.
    displacement : np.ndarray
        Displacement vector of shape (3,) to be added to each position.
    cell : np.ndarray
        Simulation cell (3x3 matrix) defining the periodic boundaries.

    Returns
    -------
    np.ndarray
        Translated and wrapped atomic positions, same shape as the input `positions`.

    """
    return wrap_positions(np.asarray(positions) + displacement, cell=cell, pbc=pbc)


def push_towards(
    current_positions, target_positions, fraction=0.1, cell=None, pbc=True
):
    displacement = minimum_image_displacement(
        np.asarray(target_positions) - current_positions, cell, pbc
    )
    new_positions = np.asarray(current_positions) + fraction * displacement
    if cell is not None:
        new_positions = wrap_positions(new_positions, cell=cell, pbc=pbc)
    return new_positions


def compute_delr(positions_1, positions_2, cell=None, pbc=True):
    displacements = minimum_image_displacement(
        np.asarray(positions_2) - positions_1, cell, pbc
    )
    distances = np.linalg.norm(displacements, axis=1)
    return np.max(distances)


def per_atom_displacement(
    positions_pre: np.ndarray,
    positions_post: np.ndarray,
    cell: np.ndarray,
) -> np.ndarray:
    """Per-atom PBC-aware displacement magnitude (orthorhombic minimum-image).

    Same minimum-image trick as `compute_delr`, but returns the full per-atom
    array of Euclidean distances instead of just the maximum.

    Parameters
    ----------
    positions_pre : np.ndarray
        Shape (N, 3) positions before the displacement.
    positions_post : np.ndarray
        Shape (N, 3) positions after the displacement.
    cell : np.ndarray
        3x3 simulation cell (orthorhombic; row-wise lattice vectors).

    Returns
    -------
    np.ndarray
        Shape (N,) of per-atom displacement magnitudes in Angstroms.

    """
    disp = positions_post - positions_pre
    cell_lengths = np.linalg.norm(cell, axis=1)
    for i in range(3):
        disp[:, i] -= cell_lengths[i] * np.round(disp[:, i] / cell_lengths[i])
    return np.linalg.norm(disp, axis=1)


def minimum_image_distance(
    position_a: np.ndarray,
    position_b: np.ndarray,
    cell: np.ndarray,
) -> float:
    """PBC minimum-image Euclidean distance between two positions (orthorhombic).

    Single-pair counterpart of `per_atom_displacement`: applies the same
    per-axis minimum-image wrap to the separation vector and returns its norm.

    Parameters
    ----------
    position_a : np.ndarray
        Shape (3,) first position.
    position_b : np.ndarray
        Shape (3,) second position.
    cell : np.ndarray
        3x3 simulation cell (orthorhombic; row-wise lattice vectors).

    Returns
    -------
    float
        Minimum-image distance in Angstroms.

    """
    dvec = position_b - position_a
    cell_lengths = np.linalg.norm(cell, axis=1)
    for i in range(3):
        dvec[i] -= cell_lengths[i] * np.round(dvec[i] / cell_lengths[i])
    return float(np.linalg.norm(dvec))
