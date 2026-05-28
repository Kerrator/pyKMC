"""Module containing function to apply geometric transformations."""

__all__ = [
    "transform_positions",
    "translate",
    "push_towards",
    "compute_distances",
    "count_moved_atoms",
    "compute_delr_max",
    "compute_delr_l2",
]
import ase.geometry
import numpy as np


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
    positions: np.ndarray, displacement: np.ndarray, cell: np.ndarray
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
    positions += displacement
    positions = ase.geometry.wrap_positions(positions=positions, cell=cell, pbc=True)
    positions[positions < 0] = 0
    return positions


def push_towards(current_positions, target_positions, fraction=0.1, cell=None):
    displacement = target_positions - current_positions

    if cell is not None:
        box = np.diag(cell)
        displacement -= np.round(displacement / box) * box
        # unwrap target
        target_positions_unwrapped = current_positions + displacement
    else:
        target_positions_unwrapped = target_positions

    new_positions = current_positions + fraction * (
        target_positions_unwrapped - current_positions
    )

    if cell is not None:
        new_positions = ase.geometry.wrap_positions(
            positions=new_positions, cell=cell, pbc=[True, True, True]
        )
    return new_positions


def compute_distances(positions_1, positions_2, cell=None) -> np.ndarray:
    """Return per-atom distances between two configurations."""
    displacements = positions_2 - positions_1

    if cell is not None:
        _wrapped_displacements, distances = ase.geometry.find_mic(
            displacements, cell=cell, pbc=True
        )
        return np.asarray(distances)

    return np.linalg.norm(displacements, axis=1)


def count_moved_atoms(positions_1, positions_2, threshold, cell=None) -> int:
    """Return the number of atoms displaced by more than ``threshold``."""
    distances = compute_distances(positions_1, positions_2, cell=cell)
    return int(np.count_nonzero(distances > threshold))


def compute_delr_max(positions_1, positions_2, cell=None):
    distances = compute_distances(positions_1, positions_2, cell=cell)
    if distances.size == 0:
        return 0.0
    return float(np.max(distances))


def compute_delr_l2(positions_1, positions_2, cell=None):
    distances = compute_distances(positions_1, positions_2, cell=cell)
    return float(np.linalg.norm(distances))
