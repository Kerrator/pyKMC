"""Module containing function to apply geometric transformations."""

__all__ = ["transform_positions", "translate", "push_towards", "compute_delr"]
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
    positions: np.ndarray, displacement: np.ndarray, cell: np.ndarray, pbc=True
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
    pbc : bool or array-like of bool
        Periodic boundary conditions per dimension.

    Returns
    -------
    np.ndarray
        Translated and wrapped atomic positions, same shape as the input `positions`.

    """
    positions += displacement
    positions = ase.geometry.wrap_positions(positions=positions, cell=cell, pbc=pbc)
    if hasattr(pbc, '__iter__') and not np.all(pbc):
        for dim in range(3):
            if pbc[dim]:
                positions[:, dim] = np.where(positions[:, dim] < 0, 0, positions[:, dim])
    else:
        positions[positions < 0] = 0
    return positions


def push_towards(current_positions, target_positions, fraction = 0.1, cell = None, pbc=None) :
    displacement = target_positions - current_positions

    if cell is not None:
        if pbc is None:
            pbc = np.array([True, True, True])
        box = np.diag(cell)
        pbc_arr = np.asarray(pbc)
        if np.all(pbc_arr):
            displacement -= np.round(displacement / box) * box
        else:
            for dim in range(3):
                if pbc_arr[dim]:
                    displacement[:, dim] -= np.round(displacement[:, dim] / box[dim]) * box[dim]
        #unwrap target
        target_positions_unwrapped = current_positions + displacement
    else:
        target_positions_unwrapped = target_positions

    new_positions = current_positions + fraction * (target_positions_unwrapped - current_positions)

    if cell is not None :
        new_positions = ase.geometry.wrap_positions(positions=new_positions, cell=cell, pbc=pbc)
    return new_positions

def compute_delr(positions_1, positions_2, cell=None, pbc=None) :
    displacements = positions_2 - positions_1

    if cell is not None :
        if pbc is None:
            pbc = np.array([True, True, True])
        cell_lengths = np.linalg.norm(cell, axis=1)
        pbc_arr = np.asarray(pbc)

        #apply pbc only in periodic dimensions
        for i in range(3) :
            if pbc_arr[i]:
                displacements[:, i] -= cell_lengths[i] * np.round(displacements[:, i] / cell_lengths[i])

    # Calcul des normes des déplacements
    distances = np.linalg.norm(displacements, axis=1)

    # Retour du déplacement maximum
    delr = np.max(distances)

    return delr


