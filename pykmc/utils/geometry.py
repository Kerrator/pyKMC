"""Module containing function to apply geometric transformations."""

__all__ = [
    "transform_positions",
    "translate",
    "push_towards",
    "compute_delr",
    "per_atom_displacement",
    "minimum_image_distance",
    "event_movers",
    "reconstruction_matches",
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


def push_towards(current_positions, target_positions, fraction = 0.1, cell = None) : 
    displacement = target_positions - current_positions

    if cell is not None:
        box = np.diag(cell)
        displacement -= np.round(displacement / box) * box
        #unwrap target
        target_positions_unwrapped = current_positions + displacement
    else:
        target_positions_unwrapped = target_positions

    new_positions = current_positions + fraction * (target_positions_unwrapped - current_positions)

    if cell is not None : 
        new_positions = ase.geometry.wrap_positions(positions=new_positions, cell=cell, pbc=[True, True, True])
    return new_positions

def compute_delr(positions_1, positions_2, cell=None) : 
    displacements = positions_2 - positions_1

    if cell is not None : 
        cell_lengths = np.linalg.norm(cell, axis=1)  

        #apply pbc 

        for i in range(3) : 
            displacements[:, i] -= cell_lengths[i] * np.round(displacements[:, i] / cell_lengths[i])
    
    # Calcul des normes des déplacements
    distances = np.linalg.norm(displacements, axis=1)

    # Retour du déplacement maximum
    delr = np.max(distances)

    return delr


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


def event_movers(
    event_displacement: np.ndarray,
    n_movers: int,
    matching_thr: float,
) -> np.ndarray:
    """Row indices of the ``n_movers`` atoms that move most during an event.

    The reconstruction acceptance check is restricted to the atoms that actually
    participate in the event (largest min1->min2 displacement); peripheral atoms
    that barely move must not veto an otherwise correct reconstruction. Falls
    back to the single largest mover when no atom exceeds ``matching_thr`` (a
    degenerate, sub-threshold event).

    Parameters
    ----------
    event_displacement : np.ndarray
        Shape (N,) per-atom min1->min2 displacement magnitudes over the rcut shell.
    n_movers : int
        Number of top movers to keep (``ReconstructionConfig.n_movers``).
    matching_thr : float
        Displacement (Angstrom) above which an atom counts as a participant.

    Returns
    -------
    np.ndarray
        Row indices (into the rcut shell) of the top movers, descending.

    """
    significant = np.where(event_displacement > matching_thr)[0]
    if len(significant) == 0:  # degenerate event: keep the single largest mover
        significant = np.array([int(np.argmax(event_displacement))])
    order = significant[np.argsort(event_displacement[significant])[::-1]]
    return order[: n_movers]


def reconstruction_matches(
    discrepancy: np.ndarray,
    movers: np.ndarray,
    matching_thr: float,
    shell_thr: float,
) -> "tuple[bool, float, float]":
    """Decide whether a reconstructed minimum matches the expected geometry.

    Two-tier rule, shared by the serial (host) and engine (basin wavefront)
    reconstruction paths so they accept/reject identically:

    * the event ``movers`` must each land within the tight ``matching_thr``;
    * the *whole* rcut shell must land within the looser ``shell_thr`` -- this
      catches a peripheral (non-mover) atom that relaxed into a **distinct** site
      (a large displacement) while tolerating the small wiggle of atoms that
      merely settled around the event. Without it the movers-only check would
      accept a reconstruction that landed on a different overall state.

    Parameters
    ----------
    discrepancy : np.ndarray
        Shape (N,) per-atom displacement between the reconstructed and the
        expected (supposed) minimum, over the whole rcut shell.
    movers : np.ndarray
        Row indices of the event movers (from :func:`event_movers`).
    matching_thr : float
        Tight threshold (Angstrom) the movers must satisfy.
    shell_thr : float
        Looser threshold (Angstrom) the whole shell must satisfy.

    Returns
    -------
    tuple of (bool, float, float)
        ``(ok, delr_movers, delr_shell)`` -- acceptance flag and the two maxima.

    """
    delr_movers = float(discrepancy[movers].max())
    delr_shell = float(discrepancy.max())
    ok = delr_movers <= matching_thr and delr_shell <= shell_thr
    return ok, delr_movers, delr_shell

