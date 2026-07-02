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
    if hasattr(pbc, "__iter__") and not np.all(pbc):
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
        if pbc_arr.ndim == 0:  # scalar bool -> per-dimension vector
            pbc_arr = np.full(3, bool(pbc_arr))
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
        if pbc_arr.ndim == 0:  # scalar bool -> per-dimension vector
            pbc_arr = np.full(3, bool(pbc_arr))

        #apply pbc only in periodic dimensions
        for i in range(3) :
            if pbc_arr[i]:
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


def align_positions_by_neighbors(
    neighbors_1: "np.ndarray | None",
    positions_1: np.ndarray,
    neighbors_2: "np.ndarray | None",
    positions_2: np.ndarray,
) -> "tuple[np.ndarray, np.ndarray, bool] | None":
    """Align two per-event position arrays onto their common atoms by atom id.

    Each active-event row stores its geometry (``saddle_positions`` /
    ``final_positions``) ordered positionally by the row's own ``neighbors``
    integer-id array: position row ``k`` belongs to absolute atom
    ``neighbors[k]``. Two rows may carry the same atoms in a **different order**
    (a recycled row keeps its event-time neighbour ordering while a fresh row is
    built from the current :class:`NeighborsList`) or may even span **different
    atom sets** (the system moved between the two events). A positional
    element-wise comparison of the two arrays therefore compares
    non-corresponding atoms. This helper builds the id->row maps and returns the
    two position subarrays restricted to the shared atoms, in a common atom-id
    order, so a caller can compare only corresponding atoms with
    :func:`compute_delr`.

    Parameters
    ----------
    neighbors_1 : np.ndarray or None
        Absolute atom ids for ``positions_1`` rows (the row's ``neighbors``
        column). ``None`` for a row whose neighbour ids were never stored.
    positions_1 : np.ndarray
        Shape (N1, 3) positions, row ``k`` belonging to atom ``neighbors_1[k]``.
    neighbors_2 : np.ndarray or None
        Absolute atom ids for ``positions_2`` rows.
    positions_2 : np.ndarray
        Shape (N2, 3) positions, row ``k`` belonging to atom ``neighbors_2[k]``.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, bool) or None
        ``(aligned_1, aligned_2, sets_equal)`` where ``aligned_1`` and
        ``aligned_2`` hold the positions of the shared atoms in the same
        atom-id order (shape (M, 3), M = number of shared atoms), and
        ``sets_equal`` is ``True`` iff the two neighbour sets are identical.
        Returns ``None`` (not comparable) when either ``neighbors`` array is
        ``None``, a ``neighbors`` length does not match its positions, or the
        two rows share no atom -- in every such case the caller must keep both
        rows.

    """
    if neighbors_1 is None or neighbors_2 is None:
        return None
    nb1 = np.asarray(neighbors_1, dtype=int)
    nb2 = np.asarray(neighbors_2, dtype=int)
    pos1 = np.asarray(positions_1)
    pos2 = np.asarray(positions_2)
    # A length mismatch means the stored ordering cannot be trusted to index the
    # positions; treat as not-comparable rather than risk a scrambled alignment.
    if nb1.shape[0] != pos1.shape[0] or nb2.shape[0] != pos2.shape[0]:
        return None

    map1 = {int(a): k for k, a in enumerate(nb1)}
    map2 = {int(a): k for k, a in enumerate(nb2)}
    common = [a for a in map1 if a in map2]  # deterministic: nb1 order
    if not common:
        return None
    idx1 = [map1[a] for a in common]
    idx2 = [map2[a] for a in common]
    sets_equal = set(map1) == set(map2)
    return pos1[idx1], pos2[idx2], sets_equal

