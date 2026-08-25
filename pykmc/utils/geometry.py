"""Module containing function to apply geometric transformations."""

__all__ = [
    "transform_positions",
    "translate",
    "push_towards",
    "adaptive_push_fraction",
    "compute_delr",
    "per_atom_displacement",
    "minimum_image_distance",
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

def adaptive_push_fraction(
    saddle_positions: np.ndarray,
    target_min_positions: np.ndarray,
    other_min_positions: np.ndarray,
    push_fraction: float,
    mover_landing_fraction: "float | None",
    cell: np.ndarray,
    pbc: "np.ndarray | list[bool] | bool | None" = None,
) -> float:
    """Fraction for the reconstruction push from the saddle toward one minimum.

    Shared by the serial (host) :meth:`Reconstruction.reconstruct` and the
    engine (basin) ``_basin_reconstruct_impl`` so the two paths push -- and
    therefore accept/reject -- identically; the rule must not drift between
    them, which is why it lives here rather than inline at the four call sites.

    Two rules, selected by ``mover_landing_fraction``:

    * ``None`` (default) -- return ``push_fraction`` unchanged. This is the
      historical fixed push and is bit-for-bit what every existing input file
      already gets, on every reconstruction, main loop included.
    * a fraction in (0, 1) -- keep the uniform "fraction of the saddle->minimum
      displacement" semantics of :func:`push_towards`, but raise the fraction to
      ``max(push_fraction, 1 - mover_landing_fraction / s)`` where ``s`` is the
      event mover's saddle offset from the target minimum expressed as a share
      of the min1<->min2 hop. The mover then lands exactly
      ``mover_landing_fraction`` of the hop from the minimum being sought,
      instead of the ~0.5 a fixed 0.15 push leaves on a saddle sitting near the
      middle of the hop (from which the minimise can fall back the wrong way and
      the reconstruction is rejected as INVALID_MIN1/INVALID_MIN2).

    The fraction is uniform over the shell precisely so the non-mover atoms keep
    a proportional share ``1 - fraction`` of their own saddle relaxation. Placing
    every atom at a fixed fraction of *its own* min1->min2 segment would instead
    drop the shell onto the minimum -- for a localized hop the shell's min1 and
    min2 coincide to a few hundredths of an Angstrom -- and the minimise would no
    longer have to find its way back, which is the connectivity evidence the
    push+minimise exists to produce.

    The mover is the atom with the largest min1->min2 displacement (the same
    "event mover" notion as :func:`event_movers`); it carries the longest hop and
    hence the tightest absolute landing requirement. Degenerate geometries fall
    back to ``push_fraction``: a vanishing hop, or a saddle further from the
    target minimum than the whole hop (``s > 1``), is not a saddle between two
    minima, and the formula would otherwise demand a fraction approaching 1, i.e.
    landing the whole shell on the minimum. Those events are rejected by the
    containment / acceptance gates instead.

    Parameters
    ----------
    saddle_positions : np.ndarray
        Shape (N, 3) saddle positions over the rcut shell (shell row order).
    target_min_positions : np.ndarray
        Shape (N, 3) positions of the minimum being pushed toward.
    other_min_positions : np.ndarray
        Shape (N, 3) positions of the opposite minimum of the same event; only
        the mover's hop length is read from it.
    push_fraction : float
        ``ReconstructionConfig.push_fraction`` -- the legacy fixed fraction, and
        the floor of the adaptive one.
    mover_landing_fraction : float or None
        ``ReconstructionConfig.mover_landing_fraction``. ``None`` selects the
        legacy rule.
    cell : np.ndarray
        3x3 simulation cell (orthorhombic; row-wise lattice vectors).
    pbc : np.ndarray or list[bool] or bool or None, optional
        Per-axis periodicity, threaded through to the minimum-image metric.

    Returns
    -------
    float
        Fraction to hand to :func:`push_towards`. Always >= ``push_fraction``
        and, when the adaptive rule applies, <= ``1 - mover_landing_fraction``.

    """
    if mover_landing_fraction is None:
        return float(push_fraction)

    target = np.asarray(target_min_positions, dtype=float)
    other = np.asarray(other_min_positions, dtype=float)
    saddle = np.asarray(saddle_positions, dtype=float)
    if target.size == 0:  # degenerate/empty shell: caller's gates reject it
        return float(push_fraction)

    hops = per_atom_displacement(target.copy(), other.copy(), cell, pbc)
    mover = int(np.argmax(hops))
    hop = float(hops[mover])
    if hop <= 0.0:  # min1 == min2: no hop to land a fraction of
        return float(push_fraction)

    offset = float(
        per_atom_displacement(
            target[mover][None, :].copy(), saddle[mover][None, :].copy(), cell, pbc
        )[0]
    )
    s = offset / hop
    if s <= mover_landing_fraction or s > 1.0:
        # Already close enough, or not a between-the-minima saddle at all.
        return float(push_fraction)
    return float(max(push_fraction, 1.0 - mover_landing_fraction / s))


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
    pbc: "np.ndarray | list[bool] | bool | None" = None,
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
    pbc : np.ndarray or list[bool] or bool or None, optional
        Per-axis periodicity. ``None`` (default) applies the minimum-image
        wrap on all three axes, preserving the historical full-PBC behaviour.
        On a non-periodic axis (e.g. the free-surface direction of a slab) the
        wrap is skipped so an across-surface displacement is reported at its
        true magnitude rather than folded back into the cell.

    Returns
    -------
    np.ndarray
        Shape (N,) of per-atom displacement magnitudes in Angstroms.

    """
    disp = positions_post - positions_pre
    cell_lengths = np.linalg.norm(cell, axis=1)
    if pbc is None:
        pbc_arr = np.array([True, True, True])
    else:
        pbc_arr = np.asarray(pbc)
        if pbc_arr.ndim == 0:  # scalar bool -> per-dimension vector
            pbc_arr = np.full(3, bool(pbc_arr))
    for i in range(3):
        if pbc_arr[i]:
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
    """Row indices of the atoms that genuinely participate in an event.

    The reconstruction acceptance check is restricted to the atoms that actually
    participate in the event (largest min1->min2 displacement); peripheral atoms
    that barely move must not veto an otherwise correct reconstruction. Every
    such participant must be tight-checked, otherwise a genuine 4th+ mover of a
    collective event could land on a *distinct* nearby site (0.1-1.0 A off) yet
    be accepted only against the loose whole-shell bound -- a wrong state (and a
    wrong barrier on the KMC clock).

    The returned set is the **union** of

    * **every** atom whose ``event_displacement`` exceeds ``matching_thr`` (a real
      participant, regardless of rank -- this is what the old top-``n_movers``
      *cap* used to truncate), and
    * a top-``n_movers`` *floor* that only bites when **no** atom exceeds
      ``matching_thr`` (a degenerate, sub-threshold event), so a peripheral atom
      that barely moved is never dragged into the tight check as long as at least
      one genuine participant exists.

    This is a strict superset of the historical set (which was the top
    ``min(n_movers, #participants)`` participants): when participants exist the
    new rule keeps *all* of them; when none do it keeps the top-``n_movers`` as a
    floor. Returned indices are ordered by descending displacement. Returns an
    empty array when ``event_displacement`` is empty so callers can convert it to
    a graceful reject instead of crashing on ``np.argmax`` of an empty sequence.

    Parameters
    ----------
    event_displacement : np.ndarray
        Shape (N,) per-atom min1->min2 displacement magnitudes over the rcut shell.
    n_movers : int
        Floor on the number of top movers kept when no atom exceeds
        ``matching_thr`` (``ReconstructionConfig.n_movers``); NOT a cap -- all
        participants above ``matching_thr`` are kept even when they exceed
        ``n_movers``.
    matching_thr : float
        Displacement (Angstrom) above which an atom counts as a participant.

    Returns
    -------
    np.ndarray
        Row indices (into the rcut shell) of the movers, descending by
        displacement. Empty when ``event_displacement`` is empty.

    """
    disp = np.asarray(event_displacement, dtype=float)
    if disp.size == 0:  # degenerate/empty event: caller must reject gracefully
        return np.empty(0, dtype=int)
    order = np.argsort(disp)[::-1]  # all rows, descending displacement
    n_participants = int((disp > matching_thr).sum())
    if n_participants == 0:  # sub-threshold event: keep the top-n_movers floor
        return order[: max(1, min(n_movers, disp.size))]
    return order[:n_participants]  # tight-check every genuine participant


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
        An empty ``discrepancy`` or empty ``movers`` yields a graceful
        non-match ``(False, inf, inf)`` so the caller rejects the reconstruction
        rather than crashing on ``max()`` of an empty sequence.

    """
    disc = np.asarray(discrepancy, dtype=float)
    mv = np.asarray(movers, dtype=int)
    if disc.size == 0 or mv.size == 0:  # degenerate: reject, never crash
        return False, float("inf"), float("inf")
    delr_movers = float(disc[mv].max())
    delr_shell = float(disc.max())
    ok = delr_movers <= matching_thr and delr_shell <= shell_thr
    return ok, delr_movers, delr_shell


def event_contained(
    central_atom: "int | None",
    neighbors: "np.ndarray | list[int]",
    movers: np.ndarray,
    min1_positions: np.ndarray,
    saddle_positions: np.ndarray,
    min2_positions: np.ndarray,
    cell: np.ndarray,
    rcut: float,
    containment_margin: float,
) -> "tuple[bool, float, float]":
    """Whether the event stays inside the stored ``rcut`` neighbourhood.

    Shared by the serial (host) :meth:`Reconstruction.reconstruct` and the engine
    (basin wavefront) ``_basin_reconstruct_impl`` so the two paths accept/reject
    identically -- the containment math must not drift between them, which is why
    it lives here rather than being duplicated inline in both.

    If a genuine event mover reaches the outer ``rcut`` shell the stored
    neighbourhood is too small: the frozen far field would truncate the relaxation
    and produce a spurious invalid-minimum rejection. The extent is measured over
    the **whole** transition -- ``min1``, the saddle, and ``min2`` -- because an
    *outward* event can sit safely inside ``rcut`` at ``min1`` yet reach or cross
    the shell edge at the saddle or ``min2`` (measuring ``min1`` alone would let it
    pass and then be truncated during the ``min2`` minimize).

    The reference point is the central atom's ``min1`` position, located by its
    absolute id in ``neighbors``. An **absent** central row is treated as
    *not contained* (a reject), not silently skipped: the guard is the only
    geometric sanity check on the reconstruction, and a corrupted/permuted
    ``neighbors`` column that dropped the central id must fail closed rather than
    bypass the check.

    Parameters
    ----------
    central_atom : int or None
        Absolute id of the event's central atom. ``None`` disables the guard
        (returns contained), mirroring the historical ``central_atom is None``
        callers that never had a centre to measure from.
    neighbors : np.ndarray or list[int]
        Absolute atom ids for the shell rows; ``central_atom`` is located here.
    movers : np.ndarray
        Row indices (into the shell) of the event movers.
    min1_positions : np.ndarray
        Shape (N, 3) supposed min1 positions over the shell.
    saddle_positions : np.ndarray
        Shape (N, 3) saddle positions over the shell (the mover rows).
    min2_positions : np.ndarray
        Shape (N, 3) supposed min2 positions over the shell.
    cell : np.ndarray
        3x3 simulation cell (orthorhombic).
    rcut : float
        Neighbourhood cutoff radius (``atomicenvironment.rcut``).
    containment_margin : float
        Margin (Angstrom) subtracted from ``rcut`` to form the limit.

    Returns
    -------
    tuple of (bool, float, float)
        ``(contained, max_mover_r, rcut_limit)``. ``contained`` is ``True`` when
        every mover stays within ``rcut - containment_margin`` of the central
        atom over the whole path. When ``central_atom`` is ``None`` the guard is
        disabled and returns ``(True, 0.0, 0.0)`` without touching ``rcut`` (which
        may be ``None`` for the disabled guard). An absent central row or empty
        ``movers`` returns ``(False, inf, rcut_limit)``.

    """
    if central_atom is None:  # no centre to measure from: historical no-op
        return True, 0.0, 0.0
    rcut_limit = float(rcut) - float(containment_margin)
    mv = np.asarray(movers, dtype=int)
    central_rows = np.where(np.asarray(neighbors) == central_atom)[0]
    #Fail closed: an absent central row means the stored ordering lost the
    #central atom (corruption/permutation). Skipping would bypass the only
    #geometric sanity check, so treat it as not-contained instead.
    if central_rows.size == 0 or mv.size == 0:
        return False, float("inf"), rcut_limit
    central_pos = min1_positions[central_rows[0]]
    max_mover_r = 0.0
    for state in (min1_positions, saddle_positions, min2_positions):
        for m in mv:
            r = minimum_image_distance(central_pos, state[m], cell)
            if r > max_mover_r:
                max_mover_r = r
    return max_mover_r <= rcut_limit, float(max_mover_r), rcut_limit


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


