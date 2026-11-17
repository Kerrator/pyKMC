"""Rules that turn a saddle search's raw output into an event search result.

A search backend finds a saddle and the two minima on either side of it. What
pyKMC does with them is the same whichever backend found them, and lives here:

- error mapping: a backend error gives ``EVENT_NOT_FOUND``;
- acceptance: one of the two minima must lie within ``eventsearch.delr_thr``
  of the starting configuration, else ``EVENT_MINIMA_NOT_MATCH_POSITIONS``;
- barriers: saddle energy minus each minimum's energy;
- orientation: the closer minimum is the initial state, so the minima and the
  barriers are swapped when the backend's second minimum is the closer one.

The module also holds the plain-array steps the pARTn backend applies before
those rules: ARTn's move-atom pick and the mapping of active-volume results
back to the full system.

Nothing here needs LAMMPS or pARTn, so the rules can be tested on recorded
search outputs.
"""

from typing import Callable, NamedTuple, Optional

import numpy as np

from .result import EventSearchOutput, Err, ErrorInfo, ErrorType, Ok, Result

__all__ = [
    "SearchEndpoints",
    "search_result",
    "artn_move_atom",
    "map_to_full_system",
]


class SearchEndpoints(NamedTuple):
    """Energies and positions of a search's two minima and saddle.

    Minimum 1 and minimum 2 are in the order the backend returned them, before
    orientation. Positions are full-system arrays in pyKMC atom order.

    Attributes
    ----------
    e_min1 : float
        Total energy of the backend's first minimum.
    e_saddle : float
        Total energy of the saddle.
    e_min2 : float
        Total energy of the backend's second minimum.
    min1_positions : np.ndarray
        Positions of the backend's first minimum.
    saddle_positions : np.ndarray
        Positions of the saddle.
    min2_positions : np.ndarray
        Positions of the backend's second minimum.
    move_atom_index : int
        Atom picked by the backend's move-atom rule.

    """

    e_min1: float
    e_saddle: float
    e_min2: float
    min1_positions: np.ndarray
    saddle_positions: np.ndarray
    min2_positions: np.ndarray
    move_atom_index: int


def search_result(
    central_atom_index: int,
    error: tuple[int, str],
    read_distances: Callable[[], tuple[float, float]],
    read_endpoints: Callable[[], SearchEndpoints],
    delr_threshold: float,
    types: Optional[list] = None,
) -> Result[EventSearchOutput, ErrorInfo]:
    """Build the event search result from a backend's output.

    The backend's values are read through two callables, so a backend reads
    the distances only when it reports no error, and the energies and
    positions only when the search is accepted.

    Parameters
    ----------
    central_atom_index : int
        Index of the atom the search was centred on.
    error : tuple[int, str]
        The backend's error code and message; code 0 means it found a saddle
        and two minima.
    read_distances : Callable[[], tuple[float, float]]
        Returns the distances from the starting configuration to the
        backend's first and second minimum.
    read_endpoints : Callable[[], SearchEndpoints]
        Returns the energies, positions and move atom of the search.
    delr_threshold : float
        Largest distance from the starting configuration at which a minimum
        counts as the start (``eventsearch.delr_thr``).
    types : Optional[list]
        Atom types, passed through to the result.

    Returns
    -------
    Result[EventSearchOutput, ErrorInfo]
        The oriented event, or ``EVENT_NOT_FOUND`` with the backend's error as
        details, or ``EVENT_MINIMA_NOT_MATCH_POSITIONS`` with both distances.

    """
    if error[0] != 0:
        return Err(
            ErrorInfo(
                type=ErrorType.EVENT_NOT_FOUND,
                message="No event found",
                # ErrorInfo.details is typed str, but has always carried this tuple
                details=error,  # type: ignore[arg-type]
            )
        )

    delr1, delr2 = read_distances()
    if not (delr1 < delr_threshold or delr2 < delr_threshold):
        return Err(
            ErrorInfo(
                type=ErrorType.EVENT_MINIMA_NOT_MATCH_POSITIONS,
                message="delr1 and delr2 > at {}".format(delr_threshold),
                variables={"delr1": delr1, "delr2": delr2},
            )
        )

    ends = read_endpoints()
    dE_forward = ends.e_saddle - ends.e_min1
    dE_backward = ends.e_saddle - ends.e_min2

    if delr1 < delr2:
        return Ok(
            EventSearchOutput(
                central_atom_index=central_atom_index,
                dE_forward=dE_forward,
                dE_backward=dE_backward,
                min1_positions=ends.min1_positions,
                saddle_positions=ends.saddle_positions,
                min2_positions=ends.min2_positions,
                move_atom_index=ends.move_atom_index,
                types=types,
            )
        )
    return Ok(
        EventSearchOutput(
            central_atom_index=central_atom_index,
            dE_forward=dE_backward,
            dE_backward=dE_forward,
            min1_positions=ends.min2_positions,
            saddle_positions=ends.saddle_positions,
            min2_positions=ends.min1_positions,
            move_atom_index=ends.move_atom_index,
            types=types,
        )
    )


def artn_move_atom(
    min1_positions: np.ndarray, saddle_positions: np.ndarray, rcut: float
) -> np.intp:
    """Pick the atom that moves most from pARTn's first minimum to the saddle.

    Distances are raw, not minimum-image. A distance above ``rcut`` is taken
    as a jump across the periodic boundary and set to zero, so an atom whose
    hop crosses the boundary is never picked.

    Parameters
    ----------
    min1_positions : np.ndarray
        Positions of pARTn's first minimum, before orientation.
    saddle_positions : np.ndarray
        Positions of the saddle, in the same atom order.
    rcut : float
        ``atomicenvironment.rcut``.

    Returns
    -------
    np.intp
        Index of the picked atom in the arrays given.

    """
    dist = (min1_positions - saddle_positions) ** 2
    dist = dist.sum(axis=-1)
    dist = np.sqrt(dist)
    dist[dist > rcut] = 0
    return np.argmax(dist)


def map_to_full_system(
    local_positions: np.ndarray, atom_map: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    """Put active-volume positions back into a full-system array.

    Parameters
    ----------
    local_positions : np.ndarray
        Positions of the active-volume atoms, in active-volume order.
    atom_map : np.ndarray
        Full-system index of each active-volume atom.
    positions : np.ndarray
        Full-system positions; atoms outside the active volume keep these.

    Returns
    -------
    np.ndarray
        A copy of ``positions`` with the active-volume atoms replaced.

    """
    mapped = positions.copy()
    mapped[atom_map] = local_positions
    return mapped
