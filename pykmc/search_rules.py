"""Rules that turn a saddle search's raw output into an event search result.

A search backend finds a saddle and the two minima on either side of it. What
pyKMC does with them is the same whichever backend found them, and lives here:

- error mapping: a backend error gives ``EVENT_NOT_FOUND``;
- acceptance: one of the two minima must lie within ``eventsearch.delr_thr``
  of the starting configuration, else ``EVENT_MINIMA_NOT_MATCH_POSITIONS``;
- barriers: saddle energy minus each minimum's energy;
- orientation: the closer minimum is the initial state, so the minima, the
  barriers and the minima's energies are swapped when the backend's second
  minimum is the closer one.

The module also holds the plain-array steps a backend applies before those
rules: ARTn's move-atom pick, the corrected move-atom pick the dimer uses, and
the mapping of active-volume results back to the full system. After the rules,
:func:`add_search_details` attaches a search's optional details (backend,
seed, force calls, curvature, mode, failure reason), which go to the log and
the info file only; :func:`partn_search_details` reads pARTn's.

Nothing here needs LAMMPS or pARTn, so the rules can be tested on recorded
search outputs.
"""

from typing import Any, Callable, NamedTuple, Optional, Sequence, Union

import numpy as np
from ase.geometry import find_mic

from .dimer import _minimum_image
from .result import EventSearchOutput, Err, ErrorInfo, ErrorType, Ok, Result

__all__ = [
    "SearchEndpoints",
    "search_result",
    "artn_move_atom",
    "move_atom",
    "orient_mode",
    "map_to_full_system",
    "lammps_box",
    "add_search_details",
    "partn_search_details",
]

# Failure reason of a search whose minima are both away from the start; the
# info file's count of these searches has the same name.
MINIMA_NOT_AT_START = "minima_not_matching_positions"


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
        The oriented event, with its three energies, or ``EVENT_NOT_FOUND``
        with the backend's error as details, or
        ``EVENT_MINIMA_NOT_MATCH_POSITIONS`` with both distances.

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
                e_initial=ends.e_min1,
                e_saddle=ends.e_saddle,
                e_final=ends.e_min2,
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
            e_initial=ends.e_min2,
            e_saddle=ends.e_saddle,
            e_final=ends.e_min1,
        )
    )


def artn_move_atom(
    min1_positions: np.ndarray,
    saddle_positions: np.ndarray,
    cell: np.ndarray,
    pbc: Union[bool, Sequence[bool]],
) -> np.intp:
    """Pick the atom that moves most from pARTn's first minimum to the saddle.

    Displacements are taken with the minimum image (``ase.geometry.find_mic``),
    as pARTn returns wrapped coordinates and the atom that hops may cross the
    cell boundary.

    Parameters
    ----------
    min1_positions : np.ndarray
        Positions of pARTn's first minimum, before orientation.
    saddle_positions : np.ndarray
        Positions of the saddle, in the same atom order.
    cell : np.ndarray
        Cell vectors as rows, (3, 3).
    pbc : bool or sequence of bool
        Periodicity, as ``find_mic`` takes it.

    Returns
    -------
    np.intp
        Index of the picked atom in the arrays given.

    """
    _, dist = find_mic(min1_positions - saddle_positions, cell, pbc=pbc)
    return np.argmax(dist)


def move_atom(
    initial_positions: np.ndarray,
    saddle_positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> int:
    """Pick the atom that moves most from the initial minimum to the saddle.

    The corrected rule: displacements are taken from the oriented initial
    minimum (the minimum at the start, after orientation), with the minimum
    image on the periodic axes, so a hop across the periodic boundary picks
    the hopping atom.

    Parameters
    ----------
    initial_positions : np.ndarray
        Positions of the oriented initial minimum, (N, 3).
    saddle_positions : np.ndarray
        Positions of the saddle, in the same atom order.
    cell : np.ndarray
        Cell vectors as rows, (3, 3).
    pbc : np.ndarray
        Periodicity of each cell axis, (3,) booleans.

    Returns
    -------
    int
        Index of the picked atom; the lowest index on a tie.

    """
    displacements = _minimum_image(
        np.asarray(saddle_positions, dtype=float)
        - np.asarray(initial_positions, dtype=float),
        cell,
        pbc,
    )
    return int(np.argmax(np.linalg.norm(displacements, axis=1)))


def orient_mode(
    mode: np.ndarray,
    initial_positions: np.ndarray,
    saddle_positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> np.ndarray:
    """Return the unstable mode as a unit vector pointing from the initial minimum to the saddle.

    Parameters
    ----------
    mode : np.ndarray
        Unstable mode at the saddle, (N, 3), any sign and norm.
    initial_positions : np.ndarray
        Positions of the oriented initial minimum, (N, 3).
    saddle_positions : np.ndarray
        Positions of the saddle, in the same atom order and frame.
    cell : np.ndarray
        Cell vectors as rows, (3, 3), in the same frame.
    pbc : np.ndarray
        Periodicity of each cell axis, (3,) booleans.

    Returns
    -------
    np.ndarray
        ``mode`` or ``-mode``, divided by its norm, so that its dot product
        with the minimum-image displacement from the initial minimum to the
        saddle is not negative. A zero mode is returned unchanged.

    """
    mode = np.array(mode, dtype=float)
    norm = float(np.linalg.norm(mode))
    if norm == 0.0:
        return mode
    displacements = _minimum_image(
        np.asarray(saddle_positions, dtype=float)
        - np.asarray(initial_positions, dtype=float),
        cell,
        pbc,
    )
    sign = -1.0 if np.vdot(mode, displacements) < 0 else 1.0
    return sign * mode / norm


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


def lammps_box(
    boxlo: np.ndarray, boxhi: np.ndarray, xy: float, yz: float, xz: float
) -> np.ndarray:
    """Return a LAMMPS box as cell vectors in rows, (3, 3), in LAMMPS's frame.

    Parameters
    ----------
    boxlo, boxhi : np.ndarray
        Lower and upper box bounds, as LAMMPS's ``extract_box`` returns them.
    xy, yz, xz : float
        Tilt factors, in ``extract_box``'s order.

    Returns
    -------
    np.ndarray
        Rows ``a = (lx, 0, 0)``, ``b = (xy, ly, 0)``, ``c = (xz, yz, lz)``.

    """
    lx, ly, lz = np.subtract(boxhi, boxlo)
    return np.array([[lx, 0.0, 0.0], [xy, ly, 0.0], [xz, yz, lz]], dtype=float)


def add_search_details(
    result: Result[EventSearchOutput, ErrorInfo], **details: Any
) -> Result[EventSearchOutput, ErrorInfo]:
    """Attach a search's optional details to its result.

    The details are the optional fields of
    :class:`~pykmc.result.EventSearchOutput` from ``backend`` on. They go to the
    log and the info file only; the reference table does not read them. Those
    given as None are left unset.

    On an event, each detail is set on the
    :class:`~pykmc.result.EventSearchOutput`. A failed search is an error, so
    its details go into ``ErrorInfo.variables`` under the same names, without
    the mode (a failed search has no event to refine), and with a
    ``failure_reason``: the backend's error message for ``EVENT_NOT_FOUND``,
    ``"minima_not_matching_positions"`` for
    ``EVENT_MINIMA_NOT_MATCH_POSITIONS``.

    Parameters
    ----------
    result : Result[EventSearchOutput, ErrorInfo]
        A result built by :func:`search_result`; it is changed in place.
    **details : Any
        Values of the optional fields, by name.

    Returns
    -------
    Result[EventSearchOutput, ErrorInfo]
        ``result``.

    """
    details = {name: value for name, value in details.items() if value is not None}
    if isinstance(result, Ok):
        output = result.ok_value()
        for name, value in details.items():
            if not hasattr(output, name):
                raise AttributeError(f"EventSearchOutput has no field {name!r}")
            setattr(output, name, value)
        return result
    info = result.err_value()
    details.pop("mode", None)
    if info.type == ErrorType.EVENT_NOT_FOUND:
        # The info file drops empty strings, so a failure never gets one.
        reason = str(info.details[1]) or "no_event_found"  # type: ignore[index]
    else:
        reason = MINIMA_NOT_AT_START
    info.variables = {**(info.variables or {}), "failure_reason": reason, **details}
    return result


def partn_search_details(
    artn: Any,
    accepted: bool,
    zseed: int,
    cell: np.ndarray,
    pbc: np.ndarray,
    to_full_system: Callable[[np.ndarray], np.ndarray],
) -> dict[str, Any]:
    """Read a pARTn search's optional details, once the search has ended.

    pARTn counts its steps from 0, one per force call, and keeps the step of
    its last call in ``nevalf`` and that of the saddle in ``nevalf_sad``; its
    report's ``evalf``, which the 2026-11-09 baseline counted, is one more. So
    the force calls are ``nevalf_sad + 1`` up to the saddle and
    ``nevalf - nevalf_sad`` after it, or ``nevalf + 1`` up to a failure before
    any saddle (and 0 after it). ``nevalf`` keeps its value across searches,
    so the caller sets it to -1 before the search; a search that never took a
    step leaves it there and reports no count. pARTn prints no step line for
    the calls of a failing last step, so on a failure the count can exceed the
    last printed ``evalf``; it equals the "number of steps" pARTn prints, plus
    one.

    pARTn resets ``zseed`` to 0 when the search ends, so the seed is ``zseed``
    when it is set, and unset when it is 0 (pARTn then draws a seed from the
    clock and writes it only to its own output file).

    The curvature is ``eigval_sad``, the Lanczos eigenvalue pARTn holds at the
    saddle step, and the mode ``eigen_sad``, its last Lanczos vector; neither is
    recomputed at the converged saddle. The mode is read only for an accepted
    search, oriented from the initial minimum to the saddle, and mapped to the
    full system (zero on atoms outside the active volume).

    Parameters
    ----------
    artn : pypARTn.artn
        The pARTn handle of the search, after the search; its ``nevalf`` was
        set to -1 before it.
    accepted : bool
        Whether the shared rules accepted the search.
    zseed : int
        ``[pARTn] zseed``, as given to pARTn.
    cell : np.ndarray
        The search's cell vectors as rows, (3, 3), in pARTn's frame.
    pbc : np.ndarray
        Periodicity of each cell axis, (3,).
    to_full_system : Callable[[np.ndarray], np.ndarray]
        Maps an (n, 3) array in pARTn's atom order and frame to a full-system
        (N, 3) array in pyKMC atom order.

    Returns
    -------
    dict[str, Any]
        Keyword arguments for :func:`add_search_details`, as plain Python
        numbers and an array.

    """
    details: dict[str, Any] = {
        "backend": "partn",
        "seed": int(zseed) if zseed else None,
    }
    calls_last = int(artn.extract("nevalf"))
    if calls_last < 0:  # still the -1 set before the search: no step taken
        return details
    if not bool(artn.extract("has_sad")):
        details["force_calls"] = calls_last + 1
        details["force_calls_endpoints"] = 0
        return details
    calls_saddle = int(artn.extract("nevalf_sad"))
    details["e_saddle"] = float(artn.extract("etot_sad"))
    details["curvature"] = float(artn.extract("eigval_sad"))
    details["force_calls"] = calls_saddle + 1
    details["force_calls_endpoints"] = calls_last - calls_saddle
    if accepted:
        if artn.extract("delr_min1") < artn.extract("delr_min2"):
            initial = artn.extract("tau_min1")
        else:
            initial = artn.extract("tau_min2")
        mode = orient_mode(
            artn.extract("eigen_sad"), initial, artn.extract("tau_sad"), cell, pbc
        )
        details["mode"] = to_full_system(mode)
    return details
