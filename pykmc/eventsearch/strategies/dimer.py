"""Dimer event search strategy.

The dimer finds the events and pARTn refines them. Each search runs as one
`dimer_search` operation on a worker (:mod:`pykmc.engine.dimer_search`), and
its raw output goes through the shared search rules on rank 0.
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, Protocol

import numpy as np
from ase.geometry import find_mic

from ...atomic_environment import AtomicEnvironment
from ...engine.dimer_search import DimerSearchExtension
from ...search_rules import SearchEndpoints, move_atom, search_result
from .base import EventSearchStrategy

if TYPE_CHECKING:
    from ...dimer import DimerResult
    from ...engine import EngineExtension
    from ...manager import Manager
    from ...result import ErrorInfo, EventSearchOutput, Result
    from ...system import System
    from ..eventsearch import SearchRequest

__all__ = ["DimerStrategy", "dimer_search_result"]


class DimerSearchConfig(Protocol):
    """Configuration read by `DimerStrategy`."""

    control: Any
    activevolume: Any
    atomicenvironment: Any
    eventsearch: Any
    dimer: Any
    frozen_atoms: Any


class DimerStrategy(EventSearchStrategy):
    """Event search with the dimer, run by the `dimer_search` engine extension.

    Each request carries a seed, drawn by the facade, and the movable atoms:
    those within `rmov` of the central atom with the active volume on, every
    atom with it off, and in both cases none of `[frozen_atoms]`.
    """

    name = "dimer"
    uses_seed = True

    def __init__(self, config: DimerSearchConfig) -> None:
        self.config = config
        self.options = config.dimer.model_dump()

    @property
    def engine_extensions(self) -> list[type[EngineExtension]]:
        """The `dimer_search` operation's extension."""
        return [DimerSearchExtension]

    def movable_atoms(self, central_atom: int, system: System) -> np.ndarray:
        """Return the indices of the atoms the dimer may move.

        Parameters
        ----------
        central_atom : int
            Index of the atom the search is centred on.
        system : System
            The atomic system.

        Returns
        -------
        np.ndarray
            Sorted atom indices.

        """
        positions = np.asarray(system.positions)
        pbc = np.array(system.pbc, dtype=bool)
        if self.config.control.active_volume:
            _, distances = find_mic(
                positions - positions[central_atom], system.cell, pbc=pbc
            )
            movable = distances <= self.config.activevolume.rmov
        else:
            movable = np.ones(len(positions), dtype=bool)
        if self.config.frozen_atoms is not None:
            frozen = AtomicEnvironment(
                style="region",
                region=self.config.frozen_atoms,
                positions=positions,
                atom_types=system.types,
            ).get_atoms_with_id("in")
            movable[list(frozen)] = False
        return np.flatnonzero(movable)

    def submit(self, requests: list[SearchRequest], manager: Manager) -> list[Future]:
        """Submit one `dimer_search` op per request.

        Parameters
        ----------
        requests : list[SearchRequest]
            The searches to perform, each with a seed and a movable set.
        manager : Manager
            The manager that runs the searches on the engines.

        Returns
        -------
        list[Future]
            One future per request, in the same order, each resolving to the
            search's `Result`. An error raised by the operation, such as a
            LAMMPS error, is raised by the future.

        Raises
        ------
        ValueError
            If the active volume is on and its radius `ract` is not larger than
            the cutoff radius `rcut` (refinement still runs in the active
            volume).

        """
        if self.config.control.active_volume:
            if self.config.activevolume.ract <= self.config.atomicenvironment.rcut:
                raise ValueError(
                    "Active Volume radius is smaller than cutoff radius. Please increase ract or decrease rcut"
                )
        delr_threshold = self.config.eventsearch.delr_thr
        futures = []
        for request in requests:
            future = manager.submit(
                "dimer_search",
                positions=request.positions,
                types=request.types,
                cell=request.cell,
                pbc=request.pbc,
                movable=request.movable,
                central_atom=request.central_atom,
                seed=request.seed,
                options=self.options,
            )
            futures.append(
                _then(
                    future,
                    lambda raw, request=request: dimer_search_result(
                        raw, request, delr_threshold
                    ),
                )
            )
        return futures


def dimer_search_result(
    raw: DimerResult, request: SearchRequest, delr_threshold: float
) -> Result[EventSearchOutput, ErrorInfo]:
    """Turn one dimer search's raw output into an event search result.

    A failed search gives ``EVENT_NOT_FOUND`` with ``(1, reason)`` as details.
    A saddle whose minima are both farther than ``delr_threshold`` from the
    start gives ``EVENT_MINIMA_NOT_MATCH_POSITIONS``. Otherwise the event is
    oriented by the shared rules and its move atom picked by the corrected
    rule, from the oriented initial minimum.

    Parameters
    ----------
    raw : DimerResult
        The `dimer_search` operation's output.
    request : SearchRequest
        The request the search was run for.
    delr_threshold : float
        ``eventsearch.delr_thr``.

    Returns
    -------
    Result[EventSearchOutput, ErrorInfo]
        The event search result.

    """
    error = (0, "") if raw.success else (1, str(raw.reason))

    def read_distances() -> tuple[float, float]:
        return raw.delr1, raw.delr2

    def read_endpoints() -> SearchEndpoints:
        # The shared rules take the closer minimum as the initial state.
        initial = raw.min1_positions if raw.delr1 < raw.delr2 else raw.min2_positions
        return SearchEndpoints(
            e_min1=raw.e_min1,
            e_saddle=raw.e_saddle,
            e_min2=raw.e_min2,
            min1_positions=raw.min1_positions,
            saddle_positions=raw.saddle_positions,
            min2_positions=raw.min2_positions,
            move_atom_index=move_atom(
                initial, raw.saddle_positions, request.cell, request.pbc
            ),
        )

    return search_result(
        central_atom_index=request.central_atom,
        error=error,
        read_distances=read_distances,
        read_endpoints=read_endpoints,
        delr_threshold=delr_threshold,
        types=request.types,
    )


def _then(future: Future, fn: Callable[[Any], Any]) -> Future:
    """Return a future resolving to ``fn`` of ``future``'s result.

    An exception raised by ``future`` or by ``fn`` is raised by the returned
    future.
    """
    mapped: Future = Future()

    def done(source: Future) -> None:
        try:
            mapped.set_result(fn(source.result()))
        except BaseException as exc:
            mapped.set_exception(exc)

    future.add_done_callback(done)
    return mapped
