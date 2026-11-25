"""Module implementing the EventSearch facade that deals with the event search procedure."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ..result import ErrorInfo, EventSearchOutput, Result
from ..utils.geometry import translate
from .strategies import EventSearchStrategy

if TYPE_CHECKING:
    from ..engine import EngineExtension
    from ..log import LogKMC
    from ..manager import Manager
    from ..system import System


@dataclass(frozen=True, eq=False)
class SearchRequest:
    """Data required to search for events around one central atom.

    Each request holds its own copies of the arrays, taken from the system when
    the search is requested.

    Attributes
    ----------
    central_atom : int
        Index of the atom around which the search is performed.
    positions : np.ndarray
        Positions of the whole system, shape (N, 3).
    cell : np.ndarray
        Simulation cell, shape (3, 3).
    types : np.ndarray
        Types of the atoms of the whole system, shape (N,).
    pbc : np.ndarray
        Periodicity of each cell axis, shape (3,).
    seed : int or None
        Seed of the search, drawn by the facade when the strategy uses one.
    movable : np.ndarray or None
        Indices of the atoms the search may move, when the strategy sets them.

    """

    central_atom: int
    positions: np.ndarray
    cell: np.ndarray
    types: np.ndarray
    pbc: np.ndarray
    seed: Optional[int] = None
    movable: Optional[np.ndarray] = None


class EventSearch:
    """Perform event searches and manage results, using a pluggable strategy.

    The facade is built from its strategy alone. The system, the manager and
    the loggers are given to each call of `execute`.

    Parameters
    ----------
    strategy : EventSearchStrategy
        The event search strategy to use.

    Attributes
    ----------
    system : System or None
        The atomic system of the last call to `execute`.
    results : list[Result[EventSearchOutput, ErrorInfo]] or None
        The results of the last call to `execute`, one per central atom, in the
        same order.

    """

    def __init__(self, strategy: EventSearchStrategy) -> None:
        self._strategy = strategy
        self._cell: Optional[np.ndarray] = None
        self.system: Optional[System] = None
        self.results: Optional[list[Result[EventSearchOutput, ErrorInfo]]] = None

    def execute(
        self,
        central_atom_research_list: list[int],
        system: System,
        manager: Manager,
        loggers: LogKMC,
    ) -> None:
        """Execute an event search for each central atom in the central_atom_research_list list.

        It stores the results of the event searches in self.results

        Parameters
        ----------
        central_atom_research_list : list[int]
            list of central atom around which we will perform the event search.
        system : System
            The atomic system.
        manager : Manager
            The manager that runs the searches on the engines.
        loggers : LogKMC
            The KMC simulation loggers.

        """
        cell = system.cell.copy()
        results = []
        loggers.info(
            "log",
            "\t :=> Searching {} reference events".format(
                len(central_atom_research_list)
            ),
        )
        # Seeds are drawn here, on rank 0 and in request order, so they do not
        # depend on how many workers run the searches or in which order they end.
        requests = [
            SearchRequest(
                central_atom=atom,
                positions=system.positions.copy(),
                cell=cell.copy(),
                types=system.types.copy(),
                pbc=np.array(system.pbc, dtype=bool),
                seed=random.getrandbits(32) if self._strategy.uses_seed else None,
                movable=self._strategy.movable_atoms(atom, system),
            )
            for atom in central_atom_research_list
        ]
        futures = self._strategy.submit(requests, manager)
        for f in futures:
            results.append(f.result())

            loggers.progress_bar(
                "progress", len(results), len(central_atom_research_list)
            )

        self.system = system
        self._cell = cell
        self.results = results

    @property
    def engine_extensions(self) -> list[type[EngineExtension]]:
        """Engine extensions the strategy needs, for the engine factory."""
        return self._strategy.engine_extensions

    @classmethod
    def create(cls, style: str, **kwargs: Any) -> "EventSearch":
        """Build an EventSearch wired with the requested strategy.

        Parameters
        ----------
        style : str
            Name of the registered strategy (e.g. "partn" or "dimer").
        **kwargs
            Forwarded to the strategy's constructor (e.g. ``config=config``).

        """
        return cls(strategy=EventSearchStrategy.create(style, **kwargs))

    def _center_event_positions(
        self, event_search_output: EventSearchOutput
    ) -> EventSearchOutput:
        """Translate positions of the events so that the atom that move the most during the event is at the center of the simulation box.

        It is used to that when we store positions in the reference table around the atom that move the most we don't have periodic bound problems.

        Parameters
        ----------
        event_search_output : EventSearchOutput
            The dataclass countaining the event search outputs.

        Returns
        -------
        EventSearchOutput
            The dataclass countaining the event search outputs with translated positions.

        """
        # Translate atoms so that the atom that moves the most is at the center of the cell at start event, prevent pbc problem with psr
        cell = self._cell
        ax, ay, az = cell[0][0], cell[1][1], cell[2][2]
        # displacement
        move_atom_idx = event_search_output.move_atom_index
        dx, dy, dz = (
            ax / 2 - event_search_output.min1_positions[move_atom_idx][0],
            ay / 2 - event_search_output.min1_positions[move_atom_idx][1],
            az / 2 - event_search_output.min1_positions[move_atom_idx][2],
        )
        displacement = np.array([dx, dy, dz])
        event_search_output.min1_positions = translate(
            event_search_output.min1_positions, displacement, cell
        )
        event_search_output.saddle_positions = translate(
            event_search_output.saddle_positions, displacement, cell
        )
        event_search_output.min2_positions = translate(
            event_search_output.min2_positions, displacement, cell
        )
        event_search_output.cell = cell
        return event_search_output

    def get_successes_results(self) -> list[EventSearchOutput]:
        """Return a list of only successful event searches.

        Returns
        -------
        list[EventSearchOutput]
            List of successful event searches.

        """
        return [
            self._center_event_positions(e.ok_value())
            for e in self.results
            if e.is_ok()
        ]
