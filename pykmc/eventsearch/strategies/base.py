"""Base class for event search strategies."""

from __future__ import annotations

from abc import abstractmethod
from concurrent.futures import Future
from typing import TYPE_CHECKING, ClassVar, Optional

import numpy as np

from pykmc._core import Registrable

if TYPE_CHECKING:
    from ...engine import EngineExtension
    from ...manager import Manager
    from ...system import System
    from ..eventsearch import SearchRequest


class EventSearchStrategy(Registrable, root=True):
    """Base class for event search methods.

    A strategy turns search requests into jobs for the engines. It is built
    from its config alone; the manager is given to each call of `submit`.

    Attributes
    ----------
    uses_seed : bool
        Whether each request needs a seed. The facade then draws one per
        request, on rank 0 and in request order, from the run-wide random
        stream that `[Control] seed` seeds.

    """

    uses_seed: ClassVar[bool] = False

    @property
    def engine_extensions(self) -> list[type[EngineExtension]]:
        """Engine extensions the strategy's operations need on every engine.

        `run.py` passes them to the engine factory on every rank.
        """
        return []

    def movable_atoms(self, central_atom: int, system: System) -> Optional[np.ndarray]:
        """Return the indices of the atoms a search may move, or None.

        None, the default, leaves the choice to the backend.

        Parameters
        ----------
        central_atom : int
            Index of the atom the search is centred on.
        system : System
            The atomic system.

        """
        return None

    @abstractmethod
    def submit(self, requests: list[SearchRequest], manager: Manager) -> list[Future]:
        """Submit one event search per request.

        Parameters
        ----------
        requests : list[SearchRequest]
            The searches to perform.
        manager : Manager
            The manager that runs the searches on the engines.

        Returns
        -------
        list[Future]
            One future per request, in the same order. Each resolves to a
            `Result[EventSearchOutput, ErrorInfo]`.

        """
