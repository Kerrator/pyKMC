"""Base class for event search strategies."""

from __future__ import annotations

from abc import abstractmethod
from concurrent.futures import Future
from typing import TYPE_CHECKING

from pykmc._core import Registrable

if TYPE_CHECKING:
    from ...manager import Manager
    from ..eventsearch import SearchRequest


class EventSearchStrategy(Registrable, root=True):
    """Base class for event search methods.

    A strategy turns search requests into jobs for the engines. It is built
    from its config alone; the manager is given to each call of `submit`.
    """

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
