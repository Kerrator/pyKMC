"""pARTn event search strategy."""

from __future__ import annotations

from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Protocol

from .base import EventSearchStrategy

if TYPE_CHECKING:
    from ...manager import Manager
    from ..eventsearch import SearchRequest


class PartnSearchConfig(Protocol):
    """Configuration required by `PartnStrategy`.

    The whole run configuration is forwarded to the engine's `partn_search`
    op, which also reads its `eventsearch` and `partn` sections.
    """

    control: Any
    activevolume: Any
    atomicenvironment: Any


class PartnStrategy(EventSearchStrategy):
    """Event search with pARTn, run by the engine's `partn_search` op."""

    name = "partn"

    def __init__(self, config: PartnSearchConfig) -> None:
        self.config = config

    def submit(self, requests: list[SearchRequest], manager: Manager) -> list[Future]:
        """Submit one `partn_search` op per request.

        With the active volume on, the op also receives the cell.

        Parameters
        ----------
        requests : list[SearchRequest]
            The searches to perform.
        manager : Manager
            The manager that runs the searches on the engines.

        Returns
        -------
        list[Future]
            One future per request, in the same order.

        Raises
        ------
        ValueError
            If the active volume is on and its radius `ract` is not larger than
            the cutoff radius `rcut`.

        """
        if self.config.control.active_volume:
            if self.config.activevolume.ract <= self.config.atomicenvironment.rcut:
                raise ValueError(
                    "Active Volume radius is smaller than cutoff radius. Please increase ract or decrease rcut"
                )
            return [
                manager.submit(
                    "partn_search",
                    config=self.config,
                    central_atom_idx=request.central_atom,
                    positions=request.positions,
                    cell=request.cell,
                    types=request.types,
                )
                for request in requests
            ]
        return [
            manager.submit(
                "partn_search",
                config=self.config,
                central_atom_idx=request.central_atom,
                positions=request.positions,
                types=request.types,
            )
            for request in requests
        ]
