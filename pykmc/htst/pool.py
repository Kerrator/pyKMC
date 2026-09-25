"""Fan-out of per-event prefactor jobs over the Manager's local session pool."""

from __future__ import annotations

from concurrent.futures import Future
from typing import Any


class EventPrefactorPool:
    """Batch interface the htst/rpa backends expect from their ``manager``.

    ``Manager.__getattr__`` only yields single-job wrappers, so the one-job-per-
    event fan-out (the oracle's ``Manager.compute_event_prefactors``) lives here,
    on top of the public ``Manager.submit``.

    Parameters
    ----------
    manager : pykmc.manager.Manager
        A started manager whose engines carry
        :class:`pykmc.htst.lammps_extension.HtstLammpsExtension`.
    """

    def __init__(self, manager: Any) -> None:
        self.manager = manager

    def compute_event_prefactors(
        self, config: object, events: list[dict[str, object]]
    ) -> list[Future]:
        """Submit one ``compute_event_prefactors`` job per event; one Future each.

        Each item in ``events`` is a dict with keys ``central_atom_idx``,
        ``min1_positions``, ``saddle_positions``, ``min2_positions``, ``types``,
        ``cell`` (the engine op's keyword arguments).
        """
        return [
            self.manager.submit("compute_event_prefactors", config=config, **event)
            for event in events
        ]
