"""Batching service for per-event HTST prefactor requests.

One :class:`PrefactorService` is built by ``KMC`` for the htst/rpa styles and
shared by the reference and active event tables. It is the only place that:

- converts the configured ``nu0_min_THz``/``nu0_max_THz`` window to Hz
  (:func:`pykmc.rate_constant.thz_to_hz`, exactly once, into a frozen
  :class:`~pykmc.htst.settings.HTSTSettings`);
- builds :class:`~pykmc.htst.request.HTSTEventRequest` objects from the full
  system geometry, taking the potential species order and masses from the one
  species rule (:func:`pykmc.engine.lammps.species_map`, imported lazily so the
  constant path never imports LAMMPS);
- fans the requests out through ``Manager.submit("compute_event_prefactors",
  request=...)`` (one job per accepted event, both directions per job), keeps
  the request-to-Future association keyed by ``event_key`` and resolves every
  Future before any consumer reads a rate, returning the results mapped by
  ``event_key`` regardless of completion order with strict cardinality.

Exceptions raised by a worker operation propagate through the Future: a
transport or programming failure is never turned into a ``k0`` fallback. Only
the scientific rejections encoded in ``EventPrefactors`` fall back, and that
policy lives in the event tables, not here.

Nothing in the constant path imports this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from pykmc.htst.request import HTSTEventRequest
from pykmc.htst.result import EventPrefactors
from pykmc.htst.settings import HTSTSettings

from .rate_constant import RateConstant
from .units import thz_to_hz

if TYPE_CHECKING:
    from pykmc.config import Config, RateConstantConfig

PREFACTOR_OPERATION: str = "compute_event_prefactors"
"""Worker operation exposed by ``LammpsHTSTExtension`` (contracts section 6)."""


def settings_from_config(rate_config: RateConstantConfig) -> HTSTSettings:
    """Build the frozen kernel settings from the rate-constant configuration.

    The THz window is converted to Hz here and nowhere else.

    Parameters
    ----------
    rate_config : RateConstantConfig
        Rate-constant section of the simulation configuration.

    Returns
    -------
    HTSTSettings
        Validated settings with ``nu0_min_hz``/``nu0_max_hz`` in Hz.

    """
    return HTSTSettings(
        free_radius=rate_config.free_radius,
        fd_step=rate_config.fd_step,
        zone_radius=rate_config.zone_radius,
        premin=rate_config.premin,
        nu0_min_hz=thz_to_hz(rate_config.nu0_min_THz),
        nu0_max_hz=thz_to_hz(rate_config.nu0_max_THz),
    )


class PrefactorService:
    """Build HTST requests and resolve them through the manager, in batches.

    Parameters
    ----------
    config : Config
        Full simulation configuration (``config.rateconstant`` is used).
    manager : Manager
        The engine manager; only ``submit`` is used.
    rate_constant : RateConstant
        The rate facade of the run; its backend must require per-event
        prefactors (htst/rpa), otherwise building the service is an error
        because the constant path must never construct one.

    Attributes
    ----------
    settings : HTSTSettings
        Kernel settings shared by every request of the run.
    n_submitted : int
        Total number of requests submitted so far (diagnostics).

    """

    def __init__(
        self, config: Config, manager: Any, rate_constant: RateConstant
    ) -> None:
        if not rate_constant.backend.requires_event_prefactors:
            raise ValueError(
                "PrefactorService is only meaningful for a backend that requires "
                f"per-event prefactors; got style {config.rateconstant.style!r}"
            )
        if manager is None:
            raise RuntimeError(
                "PrefactorService needs the engine manager; it is None (the "
                "manager must be set before KMC initialisation)"
            )
        self.config = config
        self.manager = manager
        self.rate_constant = rate_constant
        self.settings = settings_from_config(config.rateconstant)
        self.n_submitted = 0

    def build_request(
        self,
        *,
        event_key: tuple,
        min1_positions: np.ndarray,
        saddle_positions: np.ndarray,
        min2_positions: np.ndarray,
        types: Sequence[str],
        cell: np.ndarray,
        pbc: Sequence[bool],
        center_index: int,
    ) -> HTSTEventRequest:
        """Build and validate one request from full-system geometry.

        Parameters
        ----------
        event_key : tuple
            Hashable logical identity echoed back by the worker.
        min1_positions, saddle_positions, min2_positions : np.ndarray
            Full-system ``(N, 3)`` positions of the three stationary points,
            in one common atom order. They are copied.
        types : Sequence[str]
            Chemical symbol of every atom of the full system, same order.
        cell : np.ndarray
            ``(3, 3)`` simulation cell.
        pbc : Sequence[bool]
            Actual periodicity per axis of the system.
        center_index : int
            Global index of the moving atom.

        Returns
        -------
        HTSTEventRequest
            The validated request.

        """
        from pykmc.engine.lammps import species_map  # lazy: LAMMPS-bound module

        symbols = tuple(str(t) for t in types)
        species, masses = species_map(list(symbols))
        request = HTSTEventRequest(
            event_key=event_key,
            min1_positions=np.array(min1_positions, dtype=float, copy=True),
            saddle_positions=np.array(saddle_positions, dtype=float, copy=True),
            min2_positions=np.array(min2_positions, dtype=float, copy=True),
            types=symbols,
            species=species,
            masses=masses,
            cell=np.array(cell, dtype=float, copy=True),
            pbc=tuple(bool(p) for p in pbc),
            center_index=int(center_index),
            settings=self.settings,
        )
        request.validate()
        return request

    def compute(
        self, requests: Sequence[HTSTEventRequest]
    ) -> dict[tuple, EventPrefactors]:
        """Submit every request, wait for all of them and map results by key.

        Parameters
        ----------
        requests : Sequence[HTSTEventRequest]
            Requests with pairwise distinct ``event_key`` values.

        Returns
        -------
        dict[tuple, EventPrefactors]
            One result per request keyed by its ``event_key``; the completion
            order of the workers is irrelevant because each Future is held
            against the key it was submitted for.

        Raises
        ------
        ValueError
            If two requests share an ``event_key``.
        RuntimeError
            If a worker returns something that is not an ``EventPrefactors``
            for the submitted key (a transport or contract failure).
        Exception
            Whatever a worker operation raised, re-raised from its Future.

        """
        keys = [req.event_key for req in requests]
        if len(set(keys)) != len(keys):
            raise ValueError(f"event_key values must be distinct, got {keys}")
        pending: list[tuple[tuple, Any]] = []
        for req in requests:
            future = self.manager.submit(PREFACTOR_OPERATION, request=req)
            pending.append((req.event_key, future))
        self.n_submitted += len(pending)

        results: dict[tuple, EventPrefactors] = {}
        for key, future in pending:
            value = future.result()  # a worker exception propagates here
            if not isinstance(value, EventPrefactors):
                raise RuntimeError(
                    f"{PREFACTOR_OPERATION} returned {type(value).__name__} for "
                    f"event {key!r}; expected EventPrefactors"
                )
            if value.event_key != key:
                raise RuntimeError(
                    f"{PREFACTOR_OPERATION} echoed event_key {value.event_key!r} "
                    f"for the request submitted as {key!r}"
                )
            results[key] = value
        if len(results) != len(requests):
            raise RuntimeError(
                f"resolved {len(results)} results for {len(requests)} requests"
            )
        return results


__all__ = ["PREFACTOR_OPERATION", "PrefactorService", "settings_from_config"]
