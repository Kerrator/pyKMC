"""HTST (Vineyard) harmonic prefactor backend."""

import math
from typing import TYPE_CHECKING, Optional, Protocol

from .base import PrefactorBackend

if TYPE_CHECKING:
    from concurrent.futures import Future


class HtstBackendConfig(Protocol):
    """Configuration interface for HtstBackend.

    Parameters
    ----------
    k0 : float
        Per-event fallback prefactor (Hz), used when the Vineyard ``nu0`` is
        unavailable. Must be in the SAME units as ``nu0`` (Hz).

    """

    k0: float


class HtstBackend(PrefactorBackend):
    """Harmonic TST prefactor: the per-event Vineyard ``nu0``, with a ``k0`` fallback.

    ``nu0`` is computed once per reference event (LAMMPS ``dynamical_matrix`` ->
    Vineyard) and arrives through the rate-constant kwargs channel:
    ``RateConstant.compute_rate(dE, nu0=...) -> compute_prefactor(nu0=...) ->
    HtstBackend.compute(nu0=...)``. ``dE`` and ``T`` are handled by
    ``RateConstant``, not here.

    Both ``nu0`` and the ``k0`` fallback are linear frequencies in Hz.

    Batch computation: :meth:`compute_prefactors_batch` fans one nu0 job per
    payload over the injected engine ``manager``. The API is direction-agnostic
    and stateless — a future active-level rate refine (probable events at the
    per-site refined saddle) reuses it unchanged by submitting refined-saddle
    payloads and patching the active table; no signature change is needed.
    """

    name = "htst"

    def __init__(self, config: HtstBackendConfig, manager: Optional[object] = None) -> None:
        self.config = config
        self.manager = manager

    def compute(self, **kwargs: object) -> float:
        """Return the per-event Vineyard ``nu0`` (Hz) when finite, else ``k0``."""
        nu0 = kwargs.get("nu0")
        if isinstance(nu0, (int, float)) and math.isfinite(nu0):
            return float(nu0)
        return self.config.k0

    def compute_prefactors_batch(
        self, payloads: "list[dict[str, object]]", config: object
    ) -> "list[Future]":
        """Fan out one per-event Vineyard nu0 job per payload via the manager pool.

        ``config`` must be the FULL pykmc ``Config`` (the engine op reads
        ``config.rateconstant``), distinct from this backend's own sub-config.
        """
        if self.manager is None:
            raise RuntimeError(
                "HtstBackend.compute_prefactors_batch requires an engine manager; "
                "inject one via create_rate_constant(..., manager=...)."
            )
        return self.manager.compute_event_prefactors(config, payloads)
