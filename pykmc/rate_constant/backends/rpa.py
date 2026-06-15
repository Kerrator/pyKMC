"""RPA recrossing-corrected prefactor backend (bare-Vineyard until kappa)."""

import math
from typing import TYPE_CHECKING, Optional, Protocol

from pykmc.htst.constants import hz_to_per_ps

from .base import PrefactorBackend

if TYPE_CHECKING:
    from concurrent.futures import Future


class RpaBackendConfig(Protocol):
    """Configuration interface for RpaBackend.

    Parameters
    ----------
    k0 : float
        Per-event fallback prefactor (ps⁻¹), used when ``nu0`` is unavailable.

    """

    k0: float


class RpaBackend(PrefactorBackend):
    """RPA recrossing-corrected prefactor (Sharia & Henkelman 2016).

    Subclasses :class:`PrefactorBackend` DIRECTLY (not :class:`HtstBackend`) so
    the factory's non-transitive ``__subclasses__()`` registry discovers it.

    The recrossing factor ``kappa`` is deferred, so this is currently
    bare-Vineyard and numerically identical to ``htst``: it returns the per-event
    ``nu0`` converted to ps⁻¹ when finite, else the ``k0`` fallback. Multiply by
    ``kappa`` here once it is available.
    """

    name = "rpa"

    def __init__(self, config: RpaBackendConfig, manager: Optional[object] = None) -> None:
        self.config = config
        self.manager = manager

    def compute(self, **kwargs: object) -> float:
        """Return the per-event Vineyard prefactor in ps⁻¹ when finite, else ``k0``.

        ``nu0`` is supplied in Hz and converted to ps⁻¹; the ``k0`` fallback is
        already in ps⁻¹.
        """
        nu0 = kwargs.get("nu0")
        if isinstance(nu0, (int, float)) and math.isfinite(nu0):
            return hz_to_per_ps(float(nu0))  # * kappa <- future Sharia & Henkelman
        return self.config.k0

    def compute_prefactors_batch(
        self, payloads: "list[dict[str, object]]", config: object
    ) -> "list[Future]":
        """Fan out one per-event Vineyard nu0 job per payload via the manager pool.

        Mirrors :meth:`HtstBackend.compute_prefactors_batch` (duplicated rather
        than shared via subclassing — the factory registry is non-transitive).
        """
        if self.manager is None:
            raise RuntimeError(
                "RpaBackend.compute_prefactors_batch requires an engine manager; "
                "inject one via create_rate_constant(..., manager=...)."
            )
        return self.manager.compute_event_prefactors(config, payloads)
