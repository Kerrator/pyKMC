"""RPA recrossing-corrected prefactor backend (bare-Vineyard until kappa)."""

import math
from typing import Protocol

from .base import PrefactorBackend


class RpaBackendConfig(Protocol):
    """Configuration interface for RpaBackend.

    Parameters
    ----------
    k0 : float
        Per-event fallback prefactor (Hz), used when ``nu0`` is unavailable.

    """

    k0: float


class RpaBackend(PrefactorBackend):
    """RPA recrossing-corrected prefactor (Sharia & Henkelman 2016).

    Subclasses :class:`PrefactorBackend` DIRECTLY (not :class:`HtstBackend`) so
    the factory's non-transitive ``__subclasses__()`` registry discovers it.

    The recrossing factor ``kappa`` is deferred, so this is currently
    bare-Vineyard and numerically identical to ``htst``: it returns the per-event
    ``nu0`` (Hz) when finite, else the ``k0`` fallback. Multiply by ``kappa`` here
    once it is available.
    """

    name = "rpa"

    def __init__(self, config: RpaBackendConfig) -> None:
        self.config = config

    def compute(self, **kwargs: object) -> float:
        """Return the per-event Vineyard ``nu0`` (Hz) when finite, else ``k0``."""
        nu0 = kwargs.get("nu0")
        if isinstance(nu0, (int, float)) and math.isfinite(nu0):
            return float(nu0)  # * kappa  <- future Sharia & Henkelman correction
        return self.config.k0
