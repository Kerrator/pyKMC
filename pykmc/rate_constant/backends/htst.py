"""HTST (Vineyard) harmonic prefactor backend."""

import math
from typing import Protocol

from .base import PrefactorBackend


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
    """

    name = "htst"

    def __init__(self, config: HtstBackendConfig) -> None:
        self.config = config

    def compute(self, **kwargs: object) -> float:
        """Return the per-event Vineyard ``nu0`` (Hz) when finite, else ``k0``."""
        nu0 = kwargs.get("nu0")
        if isinstance(nu0, (int, float)) and math.isfinite(nu0):
            return float(nu0)
        return self.config.k0
