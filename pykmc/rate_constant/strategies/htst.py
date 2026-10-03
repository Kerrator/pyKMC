"""HTST rate constant: per-event Vineyard prefactor nu0, k0 for events without one."""

import math
from typing import Optional, Protocol

from ...config import PhysicalConstants
from .base import RateConstantStrategy


class HTSTConfig(Protocol):
    """Configuration read by `HTSTStrategy`."""

    k0: float  # ps^-1, prefactor of events without a nu0
    T: float  # K


class HTSTStrategy(RateConstantStrategy):
    """Rate constant with a per-event prefactor `nu0`, and `k0` for events without one."""

    name = "htst"

    def __init__(self, config: HTSTConfig) -> None:
        self.config = config

    def compute_rate(self, dE: float, nu0: Optional[float] = None) -> float:
        r"""Compute the rate constant from the energy barrier and the event's `nu0`.

        It uses the following equation :
        $$
        \nu_{0}*e^{-\frac{dE}{k_{b}T}}
        $$

        Parameters
        ----------
        dE : float
            The energy barrier (eV).
        nu0 : float, optional
            Attempt frequency of this event (Hz). `None` or a non-finite value
            means the event has none, and `k0` is used instead.

        Returns
        -------
        float
            The rate constant (ps^-1).

        """
        if nu0 is None or not math.isfinite(nu0):
            return self.config.k0 * math.exp(
                -dE / (PhysicalConstants.kb * self.config.T)
            )
        return nu0 * 1e-12 * math.exp(-dE / (PhysicalConstants.kb * self.config.T))
