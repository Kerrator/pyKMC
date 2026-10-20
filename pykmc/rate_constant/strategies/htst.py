"""HTST rate constant: per-event Vineyard prefactor nu0, k0 for events without one."""

import math
from typing import Optional

from .base import RateConstantStrategy


class HTSTStrategy(RateConstantStrategy):
    r"""Rate constant with a per-event prefactor `nu0`, and `k0` for events without one.

    $$
    k = \nu_{0} \, e^{-\frac{dE}{k_{b}T}}
    $$
    """

    name = "htst"

    def prefactor(self, nu0: Optional[float] = None) -> float:
        """Return the event's `nu0`, or `[RateConstant] k0` when it has none.

        Parameters
        ----------
        nu0 : float, optional
            Attempt frequency of this event (THz, i.e. ps^-1). `None` or a
            non-finite value means the event has none, and `k0` is used instead.

        Returns
        -------
        float
            `nu0` or `k0` (ps^-1).

        """
        if nu0 is None or not math.isfinite(nu0):
            return self.config.k0
        return nu0
