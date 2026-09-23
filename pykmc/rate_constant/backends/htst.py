"""Harmonic transition state theory (Vineyard) prefactor backend."""

from __future__ import annotations

import math
import numbers

from ..units import hz_to_per_ps
from .base import PrefactorBackend


class HtstBackend(PrefactorBackend):
    """Per-event Vineyard prefactor with a ``k0`` fallback.

    The HTST kernel produces the linear Vineyard frequency ``nu0`` in Hz. This
    backend converts it to ps^-1 exactly once (:func:`hz_to_per_ps`) so the rate
    layer and the KMC clock receive a ps^-1 prefactor; the ``config.k0``
    fallback is already in ps^-1 and is never converted.

    The acceptance window ``nu0_min_THz``/``nu0_max_THz`` is applied by the
    HTST kernel, not here: a value reaching :meth:`resolve_prefactor` is either
    a finite positive frequency or one of the "unavailable" encodings.
    """

    name = "htst"
    requires_event_prefactors = True

    def resolve_prefactor(self, nu0_hz: float | None) -> float:
        """Return ``nu0_hz`` in ps^-1 when usable, else ``config.k0``.

        Parameters
        ----------
        nu0_hz : float or None
            Raw Vineyard frequency in Hz. ``None``, ``NaN`` and ``+-inf`` mean
            "no estimate available" and resolve to ``config.k0``. Any real
            number (including numpy scalars) that is finite and strictly
            positive is converted to ps^-1.

        Returns
        -------
        float
            Resolved prefactor in ps^-1.

        Raises
        ------
        TypeError
            If ``nu0_hz`` is a ``bool``, a ``str`` or any other non-real value:
            those are programming errors, not missing estimates.
        ValueError
            If ``nu0_hz`` is finite but not strictly positive: a zero or
            negative frequency is never a successful HTST estimate.

        """
        if nu0_hz is None:
            return self.config.k0
        if isinstance(nu0_hz, bool) or not isinstance(nu0_hz, numbers.Real):
            raise TypeError(
                "nu0_hz must be a real number in Hz or None, got "
                f"{type(nu0_hz).__name__}: {nu0_hz!r}"
            )
        value = float(nu0_hz)
        if not math.isfinite(value):
            return self.config.k0
        if value <= 0.0:
            raise ValueError(
                f"nu0_hz must be strictly positive (Hz), got {value!r}; encode an "
                "unavailable estimate as None or NaN"
            )
        return hz_to_per_ps(value)
