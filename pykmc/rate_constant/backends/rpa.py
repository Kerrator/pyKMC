"""RPA prefactor backend: currently bare Vineyard, no recrossing correction."""

from __future__ import annotations

from .htst import HtstBackend


class RpaBackend(HtstBackend):
    """Bare Vineyard prefactor registered under the name ``rpa``.

    No recrossing correction is implemented: this backend is numerically
    identical to :class:`~pykmc.rate_constant.backends.htst.HtstBackend`
    (the recrossing factor kappa is 1). It exists so an input file can already
    request ``style = rpa`` and so a future dynamical correction has a single
    place to multiply the resolved prefactor by kappa.
    """

    name = "rpa"
