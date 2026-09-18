"""Constant prefactor backend: the configured ``k0`` for every event."""

from __future__ import annotations

from .base import PrefactorBackend


class ConstantBackend(PrefactorBackend):
    """Prefactor backend returning the configured ``k0`` (ps^-1) for every event.

    Any per-event frequency handed to :meth:`resolve_prefactor` is ignored, so
    the rate is ``k0 * exp(-dE / (kb * T))`` exactly as in the flat
    ``pykmc.rate_constant`` module this package replaces.
    """

    name = "constant"
    requires_event_prefactors = False

    def resolve_prefactor(self, nu0_hz: float | None) -> float:
        """Return ``config.k0`` regardless of ``nu0_hz``.

        Parameters
        ----------
        nu0_hz : float or None
            Ignored.

        Returns
        -------
        float
            ``config.k0`` in ps^-1.

        """
        return self.config.k0
