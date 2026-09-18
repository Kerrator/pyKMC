"""Root of the prefactor backend registry."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar

from pykmc._core import Registrable

if TYPE_CHECKING:
    from pykmc.config import RateConstantConfig


class PrefactorBackend(Registrable, root=True):
    """Resolve the rate prefactor (ps^-1) for one event.

    Concrete backends live in this package, declare a unique ``name`` matching
    their module basename and are registered through
    :func:`pykmc._core.autodiscover` when :mod:`pykmc.rate_constant.backends`
    is imported. A backend only resolves the prefactor; the Arrhenius arithmetic
    is done by :class:`pykmc.rate_constant.RateConstant`.

    Attributes
    ----------
    name : str
        Registry key, equal to the value accepted by ``RateConstantConfig.style``.
    requires_event_prefactors : bool
        ``True`` when the backend consumes a per-event Vineyard frequency and the
        orchestration layer must therefore compute one for every accepted event;
        ``False`` when the prefactor is a configuration constant.

    Parameters
    ----------
    config : RateConstantConfig
        Rate-constant section of the simulation configuration. ``config.k0``
        (ps^-1) is the constant prefactor and the fallback value.

    """

    name: ClassVar[str]
    requires_event_prefactors: ClassVar[bool]

    def __init__(self, config: RateConstantConfig) -> None:
        self.config = config

    @abstractmethod
    def resolve_prefactor(self, nu0_hz: float | None) -> float:
        """Return the prefactor in ps^-1 for one event.

        Parameters
        ----------
        nu0_hz : float or None
            Raw per-event Vineyard frequency in Hz, or ``None`` when no
            per-event estimate exists. ``None``, ``NaN`` and ``+-inf`` are the
            documented "unavailable" encodings.

        Returns
        -------
        float
            Resolved prefactor in ps^-1.

        """
