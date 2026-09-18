"""Rate constant facade: prefactor resolution plus Arrhenius arithmetic."""

from __future__ import annotations

import math as m
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pykmc.config import PhysicalConstants

from .backends import PrefactorBackend

if TYPE_CHECKING:
    from pykmc.config import Config, RateConstantConfig


@dataclass(frozen=True)
class RateComponents:
    """Resolved components of one rate computation.

    Attributes
    ----------
    prefactor : float
        Resolved rate prefactor in ps^-1.
    rate : float
        Rate ``prefactor * exp(-dE / (kb * T))`` in ps^-1.

    """

    prefactor: float
    rate: float


def rate_from_prefactor(prefactor: float, dE: float, T: float) -> float:
    r"""Compute a rate from a resolved prefactor, an energy barrier and a temperature.

    $$
    k = k_{0} e^{-\frac{\Delta E}{k_{b}T}}
    $$

    Parameters
    ----------
    prefactor : float
        Resolved prefactor in ps^-1.
    dE : float
        Energy barrier in eV.
    T : float
        Temperature in K.

    Returns
    -------
    float
        Rate in ps^-1.

    """
    p = PhysicalConstants()
    return prefactor * m.exp(-dE / (p.kb * T))


class RateConstant:
    """Compute rates by delegating prefactor resolution to a backend.

    Parameters
    ----------
    T : float
        Temperature in K.
    backend : PrefactorBackend
        Backend resolving the prefactor in ps^-1 for each event.

    """

    def __init__(self, T: float, backend: PrefactorBackend) -> None:
        self.T = T
        self.backend = backend

    def compute_rate(self, dE: float, nu0_hz: float | None = None) -> RateComponents:
        """Resolve the prefactor and compute the rate for one event.

        Parameters
        ----------
        dE : float
            Energy barrier in eV.
        nu0_hz : float or None, optional
            Raw per-event Vineyard frequency in Hz; ``None`` when no estimate
            exists. Ignored by the ``constant`` backend.

        Returns
        -------
        RateComponents
            Resolved prefactor (ps^-1) and rate (ps^-1).

        """
        prefactor = self.backend.resolve_prefactor(nu0_hz)
        return RateComponents(
            prefactor=prefactor, rate=rate_from_prefactor(prefactor, dE, self.T)
        )


def create_rate_constant(config: RateConstantConfig) -> RateConstant:
    """Build a :class:`RateConstant` for the backend named by ``config.style``.

    Parameters
    ----------
    config : RateConstantConfig
        Rate-constant section of the simulation configuration.

    Returns
    -------
    RateConstant
        Facade using ``PrefactorBackend.create(config.style, config=config)``.

    Raises
    ------
    ValueError
        If ``config.style`` names no registered backend.
    ImportError
        If the backend module failed to import when the registry was built.

    """
    backend = PrefactorBackend.create(config.style, config=config)
    return RateConstant(T=config.T, backend=backend)


def compute_rate_Eyring(dE: float, config: Config) -> float:
    r"""Compute the rate constant from an energy barrier and the configuration.

    Compatibility wrapper kept for the existing call sites; it equals
    ``create_rate_constant(config.rateconstant).compute_rate(dE).rate``. For
    ``style = constant`` the arithmetic is the same
    $k_{0} e^{-\Delta E / (k_{b}T)}$ evaluated in the same order as before, so
    results are bit-identical to the flat module this package replaces.

    Parameters
    ----------
    dE : float
        The energy barrier in eV.
    config : Config
        The configuration of the simulation.

    Returns
    -------
    float
        The rate constant in ps^-1.

    """
    return create_rate_constant(config.rateconstant).compute_rate(dE).rate
