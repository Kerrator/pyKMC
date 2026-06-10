import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pykmc.config import PhysicalConstants

if TYPE_CHECKING:
    from concurrent.futures import Future

    from .backends.base import PrefactorBackend


@dataclass(frozen=True)
class RateComponents:
    """Store the result of a rate computation.

    Attributes
    ----------
    prefactor : float
        Rate prefactor in ps^-1.
    rate : float
        Total rate in ps^-1.
    """
    prefactor: float
    rate: float


def rate_from_prefactor(prefactor: float, dE: float, T: float) -> float:
    """Compute rate from a prefactor, an energy barrier and a temperature.

    Parameters
    ----------
    prefactor : float
        Rate prefactor in ps^-1.
    dE : float
        Energy barrier in eV.
    T : float
        Temperature in K.

    Returns
    -------
    float
        Rate in ps^-1.
    """
    return prefactor * math.exp(-dE / (PhysicalConstants.kb * T))


class RateConstant:
    """Facade for rate constant computation using a pluggable prefactor backend.

    Parameters
    ----------
    T : float
        Temperature in K.
    prefactor_backend : PrefactorBackend
        Backend used to compute the prefactor.
    """

    def __init__(self, T: float, prefactor_backend: "PrefactorBackend") -> None:
        self.T = T
        self._prefactor_backend = prefactor_backend

    def compute_prefactor(self, **kwargs: object) -> float:
        """Compute the rate prefactor.

        Returns
        -------
        float
            Rate prefactor in ps^-1.
        """
        return self._prefactor_backend.compute(**kwargs)

    def compute_prefactors_batch(
        self, payloads: "list[dict[str, object]]", config: object
    ) -> "list[Future]":
        """Delegate per-event batch prefactor computation to the backend.

        See ``PrefactorBackend.compute_prefactors_batch`` for the contract
        (one future per payload, each resolving to an ``EventPrefactors``).
        """
        return self._prefactor_backend.compute_prefactors_batch(payloads, config)

    def compute_rate(self, dE: float, **kwargs: object) -> RateComponents:
        """Compute the rate for a given energy barrier.

        Parameters
        ----------
        dE : float
            Energy barrier in eV.

        Returns
        -------
        RateComponents
            Prefactor (ps^-1) and total rate (ps^-1).
        """
        prefactor = self.compute_prefactor(**kwargs)
        return RateComponents(prefactor=prefactor, rate=rate_from_prefactor(prefactor, dE, self.T))
