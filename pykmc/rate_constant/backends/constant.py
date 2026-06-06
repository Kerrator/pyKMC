from .base import PrefactorBackend
from typing import Protocol


class ConstantBackendConfig(Protocol):
    """Configuration interface for [`ConstantBackend`][pykmc.rate_constant.backends.constant.ConstantBackend].

    Attributes
    ----------
    k0 : float
        Constant prefactor value in ps^-1.
    """
    k0: float


class ConstantBackend(PrefactorBackend):
    """Backend using a constant prefactor.

    Parameters
    ----------
    config : ConstantBackendConfig
        Configuration object exposing a ``k0`` attribute.
        Compatible with the Pydantic ``RateConstantConfig`` used by pykmc.
    """
    name = "constant"

    def __init__(self, config: ConstantBackendConfig) -> None:
        self.config = config

    def compute(self, **kwargs) -> float:
        """Return the constant prefactor.

        Returns
        -------
        float
            Constant prefactor in ps^-1.
        """
        return self.config.k0