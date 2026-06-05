from .base import PrefactorBackend
from typing import Protocol


class ConstantBackendConfig(Protocol):
    """Configuration interface for ConstantBackend.

    Parameters
    ----------
    k0 : float
        Value of the constant prefactor in ps^-1
    """
    k0: float


class ConstantBackend(PrefactorBackend):
    name = "constant"

    def __init__(self, config: ConstantBackendConfig) -> None:
        self.config = config

    def compute(self, **kwargs) -> float:
        return self.config.k0