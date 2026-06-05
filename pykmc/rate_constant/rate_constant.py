import math
from dataclasses import dataclass
from pykmc.config import PhysicalConstants


@dataclass(frozen=True)
class RateComponents:
    prefactor: float
    rate: float


def rate_from_prefactor(prefactor: float, dE: float, T: float) -> float:
    return prefactor * math.exp(-dE / (PhysicalConstants.kb * T))


class RateConstant:

    def __init__(self, T: float, prefactor_backend) -> None:
        self.T = T
        self._prefactor_backend = prefactor_backend

    def compute_prefactor(self, **kwargs) -> float:
        return self._prefactor_backend.compute(**kwargs)

    def compute_rate(self, dE: float, **kwargs) -> RateComponents:
        prefactor = self.compute_prefactor(**kwargs)
        return RateComponents(prefactor=prefactor, rate=rate_from_prefactor(prefactor, dE, self.T))