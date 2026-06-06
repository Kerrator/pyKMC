"""Rate constant computation with pluggable prefactor backends.

The public API of this package is:

- [`create_rate_constant`][pykmc.rate_constant.create_rate_constant] :
  factory function to instantiate a [`RateConstant`][pykmc.rate_constant.RateConstant]
  from a backend name and temperature.
- [`RateConstant`][pykmc.rate_constant.RateConstant] :
  facade that delegates prefactor computation to a backend and use it to compute the rate.
- [`RateComponents`][pykmc.rate_constant.RateComponents] :
  result dataclass holding both the prefactor and the rate.
- [`rate_from_prefactor`][pykmc.rate_constant.rate_from_prefactor] :
  standalone function computing a rate from a known prefactor.

Adding a backend
----------------
Drop a new file in `pykmc/rate_constant/backends/`, define a class that inherits
from [`PrefactorBackend`][pykmc.rate_constant.backends.PrefactorBackend] and set
a unique ``name`` class attribute. The backend is discovered automatically.
"""

from .factory import create_rate_constant
from .rate_constant import RateConstant, RateComponents, rate_from_prefactor

__all__ = ["create_rate_constant", "RateConstant", "RateComponents", "rate_from_prefactor"]
