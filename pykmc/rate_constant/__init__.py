"""Rate constant computation with pluggable prefactor backends.

Public API:

- :func:`compute_rate_Eyring` : compatibility wrapper ``(dE, config) -> rate``
  used by the event tables and the basin solver.
- :func:`create_rate_constant` : build a :class:`RateConstant` from a
  ``RateConstantConfig`` (``config.style`` selects the backend).
- :class:`RateConstant` : facade resolving the prefactor through a backend and
  applying the Arrhenius exponential.
- :class:`RateComponents` : frozen result holding the resolved prefactor and
  the rate, both in ps^-1.
- :func:`rate_from_prefactor` : the Arrhenius arithmetic on a resolved prefactor.
- :class:`PrefactorBackend` : registry root for prefactor backends
  (``constant``, ``htst``, ``rpa``).
- :func:`hz_to_per_ps`, :func:`thz_to_hz` : the frequency unit conversions.

Adding a backend
----------------
Drop a module in ``pykmc/rate_constant/backends/`` defining a subclass of
:class:`PrefactorBackend` whose ``name`` equals the module basename; it is
discovered automatically through :func:`pykmc._core.autodiscover`.
"""

from .backends import PrefactorBackend
from .rate_constant import (
    RateComponents,
    RateConstant,
    compute_rate_Eyring,
    create_rate_constant,
    rate_from_prefactor,
)
from .units import hz_to_per_ps, thz_to_hz

__all__ = [
    "RateConstant",
    "RateComponents",
    "rate_from_prefactor",
    "compute_rate_Eyring",
    "PrefactorBackend",
    "create_rate_constant",
    "hz_to_per_ps",
    "thz_to_hz",
]
