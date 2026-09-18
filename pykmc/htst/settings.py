"""Validated, immutable HTST numerical settings."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Python and NumPy reals are accepted everywhere in the package; bools (Python or
# NumPy) are never numbers here.
_REAL_TYPES = (int, float, np.integer, np.floating)
_BOOL_TYPES = (bool, np.bool_)


def _require_finite_positive(name: str, value: float) -> float:
    """Return ``float(value)`` or raise ``ValueError`` unless it is finite and > 0."""
    if isinstance(value, _BOOL_TYPES) or not isinstance(value, _REAL_TYPES):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class HTSTSettings:
    """Frozen numerical controls for one HTST prefactor calculation.

    Attributes
    ----------
    free_radius : float
        Radius (Å) of the movable sphere around the centre atom; every atom outside
        it is frozen during the partial Hessian.
    fd_step : float
        Finite-difference displacement (Å) for the Hessian.
    zone_radius : float or None
        Optional crop radius (Å) for a scratch system built by the engine adapter;
        ``None`` means the full system. Not used by the pure kernels.
    premin : bool
        Whether the engine adapter pre-relaxes the surroundings with the event core
        frozen before computing Hessians. Not used by the pure kernels.
    nu0_min_hz, nu0_max_hz : float
        Inclusive acceptance window for the Vineyard prefactor in Hz.
    zero_mode_tol : float
        Eigenvalue tolerance in eV / (amu Å²): ``lambda < -tol`` is unstable,
        ``|lambda| <= tol`` is a zero mode, ``lambda > tol`` is stable.

    Notes
    -----
    Numeric fields accept Python and NumPy real scalars (``int``, ``float``,
    ``np.integer``, ``np.floating``) and are stored as Python ``float``; ``bool``
    and ``np.bool_`` are rejected, matching the request and free-region checks.

    """

    free_radius: float = 6.0
    fd_step: float = 0.01
    zone_radius: float | None = None
    premin: bool = False
    nu0_min_hz: float = 1.0e12
    nu0_max_hz: float = 1.0e14
    zero_mode_tol: float = 1.0e-6

    def __post_init__(self) -> None:
        """Validate finiteness, positivity and the window; store reals as float."""
        self._store(
            "free_radius", _require_finite_positive("free_radius", self.free_radius)
        )
        self._store("fd_step", _require_finite_positive("fd_step", self.fd_step))
        if self.zone_radius is not None:
            self._store(
                "zone_radius", _require_finite_positive("zone_radius", self.zone_radius)
            )
        if not isinstance(self.premin, bool):
            raise ValueError(f"premin must be a bool, got {self.premin!r}")
        self._store(
            "nu0_min_hz", _require_finite_positive("nu0_min_hz", self.nu0_min_hz)
        )
        self._store(
            "nu0_max_hz", _require_finite_positive("nu0_max_hz", self.nu0_max_hz)
        )
        if not self.nu0_min_hz < self.nu0_max_hz:
            raise ValueError(
                "nu0 window must satisfy 0 < nu0_min_hz < nu0_max_hz, got "
                f"nu0_min_hz={self.nu0_min_hz!r}, nu0_max_hz={self.nu0_max_hz!r}"
            )
        tol = self.zero_mode_tol
        if isinstance(tol, _BOOL_TYPES) or not isinstance(tol, _REAL_TYPES):
            raise ValueError(f"zero_mode_tol must be a real number, got {tol!r}")
        if not math.isfinite(tol) or tol < 0.0:
            raise ValueError(f"zero_mode_tol must be finite and >= 0, got {tol!r}")
        self._store("zero_mode_tol", float(tol))

    def _store(self, name: str, value: float) -> None:
        """Assign a validated float on this frozen instance."""
        object.__setattr__(self, name, value)


__all__ = ["HTSTSettings"]
