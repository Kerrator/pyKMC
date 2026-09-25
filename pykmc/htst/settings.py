"""Validated, immutable HTST numerical settings."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

# Python and NumPy reals are accepted everywhere in the package; bools (Python or
# NumPy) are never numbers here.
_REAL_TYPES = (int, float, np.integer, np.floating)
_BOOL_TYPES = (bool, np.bool_)

FREE_REGION_CENTERS: tuple[str, ...] = ("saddle", "min1")
"""Geometries the common free region may be centred on (``free_region_center``)."""

ZONE_RADIUS_AUTO: str = "auto"
"""``zone_radius`` spelling that resolves to ``free_radius + interaction_range``."""

DEFAULT_INTERACTION_RANGE: float = 13.0
"""Interaction range (Å) of settings or a site state built without one.

The same value as the ``interaction_range`` default of ``RateConstantConfig``
(twice the 6.5 Å cutoff of the Ni EAM potential shipped with the tests, the
largest of the shipped potentials). The configuration passes the declared
value; the default only serves direct constructions.
"""


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
    free_region_center : {"saddle", "min1"}
        Geometry in which the one common free region is selected around the
        centre atom. ``"saddle"`` (default) centres it on the mover's saddle
        position, which is symmetric between the two minima by construction;
        ``"min1"`` centres it on the mover's initial position, the original
        model, kept for comparison with older results (on the symmetric SW-Si
        hop it gives a 20 % forward/backward asymmetry that is entirely the
        frozen-boundary choice).
    fd_step : float
        Finite-difference displacement (Å) for the Hessian.
    zone_radius : float, "auto" or None
        Crop radius (Å) for a scratch system built by the engine adapter.
        ``None`` means the full system and is the dataclass default, so
        direct constructions (the pure kernels, tests) keep the full system;
        ``"auto"`` (the ``RateConstantConfig`` default) resolves to
        ``free_radius + interaction_range`` (see ``resolved_zone_radius``);
        a positive real is used as is. Not used by the pure kernels.
    interaction_range : float
        Declared interaction range (Å) of the force model. It sizes an
        ``"auto"`` zone and nothing else here: it never enters the Hessian
        arithmetic, and it is dropped from ``calculation_fields`` once the
        zone is resolved.
    premin : bool
        Whether the engine adapter pre-relaxes the surroundings with the event core
        frozen before computing Hessians. Not used by the pure kernels.
    nu0_min_hz, nu0_max_hz : float
        Inclusive acceptance window for the Vineyard prefactor in Hz.
    zero_mode_tol : float
        Eigenvalue tolerance in eV / (amu Å²): ``lambda < -tol`` is unstable,
        ``|lambda| <= tol`` is a zero mode, ``lambda > tol`` is stable.
    force_tol : float
        Native stationary-geometry tolerance in eV/Å: the largest raw force
        norm on any atom in the common vibrational set must not exceed this.
        Fixed atoms may carry reaction forces. Hessian-only callbacks cannot
        check forces; their callers must establish stationarity separately.

    Notes
    -----
    Numeric fields accept Python and NumPy real scalars (``int``, ``float``,
    ``np.integer``, ``np.floating``) and are stored as Python ``float``; ``bool``
    and ``np.bool_`` are rejected, matching the request and free-region checks.
    ``zone_radius`` additionally accepts the exact string ``"auto"``; the
    case-insensitive INI spellings are normalised by ``RateConstantConfig``.

    """

    free_radius: float = 6.0
    fd_step: float = 0.01
    zone_radius: float | Literal["auto"] | None = None
    premin: bool = False
    nu0_min_hz: float = 1.0e12
    nu0_max_hz: float = 1.0e14
    zero_mode_tol: float = 1.0e-6
    free_region_center: Literal["saddle", "min1"] = "saddle"
    force_tol: float = 0.005
    interaction_range: float = DEFAULT_INTERACTION_RANGE

    def __post_init__(self) -> None:
        """Validate finiteness, positivity and the window; store reals as float."""
        self._store(
            "free_radius", _require_finite_positive("free_radius", self.free_radius)
        )
        if (
            not isinstance(self.free_region_center, str)
            or self.free_region_center not in FREE_REGION_CENTERS
        ):
            raise ValueError(
                f"free_region_center must be one of {FREE_REGION_CENTERS}, got "
                f"{self.free_region_center!r}"
            )
        self._store("fd_step", _require_finite_positive("fd_step", self.fd_step))
        self._store("force_tol", _require_finite_positive("force_tol", self.force_tol))
        if isinstance(self.zone_radius, str):
            if self.zone_radius != ZONE_RADIUS_AUTO:
                raise ValueError(
                    f"zone_radius must be None, {ZONE_RADIUS_AUTO!r} or a real "
                    f"number, got {self.zone_radius!r}"
                )
        elif self.zone_radius is not None:
            self._store(
                "zone_radius", _require_finite_positive("zone_radius", self.zone_radius)
            )
        self._store(
            "interaction_range",
            _require_finite_positive("interaction_range", self.interaction_range),
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

    @property
    def resolved_zone_radius(self) -> float | None:
        """Return the crop radius (Å) these settings select, or ``None``.

        ``None`` stays ``None`` (the full system), ``"auto"`` resolves to
        ``free_radius + interaction_range`` and a number is returned as is.
        This is the one resolution of ``"auto"``: every consumer that needs a
        numeric zone (the engine crop, the recycling dependency radius, the
        context comparison, the physical descriptor) reads it here.
        """
        if self.zone_radius is None:
            return None
        if self.zone_radius == ZONE_RADIUS_AUTO:
            return self.free_radius + self.interaction_range
        return float(self.zone_radius)

    def calculation_fields(self) -> dict[str, Any]:
        """Return the fields that define a calculation, with the zone resolved.

        ``interaction_range`` only sizes an ``"auto"`` zone, so once
        ``zone_radius`` is replaced by :attr:`resolved_zone_radius` it carries
        no further information and is dropped: ``zone_radius="auto"`` with a
        13 Å range and ``zone_radius=19.0`` at ``free_radius=6`` describe the
        same calculation. For settings without ``"auto"`` the result equals
        the field dictionary of the settings before ``interaction_range``
        existed, so persisted identities (snapshot and calculation digests,
        physical descriptors) of tables written earlier are unchanged.
        """
        fields = asdict(self)
        del fields["interaction_range"]
        fields["zone_radius"] = self.resolved_zone_radius
        return fields

    def _store(self, name: str, value: float) -> None:
        """Assign a validated float on this frozen instance."""
        object.__setattr__(self, name, value)


__all__ = [
    "DEFAULT_INTERACTION_RANGE",
    "FREE_REGION_CENTERS",
    "HTSTSettings",
    "ZONE_RADIUS_AUTO",
]
