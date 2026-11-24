"""Tolerances and warning thresholds of HTST prefactors.

The physical constants live in `pykmc.constants`.

Units
-----
- Frequency: THz, i.e. ps^-1, `hbar * omega / constants.H` (linear, not angular).
"""

ZERO_MODE_TOL = 1e-6
"""Eigenvalues within this of zero (eV / (amu A^2)) count as zero modes."""

ROTATION_MODE_TOL = 1e-4
"""A rigid rotation u with |H u| (eV / (amu A^2)) below this is a zero mode."""

NU0_WARN_BAND_THZ = (0.1, 1000.0)
"""A nu0 outside this band (THz) is used and logged."""
