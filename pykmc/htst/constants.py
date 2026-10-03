"""Physical constants, unit conversions and thresholds for HTST prefactors.

Units
-----
- Mass-weighted Hessian and its eigenvalues `lambda`: eV / (amu A^2).
- `hbar * omega` of a mode: eV, `HBAR_OMEGA_EV * sqrt(lambda)`.
- Frequency: Hz, `hbar * omega / H_EV_S` (linear, not angular).
"""

import math

# CODATA 2018 values, used only to derive HBAR_OMEGA_EV.
EV_J = 1.602176634e-19  # J
AMU_KG = 1.66053906660e-27  # kg
HBAR_EV_S = 6.582119569e-16  # eV.s

HBAR_OMEGA_EV = HBAR_EV_S * math.sqrt(EV_J / (AMU_KG * 1e-20))
"""`hbar * omega` in eV per `sqrt(eV / (amu A^2))`, about 0.0646541513."""

H_EV_S = 4.135667696e-15
"""Planck constant (eV.s)."""

ESKM_METAL_CONVERSION = 9648.5
"""Factor LAMMPS applies in `dynamical_matrix eskm` with metal units.

Dividing an eskm matrix by it gives eV / (amu A^2).
"""

ZERO_MODE_TOL = 1e-6
"""Eigenvalues within this of zero (eV / (amu A^2)) count as zero modes."""

ROTATION_MODE_TOL = 1e-4
"""A rigid rotation whose curvature (eV / (amu A^2)) is below this is a zero mode."""

NU0_WARN_BAND_HZ = (1.0e11, 1.0e15)
"""A nu0 outside this band (0.1-1000 THz) is used and logged."""

RMOV_WARN_A = 8.0
"""Active-volume `rmov` (A) below which nu0 is not converged."""

AV_BUFFER_WARN_A = 4.0
"""Active-volume buffer `ract - rmov` (A) below which the crop error grows."""
