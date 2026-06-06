"""Eigenvalue → vibrational-frequency conversion helpers for HTST.

PROVENANCE: vendored from
apps/PyKMC_Analysis/Analysis/htst/kappa_rpa.py (the proven analysis-side
implementation). Physical constants live in
:class:`pykmc.config.PhysicalConstants` (``hbar_omega_eV``, ``hbar_eV_s``,
``eskm_div_eV_amu_A2``); this module keeps the HTST-specific solver tolerance and
the unit-conversion functions.
"""

import math

from pykmc.config import PhysicalConstants

# Algorithmic tolerance (NOT a physical constant): |λ| below this is treated as a
# projected-out zero mode by ``normal_modes_from_hessian``.
ZERO_MODE_TOL_EV2 = 1.0e-6


def eigval_to_omega_eV(lmbda: float) -> float:
    """Return ℏω in eV (angular) from a positive mass-weighted-Hessian eigenvalue.

    Parameters
    ----------
    lmbda : float
        Eigenvalue of the mass-weighted Hessian, in eV/(amu·Å²). Must be >= 0.

    Returns
    -------
    float
        ℏω in eV.

    """
    return PhysicalConstants.hbar_omega_eV * math.sqrt(lmbda)


def omega_eV_to_hz(omega_eV: float) -> float:
    """Convert ℏω [eV] (angular) to a LINEAR frequency ν [Hz] via ν = ℏω / (2π·ℏ).

    Equivalent to dividing by h = 2π·ℏ = 4.135667e-15 eV·s. This matches the
    conversion inside the vendored ``vineyard_prefactor`` (÷ 2π·HBAR_EV_S), so ν₀
    comes out in Hz and the KMC rate k = ν₀·exp(-Ea/kT) is in s⁻¹.

    Parameters
    ----------
    omega_eV : float
        ℏω in eV.

    Returns
    -------
    float
        Linear frequency ν in Hz.

    """
    return omega_eV / (2.0 * math.pi * PhysicalConstants.hbar_eV_s)


def hz_to_thz(f_hz: float) -> float:
    """Convert a frequency from Hz to THz."""
    return f_hz * 1.0e-12


def thz_to_hz(f_thz: float) -> float:
    """Convert a frequency from THz to Hz."""
    return f_thz * 1.0e12
