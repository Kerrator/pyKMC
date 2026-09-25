"""Physical constants and eigenvalue → vibrational-frequency helpers for HTST.

PROVENANCE: vendored from
apps/PyKMC_Analysis/Analysis/htst/kappa_rpa.py (the proven analysis-side
implementation). The HTST-specific physical constants live here, next to the
code that uses them, so the plugin does not touch ``pykmc.config``.
"""

import math

# ℏω [eV] from a mass-weighted-Hessian eigenvalue [eV/(amu·Å²)]:
# ℏω_eV = HBAR_OMEGA_EV * sqrt(eigenvalue).
HBAR_OMEGA_EV = 0.06466  # eV / sqrt(eV/(amu·Å²))
HBAR_EV_S = 6.582119569e-16  # ℏ in eV·s
# LAMMPS ``dynamical_matrix ... eskm`` output scaling -> eV/(amu·Å²).
ESKM_DIV_EV_AMU_A2 = 9648.5

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
    return HBAR_OMEGA_EV * math.sqrt(lmbda)


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
    return omega_eV / (2.0 * math.pi * HBAR_EV_S)


def hz_to_thz(f_hz: float) -> float:
    """Convert a frequency from Hz to THz."""
    return f_hz * 1.0e-12


def thz_to_hz(f_thz: float) -> float:
    """Convert a frequency from THz to Hz."""
    return f_thz * 1.0e12


def hz_to_per_ps(f_hz: float) -> float:
    """Convert a linear frequency from Hz (s⁻¹) to ps⁻¹.

    The rate layer and the KMC clock work in ps⁻¹ (``delta_t`` comes out in ps
    and is later scaled to seconds), so a ν₀ produced in Hz must be converted
    before it is used as a rate prefactor. Numerically ps⁻¹ ≡ THz, hence the
    same 1e-12 factor as :func:`hz_to_thz`.
    """
    return f_hz * 1.0e-12
