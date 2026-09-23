"""Physical constants and unit conversions for the HTST numerical layer.

Every constant is a CODATA 2018 primary. Derived factors are computed from the
primaries at import time rather than typed as rounded literals, so a reviewer can
trace each unit through the arithmetic below.

Unit conventions (binding for ``pykmc.htst``)
---------------------------------------------
- Mass-weighted Hessian ``H_mw`` : eV / (amu Å²)
- Eigenvalue ``lambda`` of ``H_mw``: eV / (amu Å²); ``lambda = omega**2`` in those units
- ``hbar * omega``                : eV, ``HBAR_OMEGA_EV * sqrt(lambda)``
- Linear frequency ``nu``         : Hz, ``hbar * omega / h`` (not angular; no 2 pi)

This layer emits Hz only. The Hz -> ps⁻¹ conversion belongs to the rate backend.
"""

from __future__ import annotations

import math

# --- CODATA 2018 primaries -------------------------------------------------
EV_J: float = 1.602176634e-19
"""Electron-volt in joules (exact since the 2019 SI redefinition)."""

AMU_KG: float = 1.66053906660e-27
"""Unified atomic mass unit in kilograms."""

ANGSTROM_M: float = 1.0e-10
"""Ångström in metres (exact)."""

HBAR_J_S: float = 1.054571817e-34
"""Reduced Planck constant in J s."""

HBAR_EV_S: float = 6.582119569e-16
"""Reduced Planck constant in eV s."""

H_EV_S: float = 4.135667696e-15
"""Planck constant in eV s.

CODATA rounds ``h`` and ``hbar`` independently, so ``H_EV_S == 2 * pi * HBAR_EV_S``
holds only to about 1.5e-10 relative. Unit oracles compare at ``rtol = 1e-9``.
"""

# --- Derived factors -------------------------------------------------------
OMEGA_SI_PER_SQRT_EIGVAL: float = math.sqrt(EV_J / (AMU_KG * ANGSTROM_M**2))
"""Angular frequency in rad/s per ``sqrt(eV / (amu Å²))``."""

HBAR_OMEGA_EV: float = HBAR_EV_S * OMEGA_SI_PER_SQRT_EIGVAL
"""``hbar * omega`` in eV per ``sqrt(eV / (amu Å²))``; approximately 0.06465."""

ESKM_METAL_CONVERSION: float = 9648.5
"""LAMMPS's own ``conv_energy`` literal for ``metal`` units in ``dynamical_matrix eskm``.

LAMMPS writes ``-dF / (2 dx sqrt(m_i m_j)) * 9648.5`` (eV -> 10 J/mol). Dividing an
eskm matrix by this literal, not by the exact ``EV_J * N_A / 10``, recovers
eV / (amu Å²) exactly as LAMMPS produced it.
"""


def eigval_to_hbar_omega_ev(lmbda: float) -> float:
    """Return ``hbar * omega`` in eV for a non-negative mass-weighted eigenvalue.

    Parameters
    ----------
    lmbda : float
        Eigenvalue of the mass-weighted Hessian in eV / (amu Å²). Must be >= 0.

    Returns
    -------
    float
        ``hbar * omega`` in eV.

    Raises
    ------
    ValueError
        If ``lmbda`` is negative or not finite.

    """
    if not math.isfinite(lmbda) or lmbda < 0.0:
        raise ValueError(f"eigenvalue must be finite and >= 0, got {lmbda!r}")
    return HBAR_OMEGA_EV * math.sqrt(lmbda)


def hbar_omega_ev_to_hz(hbar_omega_ev: float) -> float:
    """Convert ``hbar * omega`` in eV to a linear frequency in Hz via ``E / h``.

    Parameters
    ----------
    hbar_omega_ev : float
        Mode energy ``hbar * omega`` in eV.

    Returns
    -------
    float
        Linear frequency ``nu = omega / (2 pi)`` in Hz.

    """
    return hbar_omega_ev / H_EV_S


def eigval_to_hz(lmbda: float) -> float:
    """Return the linear frequency in Hz of a non-negative mass-weighted eigenvalue.

    Parameters
    ----------
    lmbda : float
        Eigenvalue of the mass-weighted Hessian in eV / (amu Å²). Must be >= 0.

    Returns
    -------
    float
        Linear frequency in Hz.

    """
    return hbar_omega_ev_to_hz(eigval_to_hbar_omega_ev(lmbda))


__all__ = [
    "AMU_KG",
    "ANGSTROM_M",
    "ESKM_METAL_CONVERSION",
    "EV_J",
    "HBAR_EV_S",
    "HBAR_J_S",
    "HBAR_OMEGA_EV",
    "H_EV_S",
    "OMEGA_SI_PER_SQRT_EIGVAL",
    "eigval_to_hbar_omega_ev",
    "eigval_to_hz",
    "hbar_omega_ev_to_hz",
]
