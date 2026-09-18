"""Frequency unit conversions used at the rate-constant boundary.

The HTST kernel emits the linear Vineyard frequency ``nu0`` in Hz; the rate
layer, the stored rates and the KMC clock all work in ps^-1. The conversion is
applied exactly once, by the prefactor backend, through :func:`hz_to_per_ps`.
The ``nu0_min_THz``/``nu0_max_THz`` acceptance window is entered in THz and is
converted to Hz once, at the orchestration boundary, through :func:`thz_to_hz`.
"""

from __future__ import annotations

HZ_PER_THZ: float = 1.0e12
"""Number of Hz in one THz (1 THz == 1 ps^-1)."""


def hz_to_per_ps(f_hz: float) -> float:
    """Convert a linear frequency from Hz (s^-1) to ps^-1.

    Parameters
    ----------
    f_hz : float
        Linear frequency in Hz.

    Returns
    -------
    float
        The same frequency in ps^-1 (numerically equal to THz).

    """
    return f_hz * 1.0e-12


def thz_to_hz(f_thz: float) -> float:
    """Convert a linear frequency from THz to Hz.

    Parameters
    ----------
    f_thz : float
        Linear frequency in THz.

    Returns
    -------
    float
        The same frequency in Hz.

    """
    return f_thz * HZ_PER_THZ
