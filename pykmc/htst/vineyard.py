"""Vineyard harmonic prefactor from minimum and saddle mode spectra.

``nu0 = prod(nu_i^min) / prod(nu_i^sad)`` over stable modes, with the saddle
missing exactly one stable mode (its unstable one). The products are evaluated as
sums of logarithms of ``hbar * omega`` in eV; the single leftover ``hbar * omega``
is converted to a linear frequency in Hz via ``E / h``. No Hz -> ps⁻¹ conversion
happens here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .constants import hbar_omega_ev_to_hz
from .normal_modes import ModeSpectrum, normal_modes_from_hessian
from .result import PrefactorRejected, PrefactorRejection


@dataclass(frozen=True)
class VineyardEstimate:
    """A Vineyard prefactor together with the spectrum counts that produced it.

    Attributes
    ----------
    nu0_hz : float
        Linear Vineyard frequency in Hz (finite, > 0).
    n_positive_min, n_negative_min, n_zero_min : int
        Mode counts at the minimum after projection.
    n_positive_saddle, n_negative_saddle, n_zero_saddle : int
        Mode counts at the saddle after projection.
    n_projected : int
        Assumed zero modes projected out of each spectrum.

    """

    nu0_hz: float
    n_positive_min: int
    n_negative_min: int
    n_zero_min: int
    n_positive_saddle: int
    n_negative_saddle: int
    n_zero_saddle: int
    n_projected: int


def vineyard_from_spectra(min_spec: ModeSpectrum, sad_spec: ModeSpectrum) -> float:
    """Return the Vineyard prefactor in Hz from two classified spectra.

    The saddle is checked first because its rejection applies to both directions
    of an event; the minimum is checked second. Dimensions are not compared here,
    so routing two different-sized spectra exercises ``MODE_COUNT_MISMATCH``.

    Parameters
    ----------
    min_spec : ModeSpectrum
        Spectrum classified with ``expect_saddle=False``.
    sad_spec : ModeSpectrum
        Spectrum classified with ``expect_saddle=True``.

    Returns
    -------
    float
        ``nu0`` in Hz.

    Raises
    ------
    ValueError
        If the spectra were classified with the wrong ``expect_saddle`` flags.
    PrefactorRejected
        ``SADDLE_NOT_FIRST_ORDER``, ``UNSTABLE_MINIMUM``, ``MODE_COUNT_MISMATCH`` or
        ``NONFINITE_PREFACTOR`` (the last also when the log-ratio of the products
        overflows to ``inf`` or underflows to ``0.0`` in double precision).

    """
    if min_spec.expect_saddle or not sad_spec.expect_saddle:
        raise ValueError(
            "vineyard_from_spectra needs a minimum spectrum (expect_saddle=False) "
            "and a saddle spectrum (expect_saddle=True)"
        )
    sad_spec.require_valid()
    min_spec.require_valid()
    if min_spec.n_positive != sad_spec.n_positive + 1:
        raise PrefactorRejected(
            PrefactorRejection.MODE_COUNT_MISMATCH,
            f"expected N_positive(min) == N_positive(saddle) + 1, got "
            f"{min_spec.n_positive} and {sad_spec.n_positive}",
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = float(
            np.sum(np.log(min_spec.omegas_ev)) - np.sum(np.log(sad_spec.omegas_ev))
        )
    if not math.isfinite(log_ratio):
        raise PrefactorRejected(
            PrefactorRejection.NONFINITE_PREFACTOR,
            f"log-product of mode frequencies is not finite ({log_ratio})",
        )
    # np.exp saturates to inf (and underflows to 0.0) instead of raising
    # OverflowError like math.exp, so the finite-positive guard below owns both
    # ends of the range and reports NONFINITE_PREFACTOR.
    with np.errstate(over="ignore", under="ignore"):
        leftover_hbar_omega_ev = float(np.exp(log_ratio))
    nu0_hz = hbar_omega_ev_to_hz(leftover_hbar_omega_ev)
    if not math.isfinite(nu0_hz) or nu0_hz <= 0.0:
        raise PrefactorRejected(
            PrefactorRejection.NONFINITE_PREFACTOR,
            f"Vineyard prefactor is not a finite positive number ({nu0_hz})",
        )
    return float(nu0_hz)


def _check_same_dimension(h_init: Any, h_sad: Any) -> None:
    """Raise ``ValueError`` when the two Hessians do not share one shape."""
    shape_init = np.shape(h_init)
    shape_sad = np.shape(h_sad)
    if shape_init != shape_sad:
        raise ValueError(
            "minimum and saddle Hessians must share one dimension, got "
            f"{shape_init} and {shape_sad}"
        )


def vineyard_prefactor_detailed(
    H_mw_init: Any,
    H_mw_sad: Any,
    *,
    zero_mode_tol: float,
    n_zero_modes: int = 0,
) -> VineyardEstimate:
    """Compute the Vineyard prefactor and return it with the spectrum counts.

    Parameters
    ----------
    H_mw_init : array_like
        ``(M, M)`` mass-weighted Hessian at the minimum (same free atoms and order
        as the saddle).
    H_mw_sad : array_like
        ``(M, M)`` mass-weighted Hessian at the saddle.
    zero_mode_tol : float
        Eigenvalue tolerance in eV / (amu Å²).
    n_zero_modes : int, optional
        Assumed zero modes to project out of each spectrum; ``0`` for
        frozen-boundary partial Hessians.

    Returns
    -------
    VineyardEstimate
        ``nu0_hz`` and the mode counts.

    Raises
    ------
    ValueError
        For mismatched, non-square or asymmetric Hessians (plumbing).
    PrefactorRejected
        For every scientific rejection.

    """
    _check_same_dimension(H_mw_init, H_mw_sad)
    sad_spec = normal_modes_from_hessian(
        H_mw_sad,
        expect_saddle=True,
        zero_mode_tol=zero_mode_tol,
        n_zero_modes=n_zero_modes,
    )
    min_spec = normal_modes_from_hessian(
        H_mw_init,
        expect_saddle=False,
        zero_mode_tol=zero_mode_tol,
        n_zero_modes=n_zero_modes,
    )
    nu0_hz = vineyard_from_spectra(min_spec, sad_spec)
    return VineyardEstimate(
        nu0_hz=nu0_hz,
        n_positive_min=min_spec.n_positive,
        n_negative_min=min_spec.n_negative,
        n_zero_min=min_spec.n_zero,
        n_positive_saddle=sad_spec.n_positive,
        n_negative_saddle=sad_spec.n_negative,
        n_zero_saddle=sad_spec.n_zero,
        n_projected=int(n_zero_modes),
    )


def vineyard_prefactor(
    H_mw_init: Any,
    H_mw_sad: Any,
    *,
    zero_mode_tol: float,
    n_zero_modes: int = 0,
) -> float:
    """Return the Vineyard prefactor in Hz.

    See :func:`vineyard_prefactor_detailed` for parameters and the exceptions.

    Parameters
    ----------
    H_mw_init : array_like
        ``(M, M)`` mass-weighted Hessian at the minimum.
    H_mw_sad : array_like
        ``(M, M)`` mass-weighted Hessian at the saddle.
    zero_mode_tol : float
        Eigenvalue tolerance in eV / (amu Å²).
    n_zero_modes : int, optional
        Assumed zero modes to project out; ``0`` for frozen-boundary Hessians.

    Returns
    -------
    float
        ``nu0`` in Hz (linear frequency).

    """
    return vineyard_prefactor_detailed(
        H_mw_init, H_mw_sad, zero_mode_tol=zero_mode_tol, n_zero_modes=n_zero_modes
    ).nu0_hz


__all__ = [
    "VineyardEstimate",
    "vineyard_from_spectra",
    "vineyard_prefactor",
    "vineyard_prefactor_detailed",
]
