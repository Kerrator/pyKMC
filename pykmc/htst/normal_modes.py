"""Normal-mode classification of a mass-weighted Hessian.

Tolerance semantics for an eigenvalue ``lambda`` of the mass-weighted Hessian in
eV / (amu Å²), with ``tol = zero_mode_tol``:

- ``lambda < -tol``   : unstable (negative) mode
- ``|lambda| <= tol`` : zero mode
- ``lambda > tol``    : stable (positive) mode

When ``n_zero_modes > 0`` the unstable modes are identified *before* the
``n_zero_modes`` smallest-|lambda| non-negative candidates are projected out, so a
genuine imaginary mode that is small relative to unrelaxed translational modes is
never absorbed as a zero mode. Frozen-boundary partial Hessians have no
translational invariance and use ``n_zero_modes = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .constants import HBAR_OMEGA_EV
from .result import PrefactorRejected, PrefactorRejection

SYMMETRY_ATOL: float = 1.0e-8
"""Absolute tolerance on ``H - H.T`` before a Hessian is rejected as asymmetric."""


@dataclass(frozen=True, eq=False)
class ModeSpectrum:
    """Classified eigen-spectrum of one mass-weighted Hessian.

    Attributes
    ----------
    eigenvalues : np.ndarray
        All eigenvalues in ascending order, eV / (amu Å²).
    expect_saddle : bool
        Whether the caller expected a first-order saddle.
    zero_mode_tol : float
        Tolerance used for the classification.
    n_projected : int
        Modes removed as assumed translational/rotational zero modes
        (``n_zero_modes``), chosen as the smallest-|lambda| non-negative candidates.
    n_negative : int
        Modes with ``lambda < -tol`` (identified before projection, never projected).
    n_zero : int
        Remaining modes with ``|lambda| <= tol`` after projection.
    n_positive : int
        Remaining modes with ``lambda > tol``.
    omegas_ev : np.ndarray
        ``hbar * omega`` in eV of the positive modes, ascending.
    omega_unstable_ev : float or None
        ``hbar * |omega|`` in eV of the unstable mode when exactly one exists.

    """

    eigenvalues: np.ndarray
    expect_saddle: bool
    zero_mode_tol: float
    n_projected: int
    n_negative: int
    n_zero: int
    n_positive: int
    omegas_ev: np.ndarray
    omega_unstable_ev: float | None

    @property
    def n_modes(self) -> int:
        """Return the dimension of the Hessian."""
        return int(self.eigenvalues.size)

    def rejection(self) -> tuple[PrefactorRejection, str] | None:
        """Return the scientific rejection this spectrum implies, or ``None``.

        A minimum needs no unstable and no zero modes (``UNSTABLE_MINIMUM``
        otherwise); a saddle needs exactly one unstable and no zero modes
        (``SADDLE_NOT_FIRST_ORDER`` otherwise).
        """
        if self.expect_saddle:
            if self.n_negative != 1 or self.n_zero != 0:
                return (
                    PrefactorRejection.SADDLE_NOT_FIRST_ORDER,
                    "saddle spectrum has "
                    f"{self.n_negative} unstable and {self.n_zero} zero mode(s) "
                    f"(tol={self.zero_mode_tol}); a first-order saddle needs exactly "
                    "one unstable and no zero modes; smallest eigenvalues: "
                    f"{_head(self.eigenvalues)}",
                )
            return None
        if self.n_negative != 0 or self.n_zero != 0:
            return (
                PrefactorRejection.UNSTABLE_MINIMUM,
                "minimum spectrum has "
                f"{self.n_negative} unstable and {self.n_zero} zero mode(s) "
                f"(tol={self.zero_mode_tol}); smallest eigenvalues: "
                f"{_head(self.eigenvalues)}",
            )
        return None

    def require_valid(self) -> None:
        """Raise :class:`PrefactorRejected` if :meth:`rejection` is not ``None``."""
        rejection = self.rejection()
        if rejection is not None:
            raise PrefactorRejected(*rejection)


def _head(values: np.ndarray, count: int = 4) -> list[float]:
    """Return the first ``count`` entries of ``values`` as a plain float list."""
    return [float(v) for v in values[:count]]


def _square_symmetric(h_mw: Any) -> np.ndarray:
    """Return ``h_mw`` as a float square symmetric array, or raise.

    Raises ``ValueError`` for non-square or asymmetric input (plumbing) and
    ``PrefactorRejected(NONFINITE_HESSIAN)`` for NaN/inf entries (science).
    """
    if np.iscomplexobj(h_mw):
        raise ValueError("H_mw must be real-valued, got a complex array")
    try:
        arr = np.asarray(h_mw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"H_mw is not numeric: {exc}") from exc
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] == 0:
        raise ValueError(f"H_mw must be a non-empty square matrix, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        n_bad = int(np.count_nonzero(~np.isfinite(arr)))
        raise PrefactorRejected(
            PrefactorRejection.NONFINITE_HESSIAN,
            f"mass-weighted Hessian of shape {arr.shape} has {n_bad} non-finite entries",
        )
    if not np.allclose(arr, arr.T, atol=SYMMETRY_ATOL, rtol=0.0):
        max_asym = float(np.max(np.abs(arr - arr.T)))
        raise ValueError(
            f"H_mw must be symmetric within {SYMMETRY_ATOL}; max |H - H.T| = {max_asym}"
        )
    return arr


def normal_modes_from_hessian(
    H_mw: Any,
    *,
    expect_saddle: bool,
    zero_mode_tol: float,
    n_zero_modes: int = 0,
) -> ModeSpectrum:
    """Diagonalise and classify a mass-weighted Hessian.

    This function only *classifies*; it never rejects on scientific grounds. Use
    :meth:`ModeSpectrum.rejection` or :meth:`ModeSpectrum.require_valid` for the
    acceptance rules.

    Parameters
    ----------
    H_mw : array_like
        ``(M, M)`` real mass-weighted Hessian in eV / (amu Å²), symmetric to
        within :data:`SYMMETRY_ATOL` (absolute); callers assembling it from raw
        engine output should return ``0.5 * (H + H.T)``.
    expect_saddle : bool
        Whether a first-order saddle is expected (recorded in the spectrum).
    zero_mode_tol : float
        Eigenvalue tolerance in eV / (amu Å²); must be finite and >= 0.
    n_zero_modes : int, optional
        Number of assumed translational/rotational zero modes to project out
        (``0`` for frozen-boundary partial Hessians). Chosen among non-negative
        candidates only, by smallest ``|lambda|``.

    Returns
    -------
    ModeSpectrum
        Counts of unstable, zero and stable modes and the stable ``hbar * omega``.

    Raises
    ------
    ValueError
        For a non-square, complex or asymmetric Hessian, a negative
        ``n_zero_modes``, more projected modes than non-negative candidates, or a
        bad tolerance.
    PrefactorRejected
        With ``NONFINITE_HESSIAN`` when the Hessian contains NaN or inf.

    """
    if isinstance(n_zero_modes, bool) or not isinstance(
        n_zero_modes, (int, np.integer)
    ):
        raise ValueError(f"n_zero_modes must be an int, got {n_zero_modes!r}")
    if n_zero_modes < 0:
        raise ValueError(f"n_zero_modes must be >= 0, got {n_zero_modes}")
    tol = float(zero_mode_tol)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError(
            f"zero_mode_tol must be finite and >= 0, got {zero_mode_tol!r}"
        )
    arr = _square_symmetric(H_mw)

    eigenvalues = np.linalg.eigh(arr)[0]  # ascending
    negative_mask = eigenvalues < -tol
    n_negative = int(np.count_nonzero(negative_mask))

    candidates = np.flatnonzero(~negative_mask)
    if n_zero_modes > candidates.size:
        raise ValueError(
            f"cannot project {n_zero_modes} zero modes out of {candidates.size} "
            "non-negative candidates"
        )
    order = np.argsort(np.abs(eigenvalues[candidates]), kind="stable")
    projected = candidates[order[:n_zero_modes]]
    kept_mask = ~negative_mask
    kept_mask[projected] = False
    kept = eigenvalues[kept_mask]

    zero_mask = np.abs(kept) <= tol
    positive = kept[~zero_mask]
    n_zero = int(np.count_nonzero(zero_mask))
    n_positive = int(positive.size)
    omegas_ev = HBAR_OMEGA_EV * np.sqrt(positive)

    omega_unstable_ev: float | None = None
    if n_negative == 1:
        omega_unstable_ev = float(
            HBAR_OMEGA_EV * np.sqrt(-eigenvalues[negative_mask][0])
        )

    return ModeSpectrum(
        eigenvalues=eigenvalues,
        expect_saddle=bool(expect_saddle),
        zero_mode_tol=tol,
        n_projected=int(n_zero_modes),
        n_negative=n_negative,
        n_zero=n_zero,
        n_positive=n_positive,
        omegas_ev=omegas_ev,
        omega_unstable_ev=omega_unstable_ev,
    )


__all__ = ["ModeSpectrum", "SYMMETRY_ATOL", "normal_modes_from_hessian"]
