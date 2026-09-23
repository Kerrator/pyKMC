"""Vineyard prefactor oracles: closed-form products, scaling, permutation, stability."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pytest

from pykmc.htst import (
    PrefactorRejected,
    PrefactorRejection,
    fd_hessian_fn,
    normal_modes_from_hessian,
    vineyard_from_spectra,
    vineyard_prefactor,
    vineyard_prefactor_detailed,
)

# Independent CODATA restatement for the unit oracles below.
EV_J = 1.602176634e-19
AMU_KG = 1.66053906660e-27
TOL = 1.0e-6


def nu_hz(lam: float) -> float:
    """Closed-form linear frequency in Hz of a mass-weighted eigenvalue (SI route)."""
    return math.sqrt(lam * EV_J / (AMU_KG * 1.0e-20)) / (2.0 * math.pi)


def toy_hessian(eigvals: np.ndarray, seed: int = 42) -> np.ndarray:
    """Return a symmetric matrix with the given eigenvalues and random eigenvectors."""
    n = len(eigvals)
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    h = q @ np.diag(eigvals) @ q.T
    return 0.5 * (h + h.T)


def test_diagonal_springs_leftover_mode_is_the_prefactor() -> None:
    """Saddle = minimum with one spring flipped: ``nu0 = nu(lambda_flipped)`` in Hz."""
    lam = np.array([0.3, 0.5, 0.4, 0.8])
    h_min = np.diag(lam)
    h_sad = np.diag([-0.2, 0.5, 0.4, 0.8])
    nu0 = vineyard_prefactor(h_min, h_sad, zero_mode_tol=TOL)
    assert nu0 == pytest.approx(nu_hz(0.3), rel=1.0e-9)
    assert 1.0e12 < nu0 < 1.0e14


def test_general_product_ratio_in_hz() -> None:
    """``nu0 = prod nu_min / prod nu_sad`` with every factor from the SI closed form."""
    lam_min = np.array([0.3, 0.5, 0.7, 1.1])
    lam_sad = np.array([-0.2, 0.4, 0.6, 0.9])
    nu0 = vineyard_prefactor(
        toy_hessian(lam_min, 1), toy_hessian(lam_sad, 2), zero_mode_tol=TOL
    )
    expected = np.prod([nu_hz(x) for x in lam_min]) / np.prod(
        [nu_hz(x) for x in lam_sad[1:]]
    )
    assert nu0 == pytest.approx(expected, rel=1.0e-9)


def test_coupled_oscillator_minimum_against_diagonal_saddle() -> None:
    """Coupled 2x2 minimum with closed-form eigenvalues feeds the product correctly."""
    kappa, k, m1, m2 = 1.2, 0.7, 1.0, 3.5
    a, b, c = (kappa + k) / m1, (kappa + k) / m2, -k / math.sqrt(m1 * m2)
    disc = math.sqrt(((a - b) / 2.0) ** 2 + c * c)
    lam_lo, lam_hi = (a + b) / 2.0 - disc, (a + b) / 2.0 + disc
    h_min = np.array([[a, c], [c, b]])
    h_sad = np.diag([-0.3, 0.55])
    nu0 = vineyard_prefactor(h_min, h_sad, zero_mode_tol=TOL)
    assert nu0 == pytest.approx(nu_hz(lam_lo) * nu_hz(lam_hi) / nu_hz(0.55), rel=1.0e-9)


def test_permutation_invariance_under_consistent_relabeling() -> None:
    """Applying one permutation to both Hessians leaves nu0 unchanged."""
    h_min = toy_hessian(np.array([0.1, 0.2, 0.35, 0.5, 0.7, 0.9]), seed=11)
    h_sad = toy_hessian(np.array([-0.15, 0.18, 0.3, 0.55, 0.65, 0.95]), seed=12)
    perm = np.random.default_rng(4).permutation(6)
    p = np.eye(6)[perm]
    ref = vineyard_prefactor(h_min, h_sad, zero_mode_tol=TOL)
    permuted = vineyard_prefactor(p @ h_min @ p.T, p @ h_sad @ p.T, zero_mode_tol=TOL)
    assert permuted == pytest.approx(ref, rel=1.0e-10)


@pytest.mark.parametrize("scale", [0.25, 2.0, 9.0])
def test_isotope_scaling_nu0_goes_as_inverse_sqrt_mass(scale: float) -> None:
    """``m -> a m`` scales ``nu0`` by ``a**-0.5``, both on H/a and through the FD kernel."""
    h_min = toy_hessian(np.array([0.2, 0.4, 0.6]), seed=3)
    h_sad = toy_hessian(np.array([-0.1, 0.3, 0.5]), seed=4)
    ref = vineyard_prefactor(h_min, h_sad, zero_mode_tol=TOL)
    scaled = vineyard_prefactor(h_min / scale, h_sad / scale, zero_mode_tol=TOL)
    assert scaled / ref == pytest.approx(scale**-0.5, rel=1.0e-9)

    # Through the finite-difference kernel with explicit masses: one free atom with a
    # minimum spring (k_min) and a saddle where the x spring is flipped.
    eq = np.zeros((1, 3))
    k_min = np.array([1.0, 0.5, 0.7])
    k_sad = np.array([-0.4, 0.5, 0.7])

    def make(k: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        def forces(pos: np.ndarray) -> np.ndarray:
            return -k[None, :] * (pos - eq)

        return forces

    free = np.array([0])
    m = 3.0
    hess_min = fd_hessian_fn(make(k_min), np.array([m]), 1e-3)
    hess_sad = fd_hessian_fn(make(k_sad), np.array([m]), 1e-3)
    hess_min_s = fd_hessian_fn(make(k_min), np.array([m * scale]), 1e-3)
    hess_sad_s = fd_hessian_fn(make(k_sad), np.array([m * scale]), 1e-3)
    nu_ref = vineyard_prefactor(
        hess_min(eq, free), hess_sad(eq, free), zero_mode_tol=TOL
    )
    nu_scaled = vineyard_prefactor(
        hess_min_s(eq, free), hess_sad_s(eq, free), zero_mode_tol=TOL
    )
    assert nu_scaled / nu_ref == pytest.approx(scale**-0.5, rel=1.0e-9)
    assert nu_ref == pytest.approx(nu_hz(k_min[0] / m), rel=1.0e-9)


def test_log_product_is_stable_at_500_modes() -> None:
    """A naive product of 500 stiff modes overflows; the log product does not."""
    lam_big = 2.4e8  # hbar*omega ~ 1000 eV per mode
    n = 500
    h_min = np.diag(np.full(n, lam_big))
    h_sad = np.diag(np.concatenate([[-1.0], np.full(n - 1, lam_big)]))
    spec_min = normal_modes_from_hessian(h_min, expect_saddle=False, zero_mode_tol=TOL)
    with np.errstate(over="ignore"):
        assert not np.isfinite(np.prod(spec_min.omegas_ev))  # naive route overflows
    est = vineyard_prefactor_detailed(h_min, h_sad, zero_mode_tol=TOL)
    assert math.isfinite(est.nu0_hz)
    assert est.nu0_hz == pytest.approx(nu_hz(lam_big), rel=1.0e-9)
    assert (est.n_positive_min, est.n_positive_saddle, est.n_negative_saddle) == (
        n,
        n - 1,
        1,
    )


def test_detailed_counts_and_projection_with_appended_zero_modes() -> None:
    """``n_zero_modes=3`` projects three exact zeros appended to both spectra."""
    lam_min = np.concatenate([[0.5, 0.5, 0.5, 0.5, 0.5], np.zeros(3)])
    lam_sad = np.concatenate([[-0.1, 0.5, 0.5, 0.5, 0.5], np.zeros(3)])
    est = vineyard_prefactor_detailed(
        toy_hessian(lam_min, 7),
        toy_hessian(lam_sad, 8),
        zero_mode_tol=TOL,
        n_zero_modes=3,
    )
    assert est.n_projected == 3
    assert (est.n_positive_min, est.n_zero_min, est.n_negative_min) == (5, 0, 0)
    assert (est.n_positive_saddle, est.n_zero_saddle, est.n_negative_saddle) == (
        4,
        0,
        1,
    )
    assert est.nu0_hz == pytest.approx(nu_hz(0.5), rel=1.0e-9)
    # The same spectra without projection are rejected: the zeros are real zero modes.
    with pytest.raises(PrefactorRejected) as info:
        vineyard_prefactor(
            toy_hessian(lam_min, 7), toy_hessian(lam_sad, 8), zero_mode_tol=TOL
        )
    assert info.value.reason_code is PrefactorRejection.SADDLE_NOT_FIRST_ORDER


@pytest.mark.parametrize(
    ("lam_min", "lam_sad", "code"),
    [
        ([0.1, 0.2, 0.3], [0.1, 0.2, 0.3], PrefactorRejection.SADDLE_NOT_FIRST_ORDER),
        ([0.1, 0.2, 0.3], [-0.1, -0.2, 0.3], PrefactorRejection.SADDLE_NOT_FIRST_ORDER),
        ([-0.1, 0.2, 0.3], [-0.1, 0.2, 0.3], PrefactorRejection.UNSTABLE_MINIMUM),
        ([0.0, 0.2, 0.3], [-0.1, 0.2, 0.3], PrefactorRejection.UNSTABLE_MINIMUM),
        # Both bad: the shared saddle is reported first.
        ([-0.1, 0.2, 0.3], [0.1, 0.2, 0.3], PrefactorRejection.SADDLE_NOT_FIRST_ORDER),
    ],
)
def test_scientific_rejections_raise_prefactor_rejected(
    lam_min: list[float], lam_sad: list[float], code: PrefactorRejection
) -> None:
    """Bad spectra raise PrefactorRejected with the contract reason codes."""
    with pytest.raises(PrefactorRejected) as info:
        vineyard_prefactor(
            toy_hessian(np.array(lam_min), 1),
            toy_hessian(np.array(lam_sad), 2),
            zero_mode_tol=TOL,
        )
    assert info.value.reason_code is code


def test_mismatched_dimensions_raise_value_error() -> None:
    """Different Hessian sizes are a plumbing bug, not a scientific rejection."""
    with pytest.raises(ValueError):
        vineyard_prefactor(
            np.diag([0.1, 0.2, 0.3]), np.diag([-0.1, 0.2]), zero_mode_tol=TOL
        )


def test_mode_count_mismatch_via_two_different_sized_spectra() -> None:
    """Routing accepted spectra of different size through the ratio hits the guard."""
    spec_min = normal_modes_from_hessian(
        np.diag([0.1, 0.2, 0.3, 0.4]), expect_saddle=False, zero_mode_tol=TOL
    )
    spec_sad = normal_modes_from_hessian(
        np.diag([-0.1, 0.2, 0.3]), expect_saddle=True, zero_mode_tol=TOL
    )
    with pytest.raises(PrefactorRejected) as info:
        vineyard_from_spectra(spec_min, spec_sad)
    assert info.value.reason_code is PrefactorRejection.MODE_COUNT_MISMATCH
    with pytest.raises(ValueError):
        vineyard_from_spectra(spec_sad, spec_min)  # swapped roles


def test_nonfinite_hessian_propagates_as_rejection() -> None:
    """A NaN in either Hessian surfaces as NONFINITE_HESSIAN."""
    h = np.diag([0.1, 0.2, 0.3])
    bad = h.copy()
    bad[1, 1] = np.nan
    with pytest.raises(PrefactorRejected) as info:
        vineyard_prefactor(bad, np.diag([-0.1, 0.2, 0.3]), zero_mode_tol=TOL)
    assert info.value.reason_code is PrefactorRejection.NONFINITE_HESSIAN


def test_contract_keyword_spelling_is_accepted() -> None:
    """The contract writes ``H_mw_init``/``H_mw_sad``; keyword calls work."""
    h_min = np.diag([0.3, 0.5, 0.7])
    h_sad = np.diag([-0.2, 0.4, 0.6])
    nu0 = vineyard_prefactor(H_mw_init=h_min, H_mw_sad=h_sad, zero_mode_tol=1.0e-6)
    detailed = vineyard_prefactor_detailed(
        H_mw_init=h_min, H_mw_sad=h_sad, zero_mode_tol=1.0e-6
    )
    assert nu0 == detailed.nu0_hz > 0.0


@pytest.mark.parametrize(
    ("lam_min", "lam_sad"),
    [
        # log-ratio ~ +1034 > 709.78: math.exp would raise OverflowError.
        ([1.0e300] * 3, [-0.2, 0.5, 0.5]),
        # 60 modes, every saddle mode just above tol: same overflow at scale.
        ([1.0e5] * 60, [-0.2] + [2.0e-6] * 59),
        # Underflow side: the product is exactly 0.0, not a positive frequency.
        ([2.0e-6] * 60, [-0.2] + [1.0e300] * 59),
    ],
)
def test_overflowing_log_ratio_is_nonfinite_prefactor_not_an_exception(
    lam_min: list[float], lam_sad: list[float]
) -> None:
    """Finite Hessians whose frequency ratio leaves double range are rejected."""
    with pytest.raises(PrefactorRejected) as info:
        vineyard_prefactor(np.diag(lam_min), np.diag(lam_sad), zero_mode_tol=TOL)
    assert info.value.reason_code is PrefactorRejection.NONFINITE_PREFACTOR
    assert "not a finite positive number" in info.value.detail
