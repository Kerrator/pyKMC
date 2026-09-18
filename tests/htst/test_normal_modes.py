"""Tests for normal-mode classification and the tolerance/zero-mode rules."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pykmc.htst import (
    PrefactorRejected,
    PrefactorRejection,
    fd_hessian_fn,
    normal_modes_from_hessian,
)

# Independent CODATA restatement for the unit oracles below.
EV_J = 1.602176634e-19
AMU_KG = 1.66053906660e-27
HBAR_EV_S = 6.582119569e-16
H_EV_S = 4.135667696e-15
TOL = 1.0e-6


def toy_hessian(eigvals: np.ndarray, seed: int = 42) -> np.ndarray:
    """Return a symmetric matrix with the given eigenvalues and random eigenvectors."""
    n = len(eigvals)
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    h = q @ np.diag(eigvals) @ q.T
    return 0.5 * (h + h.T)


def test_diagonal_springs_unequal_masses_eigenvalues_hbar_omega_and_hz() -> None:
    """Closed-form ``lambda_i = k_i/m_i``, ``hbar omega`` and Hz from CODATA literals."""
    k = np.array([2.0, 0.5, 3.0])
    m = np.array([1.0, 4.0, 58.69])
    h_mw = np.diag(np.repeat(k / m, 3))
    spec = normal_modes_from_hessian(h_mw, expect_saddle=False, zero_mode_tol=TOL)
    assert (spec.n_negative, spec.n_zero, spec.n_positive) == (0, 0, 9)
    assert spec.rejection() is None
    lam = np.sort(np.repeat(k / m, 3))
    np.testing.assert_allclose(spec.eigenvalues, lam, rtol=1.0e-12)
    hbar_omega = HBAR_EV_S * np.sqrt(lam * EV_J / (AMU_KG * 1.0e-20))
    np.testing.assert_allclose(spec.omegas_ev, hbar_omega, rtol=1.0e-9)
    nu_hz = np.sqrt(lam * EV_J / (AMU_KG * 1.0e-20)) / (2.0 * math.pi)
    np.testing.assert_allclose(spec.omegas_ev / H_EV_S, nu_hz, rtol=1.0e-9)


def test_coupled_two_mass_oscillator_closed_form() -> None:
    """Two masses on walls (kappa) coupled by k: analytic 2x2 eigenvalues, also via FD."""
    kappa, k, m1, m2 = 1.2, 0.7, 1.0, 3.5
    a, b, c = (kappa + k) / m1, (kappa + k) / m2, -k / math.sqrt(m1 * m2)
    h2 = np.array([[a, c], [c, b]])
    disc = math.sqrt(((a - b) / 2.0) ** 2 + c * c)
    lam_lo, lam_hi = (a + b) / 2.0 - disc, (a + b) / 2.0 + disc
    spec = normal_modes_from_hessian(h2, expect_saddle=False, zero_mode_tol=TOL)
    np.testing.assert_allclose(spec.eigenvalues, [lam_lo, lam_hi], rtol=1.0e-12)

    eq = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    def forces(pos: np.ndarray) -> np.ndarray:
        d = pos - eq
        f = -kappa * d
        stretch = d[0, 0] - d[1, 0]
        f[0, 0] -= k * stretch
        f[1, 0] += k * stretch
        return f

    h6 = fd_hessian_fn(forces, np.array([m1, m2]), 1.0e-3)(eq, np.array([0, 1]))
    spec6 = normal_modes_from_hessian(h6, expect_saddle=False, zero_mode_tol=TOL)
    expected = np.sort([lam_lo, lam_hi, kappa / m1, kappa / m1, kappa / m2, kappa / m2])
    np.testing.assert_allclose(spec6.eigenvalues, expected, rtol=1.0e-9)


# A sub-converged saddle whose imaginary mode is SMALLER in magnitude than the three
# smallest unrelaxed near-zero positives (0.015, 0.018, 0.025). Projecting the three
# smallest-|lambda| modes from the whole spectrum (the legacy rule) would absorb
# -0.010 as a zero mode and report (0, 0, 4); the contract rule keeps it and reports
# (1, 0, 3). A spectrum where the imaginary mode is larger than the three smallest
# positives cannot tell the two rules apart.
DISCRIMINATING_EIGS = np.array([-0.010, 0.015, 0.018, 0.025, 0.029, 0.030, 0.04])
LAM_TO_HBAR_OMEGA_SQ = HBAR_EV_S**2 * EV_J / (AMU_KG * 1.0e-20)


def test_unstable_mode_identified_before_zero_mode_projection() -> None:
    """The unstable mode is protected even when it is among the smallest |lambda|."""
    h = toy_hessian(DISCRIMINATING_EIGS, seed=99)
    spec = normal_modes_from_hessian(
        h, expect_saddle=True, zero_mode_tol=TOL, n_zero_modes=3
    )
    assert spec.n_negative == 1
    assert spec.n_projected == 3
    assert spec.n_zero == 0
    assert spec.n_positive == 3
    assert spec.rejection() is None
    hbar_omega_expected = HBAR_EV_S * math.sqrt(0.010 * EV_J / (AMU_KG * 1.0e-20))
    assert spec.omega_unstable_ev == pytest.approx(hbar_omega_expected, rel=1.0e-9)
    # The projected set is exactly {0.015, 0.018, 0.025}: the survivors are the rest.
    np.testing.assert_allclose(
        spec.omegas_ev**2,
        LAM_TO_HBAR_OMEGA_SQ * np.array([0.029, 0.030, 0.04]),
        rtol=1.0e-9,
    )


def test_projection_never_absorbs_the_smallest_magnitude_negative_mode() -> None:
    """Projecting exactly n_zero_modes = number of smaller positives keeps the negative."""
    # -0.010 is the smallest |lambda| of all; with n_zero_modes=1 the legacy rule
    # would project it and see a pure minimum, the contract rule projects 0.015.
    h = toy_hessian(DISCRIMINATING_EIGS, seed=7)
    spec = normal_modes_from_hessian(
        h, expect_saddle=True, zero_mode_tol=TOL, n_zero_modes=1
    )
    assert (spec.n_negative, spec.n_zero, spec.n_positive) == (1, 0, 5)
    np.testing.assert_allclose(
        spec.omegas_ev**2,
        LAM_TO_HBAR_OMEGA_SQ * np.array([0.018, 0.025, 0.029, 0.030, 0.04]),
        rtol=1.0e-9,
    )


def test_n_zero_modes_zero_versus_three_on_the_same_spectrum() -> None:
    """With no projection every near-zero positive mode counts as stable."""
    h = toy_hessian(DISCRIMINATING_EIGS, seed=99)
    spec0 = normal_modes_from_hessian(h, expect_saddle=True, zero_mode_tol=TOL)
    spec3 = normal_modes_from_hessian(
        h, expect_saddle=True, zero_mode_tol=TOL, n_zero_modes=3
    )
    assert (spec0.n_negative, spec0.n_zero, spec0.n_positive) == (1, 0, 6)
    assert (spec3.n_negative, spec3.n_zero, spec3.n_positive) == (1, 0, 3)
    assert spec0.n_projected == 0
    np.testing.assert_allclose(
        spec0.omegas_ev**2,
        LAM_TO_HBAR_OMEGA_SQ * np.array([0.015, 0.018, 0.025, 0.029, 0.030, 0.04]),
        rtol=1.0e-9,
    )
    assert spec0.omega_unstable_ev == pytest.approx(spec3.omega_unstable_ev)


def test_contract_keyword_spelling_is_accepted() -> None:
    """The contract writes ``H_mw``; keyword calls with that spelling work."""
    h = toy_hessian(np.array([-0.1, 0.2, 0.3]))
    spec = normal_modes_from_hessian(H_mw=h, expect_saddle=True, zero_mode_tol=TOL)
    assert spec.n_negative == 1


def test_tolerance_readings_pinned_on_both_sides() -> None:
    """``lambda < -tol`` unstable, ``|lambda| <= tol`` zero, ``lambda > tol`` stable."""
    # 1x1 problems make eigh exact, so the boundaries themselves can be pinned.
    for lam, expected in [
        (-2.0 * TOL, (1, 0, 0)),
        (-TOL * (1.0 + 1.0e-6), (1, 0, 0)),
        (-TOL, (0, 1, 0)),
        (-0.5 * TOL, (0, 1, 0)),
        (0.0, (0, 1, 0)),
        (0.5 * TOL, (0, 1, 0)),
        (TOL, (0, 1, 0)),
        (TOL * (1.0 + 1.0e-6), (0, 0, 1)),
        (2.0 * TOL, (0, 0, 1)),
    ]:
        spec = normal_modes_from_hessian(
            np.array([[lam]]), expect_saddle=False, zero_mode_tol=TOL
        )
        assert (spec.n_negative, spec.n_zero, spec.n_positive) == expected, lam
    # Mixed spectrum: only -2tol is unstable, five sit inside the band, one is stable.
    h = np.diag([-2 * TOL, -TOL, -0.5 * TOL, 0.0, 0.5 * TOL, TOL, 2 * TOL])
    spec = normal_modes_from_hessian(h, expect_saddle=True, zero_mode_tol=TOL)
    assert (spec.n_negative, spec.n_zero, spec.n_positive) == (1, 5, 1)


@pytest.mark.parametrize(
    ("eigs", "expect_saddle", "code"),
    [
        (np.array([0.1, 0.2, 0.3]), True, PrefactorRejection.SADDLE_NOT_FIRST_ORDER),
        (np.array([-0.1, -0.05, 0.3]), True, PrefactorRejection.SADDLE_NOT_FIRST_ORDER),
        (np.array([-0.1, 0.0, 0.3]), True, PrefactorRejection.SADDLE_NOT_FIRST_ORDER),
        (np.array([-0.1, 0.2, 0.3]), False, PrefactorRejection.UNSTABLE_MINIMUM),
        (np.array([0.0, 0.2, 0.3]), False, PrefactorRejection.UNSTABLE_MINIMUM),
        (np.array([-0.5e-6, 0.2, 0.3]), False, PrefactorRejection.UNSTABLE_MINIMUM),
    ],
)
def test_spectrum_rejections(
    eigs: np.ndarray, expect_saddle: bool, code: PrefactorRejection
) -> None:
    """Zero/two unstable saddle modes, zero modes, and unstable minima are rejections."""
    spec = normal_modes_from_hessian(
        toy_hessian(eigs), expect_saddle=expect_saddle, zero_mode_tol=TOL
    )
    rejection = spec.rejection()
    assert rejection is not None and rejection[0] is code
    with pytest.raises(PrefactorRejected) as info:
        spec.require_valid()
    assert info.value.reason_code is code


def test_valid_saddle_and_minimum_have_no_rejection() -> None:
    """A first-order saddle and a stable minimum pass."""
    sad = normal_modes_from_hessian(
        toy_hessian(np.array([-0.1, 0.2, 0.3])), expect_saddle=True, zero_mode_tol=TOL
    )
    assert sad.rejection() is None and sad.n_negative == 1
    mini = normal_modes_from_hessian(
        toy_hessian(np.array([0.1, 0.2, 0.3])), expect_saddle=False, zero_mode_tol=TOL
    )
    assert mini.rejection() is None and mini.omega_unstable_ev is None


def test_nonfinite_hessian_is_a_rejection_but_shape_errors_raise() -> None:
    """NaN/inf entries are NONFINITE_HESSIAN; non-square or asymmetric input raises."""
    h = np.diag([0.1, 0.2, 0.3])
    h_nan = h.copy()
    h_nan[0, 1] = h_nan[1, 0] = np.nan
    with pytest.raises(PrefactorRejected) as info:
        normal_modes_from_hessian(h_nan, expect_saddle=False, zero_mode_tol=TOL)
    assert info.value.reason_code is PrefactorRejection.NONFINITE_HESSIAN
    h_inf = h.copy()
    h_inf[2, 2] = np.inf
    with pytest.raises(PrefactorRejected):
        normal_modes_from_hessian(h_inf, expect_saddle=False, zero_mode_tol=TOL)
    with pytest.raises(ValueError):
        normal_modes_from_hessian(
            np.zeros((2, 3)), expect_saddle=False, zero_mode_tol=TOL
        )
    asym = h.copy()
    asym[0, 1] = 1.0e-3
    with pytest.raises(ValueError):
        normal_modes_from_hessian(asym, expect_saddle=False, zero_mode_tol=TOL)
    with pytest.raises(ValueError, match="complex"):
        normal_modes_from_hessian(
            h.astype(complex), expect_saddle=False, zero_mode_tol=TOL
        )
    with pytest.raises(ValueError):
        normal_modes_from_hessian(
            h, expect_saddle=False, zero_mode_tol=TOL, n_zero_modes=4
        )
    with pytest.raises(ValueError):
        normal_modes_from_hessian(
            h, expect_saddle=False, zero_mode_tol=TOL, n_zero_modes=-1
        )
    with pytest.raises(ValueError):
        normal_modes_from_hessian(h, expect_saddle=False, zero_mode_tol=-1.0)


def test_spectrum_is_permutation_invariant() -> None:
    """Relabelling degrees of freedom leaves the classified spectrum unchanged."""
    eigs = np.array([-0.07, 0.01, 0.2, 0.3, 0.45, 0.9])
    h = toy_hessian(eigs, seed=5)
    perm = np.random.default_rng(1).permutation(6)
    p = np.eye(6)[perm]
    spec_a = normal_modes_from_hessian(h, expect_saddle=True, zero_mode_tol=TOL)
    spec_b = normal_modes_from_hessian(
        p @ h @ p.T, expect_saddle=True, zero_mode_tol=TOL
    )
    np.testing.assert_allclose(spec_a.eigenvalues, spec_b.eigenvalues, rtol=1.0e-10)
    np.testing.assert_allclose(spec_a.omegas_ev, spec_b.omegas_ev, rtol=1.0e-10)
