"""Finite-difference mass-weighted partial Hessian against analytic second derivatives."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from pykmc.htst import HTSTRequestError, fd_hessian_fn, mass_weighted_partial_hessian

ForcesFn = Callable[[np.ndarray], np.ndarray]


def _diagonal_springs(k: np.ndarray, eq: np.ndarray) -> ForcesFn:
    """Return forces for independent per-atom isotropic springs ``F = -k_i (x - eq)``."""

    def forces(pos: np.ndarray) -> np.ndarray:
        return -k[:, None] * (pos - eq)

    return forces


def test_diagonal_springs_with_unequal_masses() -> None:
    """``H_mw = diag(k_i / m_i)`` with each atom's block scaled by its own mass."""
    k = np.array([2.0, 0.5, 3.0])
    m = np.array([1.0, 4.0, 58.69])
    eq = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    free = np.array([0, 1, 2])
    h = mass_weighted_partial_hessian(_diagonal_springs(k, eq), eq, m, free, 1.0e-3)
    assert h.shape == (9, 9)
    np.testing.assert_allclose(h, np.diag(np.repeat(k / m, 3)), atol=1.0e-9)


def test_quartic_per_dof_error_is_exactly_second_order() -> None:
    """For ``F = -k x - q x^3`` the central-difference error is exactly ``q h^2 / m``."""
    k, q, m, x0 = 1.5, 0.8, 2.0, 0.3
    eq = np.zeros((1, 3))
    pos = np.array([[x0, 0.0, 0.0]])

    def forces(p: np.ndarray) -> np.ndarray:
        d = p - eq
        return -k * d - q * d**3

    exact = (k + 3.0 * q * x0**2) / m
    errors = []
    for h_step in (2.0e-2, 1.0e-2, 5.0e-3):
        h = mass_weighted_partial_hessian(
            forces, pos, np.array([m]), np.array([0]), h_step
        )
        errors.append(h[0, 0] - exact)
        assert h[0, 0] - exact == pytest.approx(q * h_step**2 / m, rel=1.0e-6)
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=1.0e-6)
    assert errors[1] / errors[2] == pytest.approx(4.0, rel=1.0e-6)


def _morse_forces_and_hessian(
    d_e: float, a: float, r0: float
) -> tuple[ForcesFn, Callable[[np.ndarray], np.ndarray]]:
    """Return all-pairs Morse forces and the analytic full ``(3N, 3N)`` Hessian."""

    def phi_d1(r: float) -> float:
        e = np.exp(-a * (r - r0))
        return 2.0 * d_e * a * e * (1.0 - e)

    def phi_d2(r: float) -> float:
        e = np.exp(-a * (r - r0))
        return 2.0 * d_e * a * a * e * (2.0 * e - 1.0)

    def forces(pos: np.ndarray) -> np.ndarray:
        n = pos.shape[0]
        f = np.zeros_like(pos)
        for i in range(n):
            for j in range(i + 1, n):
                rv = pos[i] - pos[j]
                r = float(np.linalg.norm(rv))
                g = phi_d1(r) * rv / r
                f[i] -= g
                f[j] += g
        return f

    def hessian(pos: np.ndarray) -> np.ndarray:
        n = pos.shape[0]
        hess = np.zeros((3 * n, 3 * n))
        for i in range(n):
            for j in range(i + 1, n):
                rv = pos[i] - pos[j]
                r = float(np.linalg.norm(rv))
                u = rv / r
                block = (phi_d2(r) - phi_d1(r) / r) * np.outer(u, u) + (
                    phi_d1(r) / r
                ) * np.eye(3)
                si, sj = slice(3 * i, 3 * i + 3), slice(3 * j, 3 * j + 3)
                hess[si, si] += block
                hess[sj, sj] += block
                hess[si, sj] -= block
                hess[sj, si] -= block
        return hess

    return forces, hessian


def test_morse_trimer_partial_hessian_matches_analytic_free_block() -> None:
    """Off-diagonal coupling, unequal masses and a frozen atom absent from the result."""
    forces, full_hessian = _morse_forces_and_hessian(d_e=0.7, a=1.4, r0=2.5)
    pos = np.array([[0.0, 0.0, 0.0], [2.6, 0.3, 0.0], [1.1, 2.4, 0.2]])
    m = np.array([1.0, 3.0, 12.0])
    free = np.array([0, 2])  # atom 1 frozen
    h_fd = mass_weighted_partial_hessian(forces, pos, m, free, 1.0e-3)
    assert h_fd.shape == (6, 6)
    np.testing.assert_array_equal(h_fd, h_fd.T)

    full = full_hessian(pos)
    dof = np.concatenate([np.arange(3 * i, 3 * i + 3) for i in free])
    block = full[np.ix_(dof, dof)]
    inv_sqrt_m = np.repeat(1.0 / np.sqrt(m[free]), 3)
    expected = block * np.outer(inv_sqrt_m, inv_sqrt_m)
    np.testing.assert_allclose(h_fd, expected, atol=1.0e-5)
    # Truncation error is O(h^2): halving the step divides the error by ~4.
    err_2h = np.max(
        np.abs(mass_weighted_partial_hessian(forces, pos, m, free, 2e-3) - expected)
    )
    err_h = np.max(np.abs(h_fd - expected))
    assert err_2h / err_h == pytest.approx(4.0, rel=0.05)
    # The frozen atom's coupling rows are not in the partial Hessian.
    assert not np.allclose(full[:6, :6] * np.outer(inv_sqrt_m, inv_sqrt_m), h_fd)


def test_fd_hessian_fn_binds_masses_and_step() -> None:
    """``fd_hessian_fn`` reproduces the direct kernel call."""
    k = np.array([1.0, 2.0])
    m = np.array([2.0, 5.0])
    eq = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    fn = fd_hessian_fn(_diagonal_springs(k, eq), m, 1.0e-3)
    free = np.array([1])
    direct = mass_weighted_partial_hessian(
        _diagonal_springs(k, eq), eq, m, free, 1.0e-3
    )
    np.testing.assert_array_equal(fn(eq, free), direct)
    np.testing.assert_allclose(fn(eq, free), 0.4 * np.eye(3), atol=1.0e-9)


def test_invalid_inputs_raise() -> None:
    """Empty selection, bad step, bad masses and wrong force shapes are plumbing errors."""
    k = np.array([1.0, 2.0])
    eq = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    forces = _diagonal_springs(k, eq)
    m = np.array([1.0, 1.0])
    with pytest.raises(ValueError):
        mass_weighted_partial_hessian(forces, eq, m, np.array([], dtype=int), 1.0e-3)
    with pytest.raises(ValueError):
        mass_weighted_partial_hessian(forces, eq, m, np.array([0]), 0.0)
    with pytest.raises(ValueError):
        mass_weighted_partial_hessian(forces, eq, m, np.array([0, 0]), 1.0e-3)
    with pytest.raises(ValueError):
        mass_weighted_partial_hessian(forces, eq, m, np.array([5]), 1.0e-3)
    with pytest.raises(ValueError):
        mass_weighted_partial_hessian(
            forces, eq, np.array([0.0, 1.0]), np.array([0]), 1e-3
        )
    with pytest.raises(HTSTRequestError):
        mass_weighted_partial_hessian(forces, eq, np.array([1.0]), np.array([0]), 1e-3)

    def bad_forces(p: np.ndarray) -> np.ndarray:
        return np.zeros(3)

    with pytest.raises(ValueError):
        mass_weighted_partial_hessian(bad_forces, eq, m, np.array([0]), 1.0e-3)
