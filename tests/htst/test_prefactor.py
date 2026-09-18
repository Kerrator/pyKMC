"""Orchestrator tests: directional independence, one saddle Hessian, typed rejections."""

from __future__ import annotations

import math
import pickle
from typing import Any

import numpy as np
import pytest

from pykmc.htst import (
    HTSTEventRequest,
    HTSTGeometryError,
    HTSTRequestError,
    HTSTSettings,
    PrefactorRejected,
    PrefactorRejection,
    compute_event_prefactors,
    fd_hessian_fn,
)

EV_J = 1.602176634e-19
AMU_KG = 1.66053906660e-27
WIDE = HTSTSettings(nu0_min_hz=1.0e6, nu0_max_hz=1.0e20, free_radius=2.0)


def nu_hz(lam: float) -> float:
    """Closed-form linear frequency in Hz of a mass-weighted eigenvalue."""
    return math.sqrt(lam * EV_J / (AMU_KG * 1.0e-20)) / (2.0 * math.pi)


def toy_hessian(eigvals: np.ndarray, seed: int) -> np.ndarray:
    """Return a symmetric matrix with the given eigenvalues and random eigenvectors."""
    n = len(eigvals)
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    h = q @ np.diag(eigvals) @ q.T
    return 0.5 * (h + h.T)


def make_request(
    settings: HTSTSettings = WIDE, n_atoms: int = 2, **overrides: Any
) -> HTSTEventRequest:
    """Two Ni atoms 5 Å apart; only the centre is inside ``free_radius=2``."""
    base = np.array([[5.0, 5.0, 5.0], [10.0, 5.0, 5.0], [5.0, 10.0, 5.0]])[:n_atoms]
    fields: dict[str, Any] = {
        "event_key": ("ref", 42),
        "min1_positions": base.copy(),
        "saddle_positions": base + np.array([0.5, 0.0, 0.0]),
        "min2_positions": base + np.array([1.0, 0.0, 0.0]),
        "types": ("Ni",) * n_atoms,
        "species": ("Ni", "Fe"),
        "masses": (58.6934, 55.845),
        "cell": np.diag([20.0, 20.0, 20.0]),
        "pbc": (True, True, True),
        "center_index": 0,
        "settings": settings,
    }
    fields.update(overrides)
    return HTSTEventRequest(**fields)


class SpectrumHessian:
    """Hessian callable keyed by geometry, counting calls per geometry."""

    def __init__(
        self,
        request: HTSTEventRequest,
        min1: np.ndarray,
        saddle: np.ndarray,
        min2: np.ndarray,
    ) -> None:
        self._table = [
            (np.asarray(request.min1_positions, dtype=float), "min1", min1),
            (np.asarray(request.saddle_positions, dtype=float), "saddle", saddle),
            (np.asarray(request.min2_positions, dtype=float), "min2", min2),
        ]
        self.calls: dict[str, int] = {"min1": 0, "saddle": 0, "min2": 0}
        self.free_seen: list[np.ndarray] = []

    def __call__(self, positions: np.ndarray, free_indices: np.ndarray) -> np.ndarray:
        """Return the matrix registered for ``positions``."""
        self.free_seen.append(np.asarray(free_indices))
        for ref, label, matrix in self._table:
            if np.array_equal(positions, ref):
                self.calls[label] += 1
                return matrix
        raise AssertionError("unexpected geometry passed to hessian_fn")


def three_mode_spectra(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return valid min1, saddle, min2 Hessians for one free atom (3 DOF)."""
    return (
        toy_hessian(np.array([0.3, 0.5, 0.7]), seed + 1),
        toy_hessian(np.array([-0.2, 0.4, 0.6]), seed + 2),
        toy_hessian(np.array([0.35, 0.45, 0.8]), seed + 3),
    )


def test_double_well_fd_oracle_both_directions() -> None:
    """One free atom in a quartic double well: ``nu0 = nu(2a/m)`` in both directions."""
    a, b, kappa = 0.8, 0.5, 1.3  # eV/Å^2, eV/Å^4, eV/Å^2
    x0 = math.sqrt(a / b)
    centre = np.array([5.0, 5.0, 5.0])
    far = np.array([15.0, 5.0, 5.0])  # outside free_radius=2, frozen
    min1 = np.array([centre - [x0, 0, 0], far])
    sad = np.array([centre, far])
    min2 = np.array([centre + [x0, 0, 0], far])

    def forces(pos: np.ndarray) -> np.ndarray:
        d = pos - np.array([centre, far])
        f = -kappa * d
        x = d[:, 0]
        f[:, 0] = a * x - b * x**3
        return f

    m_ni = 58.6934
    settings = HTSTSettings(
        free_radius=2.0, fd_step=1.0e-3, nu0_min_hz=1e6, nu0_max_hz=1e20
    )
    req = make_request(
        settings, min1_positions=min1, saddle_positions=sad, min2_positions=min2
    )
    res = compute_event_prefactors(
        req, fd_hessian_fn(forces, req.masses_per_atom(), 1e-3)
    )
    assert res.n_free == 1
    assert res.forward.ok and res.backward.ok
    assert res.forward.n_negative_saddle == 1 and res.backward.n_negative_saddle == 1
    assert res.forward.n_positive_min == 3
    expected = nu_hz(2.0 * a / m_ni)
    assert res.forward.nu0_hz == pytest.approx(expected, rel=1.0e-5)
    assert res.backward.nu0_hz == pytest.approx(res.forward.nu0_hz, rel=1.0e-9)
    assert res.method == "fd" and res.event_key == ("ref", 42)
    assert res.settings == settings


def test_mixed_direction_independence_backward_unstable_minimum() -> None:
    """A bad min2 rejects only backward; forward keeps its value and diagnostics."""
    req = make_request()
    min1, sad, _ = three_mode_spectra()
    bad_min2 = toy_hessian(np.array([-0.05, 0.4, 0.6]), 9)
    hess = SpectrumHessian(req, min1, sad, bad_min2)
    res = compute_event_prefactors(req, hess, method="lammps_eskm")
    assert res.forward.ok
    assert res.forward.nu0_hz == pytest.approx(
        nu_hz(0.3) * nu_hz(0.5) * nu_hz(0.7) / (nu_hz(0.4) * nu_hz(0.6)), rel=1e-9
    )
    assert res.backward.status == "rejected"
    assert res.backward.reason_code is PrefactorRejection.UNSTABLE_MINIMUM
    assert res.backward.nu0_hz is None
    assert res.backward.reason and "unstable" in res.backward.reason
    assert res.forward.n_negative_saddle == 1 and res.backward.n_negative_saddle == 1
    assert res.backward.n_positive_min == 2
    assert res.method == "lammps_eskm"


def test_mixed_direction_independence_forward_unstable_minimum() -> None:
    """The symmetric case: a bad min1 rejects only forward."""
    req = make_request()
    _, sad, min2 = three_mode_spectra()
    bad_min1 = toy_hessian(np.array([0.0, 0.4, 0.6]), 10)  # zero mode at a minimum
    res = compute_event_prefactors(req, SpectrumHessian(req, bad_min1, sad, min2))
    assert res.forward.reason_code is PrefactorRejection.UNSTABLE_MINIMUM
    assert res.backward.ok
    assert res.backward.nu0_hz == pytest.approx(
        nu_hz(0.35) * nu_hz(0.45) * nu_hz(0.8) / (nu_hz(0.4) * nu_hz(0.6)), rel=1e-9
    )


def test_saddle_hessian_computed_exactly_once() -> None:
    """One saddle call, one call per minimum, all with the same free selection."""
    req = make_request()
    hess = SpectrumHessian(req, *three_mode_spectra())
    res = compute_event_prefactors(req, hess)
    assert hess.calls == {"min1": 1, "saddle": 1, "min2": 1}
    assert all(f.tolist() == [0] for f in hess.free_seen)
    assert res.forward.ok and res.backward.ok


@pytest.mark.parametrize(
    ("eigs", "n_neg"),
    [
        (np.array([0.2, 0.4, 0.6]), 0),
        (np.array([-0.2, -0.1, 0.6]), 2),
        (np.array([-0.2, 0.0, 0.6]), 1),  # one unstable plus a zero mode
    ],
)
def test_saddle_not_first_order_rejects_both_directions_without_minimum_hessians(
    eigs: np.ndarray, n_neg: int
) -> None:
    """A rejected saddle short-circuits: both directions rejected, no minimum Hessian."""
    req = make_request()
    min1, _, min2 = three_mode_spectra()
    hess = SpectrumHessian(req, min1, toy_hessian(eigs, 5), min2)
    res = compute_event_prefactors(req, hess)
    for d in (res.forward, res.backward):
        assert d.reason_code is PrefactorRejection.SADDLE_NOT_FIRST_ORDER
        assert d.n_negative_saddle == n_neg
        assert d.nu0_hz is None
        assert d.n_positive_min is None  # never computed
        assert d.n_free == 1
    assert hess.calls == {"min1": 0, "saddle": 1, "min2": 0}
    assert res.n_free == 1


def test_rejected_saddle_takes_precedence_over_a_bad_minimum() -> None:
    """Saddle first: a NaN min1 never masks SADDLE_NOT_FIRST_ORDER."""
    req = make_request()
    min1, _, min2 = three_mode_spectra()
    bad_min1 = min1.copy()
    bad_min1[0, 0] = np.nan
    hess = SpectrumHessian(
        req, bad_min1, toy_hessian(np.array([0.2, 0.4, 0.6]), 5), min2
    )
    res = compute_event_prefactors(req, hess)
    assert res.forward.reason_code is PrefactorRejection.SADDLE_NOT_FIRST_ORDER
    assert res.backward.reason_code is PrefactorRejection.SADDLE_NOT_FIRST_ORDER
    assert hess.calls == {"min1": 0, "saddle": 1, "min2": 0}


def test_hessian_fn_may_raise_prefactor_rejected_with_empty_detail() -> None:
    """A PrefactorRejected of any code, even with an empty detail, is a rejection."""
    req = make_request()
    min1, sad, min2 = three_mode_spectra()

    def rejecting_saddle(positions: np.ndarray, free: np.ndarray) -> np.ndarray:
        if np.array_equal(positions, np.asarray(req.saddle_positions, dtype=float)):
            raise PrefactorRejected(PrefactorRejection.NONFINITE_HESSIAN, "")
        raise AssertionError("no minimum Hessian should be requested")

    res = compute_event_prefactors(req, rejecting_saddle)
    for d in (res.forward, res.backward):
        assert d.reason_code is PrefactorRejection.NONFINITE_HESSIAN
        assert d.reason == "nonfinite_hessian"
        assert d.n_negative_saddle is None and d.n_positive_min is None

    def rejecting_min1(positions: np.ndarray, free: np.ndarray) -> np.ndarray:
        if np.array_equal(positions, np.asarray(req.min1_positions, dtype=float)):
            raise PrefactorRejected(PrefactorRejection.UNSTABLE_MINIMUM, "")
        if np.array_equal(positions, np.asarray(req.saddle_positions, dtype=float)):
            return sad
        return min2

    res = compute_event_prefactors(req, rejecting_min1)
    assert res.forward.reason_code is PrefactorRejection.UNSTABLE_MINIMUM
    assert res.forward.reason == "unstable_minimum"
    assert res.forward.n_negative_saddle == 1 and res.forward.n_positive_min is None
    assert res.backward.ok


def test_nonfinite_saddle_hessian_rejects_both_with_unknown_counts() -> None:
    """A NaN saddle Hessian is NONFINITE_HESSIAN for both; no spectrum counts exist."""
    req = make_request()
    min1, sad, min2 = three_mode_spectra()
    bad = sad.copy()
    bad[0, 0] = np.nan
    hess = SpectrumHessian(req, min1, bad, min2)
    res = compute_event_prefactors(req, hess)
    for d in (res.forward, res.backward):
        assert d.reason_code is PrefactorRejection.NONFINITE_HESSIAN
        assert d.n_negative_saddle is None
        assert d.n_positive_min is None
        assert d.n_free == 1
    assert hess.calls == {"min1": 0, "saddle": 1, "min2": 0}
    assert res.n_free == 1


def test_nonfinite_minimum_hessian_rejects_only_that_direction() -> None:
    """An inf in min1 rejects forward with the saddle count known; backward is fine."""
    req = make_request()
    min1, sad, min2 = three_mode_spectra()
    bad = min1.copy()
    bad[1, 2] = bad[2, 1] = np.inf
    res = compute_event_prefactors(req, SpectrumHessian(req, bad, sad, min2))
    assert res.forward.reason_code is PrefactorRejection.NONFINITE_HESSIAN
    assert res.forward.n_negative_saddle == 1
    assert res.forward.n_positive_min is None
    assert res.backward.ok


def test_window_endpoints_inclusive_and_just_outside_rejected() -> None:
    """Both endpoints accept; a value a hair outside either endpoint is OUT_OF_WINDOW."""
    req = make_request()
    spectra = three_mode_spectra()
    nu0 = compute_event_prefactors(req, SpectrumHessian(req, *spectra)).forward.nu0_hz
    assert nu0 is not None
    eps = 1.0e-9
    cases = [
        (nu0, nu0 * 2.0, True),  # nu0 == min endpoint
        (nu0 / 2.0, nu0, True),  # nu0 == max endpoint
        (nu0 * (1.0 + eps), nu0 * 2.0, False),  # just below min
        (nu0 / 2.0, nu0 * (1.0 - eps), False),  # just above max
    ]
    for lo, hi, accepted in cases:
        settings = HTSTSettings(free_radius=2.0, nu0_min_hz=lo, nu0_max_hz=hi)
        r = make_request(settings)
        res = compute_event_prefactors(r, SpectrumHessian(r, *spectra))
        assert res.forward.ok is accepted, (lo, hi)
        if not accepted:
            assert res.forward.reason_code is PrefactorRejection.OUT_OF_WINDOW
            assert res.forward.nu0_hz is None
            assert res.forward.reason and "Hz" in res.forward.reason
            assert res.forward.n_negative_saddle == 1
            assert res.forward.n_positive_min == 3
        else:
            assert res.forward.nu0_hz == nu0


def test_request_validation_errors_raise_not_reject() -> None:
    """Payload violations raise HTSTRequestError/HTSTGeometryError before any Hessian."""
    req = make_request(center_index=5)
    hess = SpectrumHessian(req, *three_mode_spectra())
    with pytest.raises(HTSTRequestError):
        compute_event_prefactors(req, hess)
    tric = make_request(cell=np.array([[20.0, 0, 0], [3.0, 20.0, 0], [0, 0, 20.0]]))
    with pytest.raises(HTSTGeometryError):
        compute_event_prefactors(tric, SpectrumHessian(tric, *three_mode_spectra()))
    with pytest.raises(HTSTRequestError):
        compute_event_prefactors("not a request", hess)  # type: ignore[arg-type]
    assert hess.calls == {"min1": 0, "saddle": 0, "min2": 0}


def test_non_scientific_exceptions_propagate() -> None:
    """Engine failures and plumbing errors are never turned into rejections."""
    req = make_request()

    def dead_engine(positions: np.ndarray, free: np.ndarray) -> np.ndarray:
        raise RuntimeError("engine died")

    with pytest.raises(RuntimeError, match="engine died"):
        compute_event_prefactors(req, dead_engine)

    def wrong_shape(positions: np.ndarray, free: np.ndarray) -> np.ndarray:
        return np.eye(6)

    with pytest.raises(ValueError):
        compute_event_prefactors(req, wrong_shape)

    def asymmetric(positions: np.ndarray, free: np.ndarray) -> np.ndarray:
        h = np.diag([0.1, 0.2, 0.3])
        h[0, 1] = 0.05
        return h

    with pytest.raises(ValueError):
        compute_event_prefactors(req, asymmetric)

    def complex_valued(positions: np.ndarray, free: np.ndarray) -> np.ndarray:
        return np.eye(3, dtype=complex)

    with pytest.raises(ValueError, match="complex"):
        compute_event_prefactors(req, complex_valued)
    with pytest.raises(ValueError):
        compute_event_prefactors(
            req, SpectrumHessian(req, *three_mode_spectra()), method=""
        )


def test_request_built_from_numpy_bools_and_float32_masses_is_accepted() -> None:
    """``tuple(atoms.pbc)`` and NumPy scalar masses pass validation and the kernel."""
    req = make_request(
        pbc=tuple(np.array([True, True, True])),
        masses=(np.float64(58.6934), np.float32(55.845)),
    )
    hess = SpectrumHessian(req, *three_mode_spectra())
    res = compute_event_prefactors(req, hess)
    assert res.forward.ok and res.backward.ok
    assert all(f.tolist() == [0] for f in hess.free_seen)


def test_empty_free_region_via_injected_selection() -> None:
    """A caller-supplied empty selection rejects both directions without any Hessian."""
    req = make_request()
    hess = SpectrumHessian(req, *three_mode_spectra())
    res = compute_event_prefactors(req, hess, free_indices=np.array([], dtype=int))
    for d in (res.forward, res.backward):
        assert d.reason_code is PrefactorRejection.EMPTY_FREE_REGION
        assert d.n_free == 0
        assert d.n_negative_saddle is None and d.n_positive_min is None
    assert res.n_free == 0
    assert hess.calls == {"min1": 0, "saddle": 0, "min2": 0}


def test_injected_free_selection_is_validated_and_sorted() -> None:
    """An explicit selection is sorted before use; bad selections raise ValueError."""
    req = make_request(n_atoms=3)
    seen: list[np.ndarray] = []

    def hess(positions: np.ndarray, free: np.ndarray) -> np.ndarray:
        seen.append(free)
        lam = np.array([-0.2, 0.4, 0.6, 0.3, 0.5, 0.7])
        if np.array_equal(positions, np.asarray(req.saddle_positions, dtype=float)):
            return toy_hessian(lam, 1)
        return toy_hessian(np.abs(lam), 2)

    res = compute_event_prefactors(req, hess, free_indices=[2, 0])
    assert res.n_free == 2 and res.forward.ok
    assert all(f.tolist() == [0, 2] for f in seen)
    with pytest.raises(ValueError):
        compute_event_prefactors(req, hess, free_indices=[0, 7])
    with pytest.raises(ValueError):
        compute_event_prefactors(req, hess, free_indices=[0, 0])
    with pytest.raises(ValueError):
        compute_event_prefactors(req, hess, free_indices=[[0, 1]])


def test_results_are_picklable_with_equality() -> None:
    """The typed result survives a pickle round-trip with value equality."""
    req = make_request()
    min1, sad, _ = three_mode_spectra()
    res = compute_event_prefactors(
        req,
        SpectrumHessian(req, min1, sad, toy_hessian(np.array([-0.05, 0.4, 0.6]), 9)),
    )
    back = pickle.loads(pickle.dumps(res))
    assert back == res
    assert (
        back.forward.ok
        and back.backward.reason_code is PrefactorRejection.UNSTABLE_MINIMUM
    )


def test_overflowing_prefactor_is_rejected_not_raised() -> None:
    """A min1 Hessian whose ratio to the saddle overflows rejects forward only."""
    req = make_request()
    _, sad, min2 = three_mode_spectra()
    stiff = np.diag([1.0e300, 1.0e300, 1.0e300])
    res = compute_event_prefactors(req, SpectrumHessian(req, stiff, sad, min2))
    assert res.forward.status == "rejected"
    assert res.forward.reason_code is PrefactorRejection.NONFINITE_PREFACTOR
    assert res.forward.nu0_hz is None
    assert res.forward.n_positive_min == 3
    assert res.forward.n_negative_saddle == 1
    assert res.backward.ok


def test_hessian_fn_prefactor_rejected_with_none_detail_falls_back_to_code() -> None:
    """``PrefactorRejected(code, None)`` yields the code value as the reason."""
    req = make_request()

    def reject(positions: np.ndarray, free_indices: np.ndarray) -> np.ndarray:
        raise PrefactorRejected(PrefactorRejection.NONFINITE_HESSIAN, None)

    res = compute_event_prefactors(req, reject)
    for d in (res.forward, res.backward):
        assert d.reason_code is PrefactorRejection.NONFINITE_HESSIAN
        assert d.reason == "nonfinite_hessian"
