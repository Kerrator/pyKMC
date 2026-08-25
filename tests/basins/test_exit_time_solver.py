import logging
import numpy as np
import numpy.testing as npt
from fractions import Fraction

from pykmc.basins import solve_master_equation, BisectionSolver, QSDSolver


def _exact_mfpt_fraction(M: np.ndarray) -> float :
    """Exact mean first-passage time of the chain defined by ``M``'s off-diagonals.

    ``M`` is a reduced generator: ``M[j, i] = -k(i->j)`` off the diagonal,
    ``M[-1, i] = -gamma_i`` the escape rate of transient state ``i``. Only those
    off-diagonals are data; the stored diagonal is a derived float sum that can
    round. The reference therefore rebuilds ``d_i = gamma_i + sum_j k(i->j)`` and
    solves ``Q tau = e_0`` in exact rational arithmetic (``fractions.Fraction``),
    which is what the solver under test is required to reproduce.
    """
    n = len(M) - 1
    k = [[Fraction(0)] * n for _ in range(n)]
    for i in range(n) :
        for j in range(n) :
            if i != j :
                k[i][j] = Fraction(float(-M[j][i]))
    gamma = [Fraction(float(-M[-1][i])) for i in range(n)]
    d = [gamma[i] + sum(k[i][j] for j in range(n) if j != i) for i in range(n)]
    Q = [[(d[i] if i == j else -k[j][i]) for j in range(n)] for i in range(n)]
    b = [Fraction(0)] * n
    b[0] = Fraction(1)
    for c in range(n) :
        piv = max(range(c, n), key=lambda rr: abs(Q[rr][c]))
        Q[c], Q[piv] = Q[piv], Q[c]
        b[c], b[piv] = b[piv], b[c]
        for rr in range(c + 1, n) :
            if Q[rr][c] == 0 :
                continue
            f = Q[rr][c] / Q[c][c]
            for cc in range(c, n) :
                Q[rr][cc] -= f * Q[c][cc]
            b[rr] -= f * b[c]
    x = [Fraction(0)] * n
    for rr in range(n - 1, -1, -1) :
        x[rr] = (b[rr] - sum(Q[rr][cc] * x[cc] for cc in range(rr + 1, n))) / Q[rr][rr]
    return float(sum(x))


def _closed_chain_keff(M: np.ndarray) -> float :
    """Pre-MFPT stiff-limit asymptote: k_eff of the closed-chain stationary form.

    Kept in the tests only, as the reference the hybrid solver had to beat
    (git show af1a40d:pykmc/basins/exit_time_solver.py).
    """
    n = len(M) - 1
    gamma = -M[-1, :n]
    q_tt = M[:n, :n].copy()
    for i in range(n) :
        q_tt[i, i] -= gamma[i]
    _, _, vh = np.linalg.svd(q_tt)
    null_vec = np.abs(np.real(vh[-1, :]))
    return float((null_vec / np.sum(null_vec)) @ gamma)



class TestSolver :

    def test_solve_master_equation(self, test_logger):

        #Mock reduced matric 
        M_abs_reduced = np.array([
            [+0.17 ,  -0.30 , - 0.20 ,  0.00],
            [- 0.05 , 0.48 ,  -0.25 ,  0.00],
            [- 0.10 ,  -0.10 , 0.45 ,  0.00],
            [- 0.02 ,  -0.08 ,  -0.00 ,  0.00]])
        
        p0 = np.array([1,0,0,0])
        t0 = 1.0/np.sum(np.diag(M_abs_reduced))

        test_logger.debug("Solve Master Equation for : ")
        test_logger.debug("M = \n {}".format(M_abs_reduced))
        test_logger.debug("p0 = \n {}".format(p0))
        test_logger.debug("t = {}".format(t0))
        p = solve_master_equation(M_abs_reduced, t0, p0, False)
        test_logger.debug("found p = \n {}".format(p))

        #computed with gnu octave
        res_expected = np.array([0.869014, 0.041671, 0.070828, 0.018488])
        test_logger.debug("Expected p = \n {}".format(res_expected))

        npt.assert_allclose(p, res_expected, rtol=1e-4)
        
        test_logger.debug("Solve Master Equation Using Sprectral Decomposition for : ")
        test_logger.debug("M = \n {}".format(M_abs_reduced))
        test_logger.debug("p0 = \n {}".format(p0))
        test_logger.debug("t = {}".format(t0))
        p = solve_master_equation(M_abs_reduced, t0, p0, True)
        test_logger.debug("found p = \n {}".format(p))

        test_logger.debug("Expected p = \n {}".format(res_expected))

        npt.assert_allclose(p, res_expected, rtol=1e-4)

    def test_find_texit(self, test_logger) : 
        
        M_abs_reduced = np.array([[ 1.89645002e-02,-9.48225009e-03,-9.48225009e-03, 0.00000000e+00],
 [-9.48225009e-03, 1.89645002e-02,-9.48225009e-03, 0.00000000e+00],
 [-9.48225009e-03,-9.48225009e-03, 1.89645002e-02, 0.00000000e+00],
 [-2.83934789e-10,-2.83934789e-10,-2.83934789e-10, 0.00000000e+00]])
        
        p0 = np.array([1,0,0,0])
        r = 0.9

        solver = BisectionSolver(M=M_abs_reduced, p0=p0, r=r, spectral_decomposition=True) 

        test_logger.debug("Find t_exit for r = {}".format(r))
        test_logger.debug("With M = \n {}".format(M_abs_reduced))
        test_logger.debug("And p0 = \n {}".format(p0))

        res = solver.solve()

        if res.is_ok() :
            t_exit = res.ok_value().t_exit
            test_logger.debug("Find t_exit = {}ps".format(t_exit))
        else :
            err = res.err_value()
            test_logger.debug("Err while searching t_exit : {}".format(err))

    def test_qsd_solver_two_state(self, test_logger) :
        """QSD solver: closed-form exit time for a stiff two-transient generator.

        Two transient states with fast mutual mixing rate w and small, equal escape
        rate g. The quasi-stationary distribution is uniform (pi = [0.5, 0.5]), so
        k_eff = g and t_exit = -ln(1 - r) / g.
        """
        w = 1.0e3   # fast transient mixing
        g = 1.0e-3  # slow absorbing escape
        # Reduced generator (columns ~ rates out): 2 transient + merged absorbing.
        # Diagonal = total out-rate; off-diagonal transient = -mixing; last row = -escape.
        M = np.array([
            [w + g,   -w,      0.0],
            [-w,       w + g,  0.0],
            [-g,      -g,      0.0],
        ])
        r = 0.9
        solver = QSDSolver(M=M, p0=np.array([1.0, 0.0, 0.0]), r=r)
        res = solver.solve()
        assert res.is_ok()
        # occupation-time weights carry the O(g/w) = 1e-6 bias toward the entry state
        npt.assert_allclose(solver.qsd, [0.5, 0.5], atol=1e-5)
        npt.assert_allclose(solver.k_eff, g, rtol=1e-6)
        npt.assert_allclose(res.ok_value().t_exit, -np.log(1.0 - r) / g, rtol=1e-6)

    def test_qsd_solver_no_escape_errors(self, test_logger) :
        """QSD solver returns Err when no absorbing escape is possible (k_eff <= 0)."""
        M = np.array([
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0,  0.0, 0.0],   # zero escape rates
        ])
        res = QSDSolver(M=M, p0=np.array([1.0, 0.0, 0.0]), r=0.9).solve()
        assert not res.is_ok()
    def test_qsd_solver_slow_internal_passage(self, test_logger: logging.Logger) -> None :
        """QSD (MFPT) solver on a stiff basin whose escape sits behind a slow internal step.

        Fast 0<->1 mixing (w), slow 1<->2 passage (s), escape only from state 2 (g), entry in 0.
        The stiffness heuristic (max transient / max absorbing rate = w/g = 1e7) routes
        solver='auto' here. The old closed-chain stationary distribution was uniform and
        predicted k_eff = g/3, i.e. an exit time ~1e3x too short; the MFPT-based solver
        must match the exact expm quantile to well within the exponential-shape error.
        """
        from scipy.linalg import expm
        from scipy.optimize import brentq
        w, s, g = 1.0e4, 1.0e-6, 1.0e-3
        M = np.array([
            [w,    -w,     0.0,    0.0],
            [-w,   w + s,  -s,     0.0],
            [0.0,  -s,     s + g,  0.0],
            [0.0,  0.0,    -g,     0.0],
        ])
        p0 = np.array([1.0, 0.0, 0.0, 0.0])
        r = 0.9
        Q = M[:3, :3]
        e0 = p0[:3]
        mfpt = float(np.sum(np.linalg.solve(Q, e0)))
        p_abs = lambda t: 1.0 - float(np.sum(expm(-Q * t) @ e0))  # noqa: E731
        t_exact = brentq(lambda t: p_abs(t) - r, 1e-3, 1e3 * mfpt)

        solver = QSDSolver(M=M, p0=p0, r=r)
        res = solver.solve()
        assert res.is_ok()
        npt.assert_allclose(1.0 / solver.k_eff, mfpt, rtol=1e-9)
        npt.assert_allclose(res.ok_value().t_exit, t_exact, rtol=0.05)
        # occupation-time weights sum to one and are non-negative
        npt.assert_allclose(np.sum(solver.qsd), 1.0, rtol=1e-12)
        assert np.all(solver.qsd >= 0)

    def test_bisection_expm_path_matches_spectral(self, test_logger: logging.Logger) -> None :
        """spectral_decomposition=False (scipy expm) must run and agree with the spectral path."""
        M_abs_reduced = np.array([
            [+0.17, -0.30, -0.20, 0.00],
            [-0.05, +0.48, -0.25, 0.00],
            [-0.10, -0.10, +0.45, 0.00],
            [-0.02, -0.08, -0.00, 0.00]])
        p0 = np.array([1.0, 0.0, 0.0, 0.0])
        r = 0.7
        res_spec = BisectionSolver(M=M_abs_reduced, p0=p0, r=r, spectral_decomposition=True, tolerance=1e-6).solve()
        res_expm = BisectionSolver(M=M_abs_reduced, p0=p0, r=r, spectral_decomposition=False, tolerance=1e-6).solve()
        assert res_spec.is_ok() and res_expm.is_ok()
        npt.assert_allclose(res_expm.ok_value().t_exit, res_spec.ok_value().t_exit, rtol=1e-4)

    def test_qsd_solver_slow_internal_passage_uses_direct_solve(self, test_logger: logging.Logger) -> None :
        """The regime switch is on conditioning, not stiffness.

        Same generator as ``test_qsd_solver_slow_internal_passage`` (stiffness
        w/g = 1e7): the direct ``np.linalg.solve`` MFPT is accurate there
        (relative residual ~6e-7, below the 1e-5 tolerance), so the solver must
        stay on the direct path and reproduce ``np.linalg.solve`` bit for bit.
        """
        w, s, g = 1.0e4, 1.0e-6, 1.0e-3
        M = np.array([
            [w,    -w,     0.0,    0.0],
            [-w,   w + s,  -s,     0.0],
            [0.0,  -s,     s + g,  0.0],
            [0.0,  0.0,    -g,     0.0],
        ])
        solver = QSDSolver(M=M, p0=np.array([1.0, 0.0, 0.0, 0.0]), r=0.9)
        assert solver.solve().is_ok()
        assert solver.method == "mfpt-direct"
        assert solver.residual < 1e-5

    def test_qsd_solver_two_state_stiffness_sweep(self, test_logger: logging.Logger) -> None :
        """k_eff must stay within 1e-3 of exact for w/g from 1e6 to 1e20.

        Two-state chain 0<->1 at w with escape g from state 1 only; the exact
        mean first-passage time from state 0 is 2/g + 1/w. At c4b8aea the direct
        ``np.linalg.solve`` MFPT drifted from 1e13 (0.9995) through 1.95 at 1e16
        and returned BASIN_TEXIT_NOT_FOUND from 1e17 on, because Q's diagonal
        w + g rounds to w once g/w < eps.
        """
        g = 1.0e-3
        worst = 0.0
        for exponent in range(6, 21) :
            w = g * 10.0 ** exponent
            M = np.array([[w, -w, 0.0], [-w, w + g, 0.0], [0.0, -g, 0.0]])
            solver = QSDSolver(M=M, p0=np.array([1.0, 0.0, 0.0]), r=0.5)
            res = solver.solve()
            assert res.is_ok(), "w/g=1e{} failed: {}".format(exponent, res.err_value())
            exact = 1.0 / (2.0 / g + 1.0 / w)
            ratio = solver.k_eff / exact
            worst = max(worst, abs(ratio - 1.0))
            test_logger.debug("w/g=1e%d k_eff/exact=%.12f method=%s", exponent, ratio, solver.method)
            assert abs(ratio - 1.0) < 1.0e-3, "w/g=1e{}: k_eff/exact={}".format(exponent, ratio)
        test_logger.debug("worst |k_eff/exact - 1| over the sweep: %.3e", worst)

    def test_qsd_solver_slow_passage_extreme_stiffness(self, test_logger: logging.Logger) -> None :
        """Separating test: slow internal passage at stiffness 1e20.

        Fast 0<->1 at w = 1e17 g, slow 1<->2 at s = 1e-3 g, escape g from state 2.
        The direct MFPT solve is dead here (Q's 0/1 block is numerically singular)
        and the closed-chain stationary form - the stiff-limit asymptote of the
        pre-MFPT solver - is ~2e3x wrong because it assumes every transient state
        mixes faster than any escape. The cancellation-free GTH reduction must
        match the exact rational MFPT of the same rates.
        """
        w, s, g = 1.0e14, 1.0e-6, 1.0e-3
        M = np.array([
            [w,    -w,     0.0,    0.0],
            [-w,   w + s,  -s,     0.0],
            [0.0,  -s,     s + g,  0.0],
            [0.0,  0.0,    -g,     0.0],
        ])
        p0 = np.array([1.0, 0.0, 0.0, 0.0])

        exact_mfpt = _exact_mfpt_fraction(M)
        # the direct solve is unusable at this conditioning
        Q = M[:3, :3]
        direct = np.linalg.solve(Q, p0[:3])
        assert not (np.all(direct > 0) and abs(float(np.sum(direct)) / exact_mfpt - 1.0) < 1.0e-3)
        # so is the closed-chain stationary asymptote
        assert abs(_closed_chain_keff(M) * exact_mfpt - 1.0) > 1.0e2

        solver = QSDSolver(M=M, p0=p0, r=0.5)
        res = solver.solve()
        assert res.is_ok(), res.err_value() if not res.is_ok() else ""
        assert solver.method == "mfpt-gth"
        npt.assert_allclose(1.0 / solver.k_eff, exact_mfpt, rtol=1e-9)
        npt.assert_allclose(np.sum(solver.qsd), 1.0, rtol=1e-12)
        assert np.all(solver.qsd >= 0)
