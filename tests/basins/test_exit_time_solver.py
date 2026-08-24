import logging
import numpy as np
import numpy.testing as npt
from pykmc.basins import solve_master_equation, BisectionSolver, QSDSolver

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
