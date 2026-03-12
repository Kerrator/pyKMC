import numpy as np
import numpy.testing as npt
from pykmc.basins import solve_master_equation, BisectionSolver, QSDSolver
from pykmc.basins.selection import FPTASelector

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

    def test_qsd_solver_stiff_system(self, test_logger):
        """Test QSD solver on a stiff system where bisection fails."""

        # 3 transient states + 1 absorbing state
        # Transient rates ~1e-2, absorbing rates ~1e-18 => stiffness ~1e16
        M = np.array([
            [+2.00e-02, -1.00e-02, -1.00e-02, 0.0],
            [-1.00e-02, +2.00e-02, -1.00e-02, 0.0],
            [-1.00e-02, -1.00e-02, +2.00e-02, 0.0],
            [-1.00e-18, -1.00e-18, -1.00e-18, 0.0],
        ])

        p0 = np.array([1, 0, 0, 0])
        r = 0.5

        solver = QSDSolver(M=M, p0=p0, r=r)
        result = solver.solve()

        assert result.is_ok(), f"QSD solver failed: {result.err_value()}"
        t_exit = result.ok_value().t_exit
        test_logger.debug("QSD t_exit = %.6e", t_exit)

        assert t_exit > 0, "t_exit should be positive"
        assert np.isfinite(t_exit), "t_exit should be finite"

        # QSD should sum to 1
        assert solver.qsd is not None
        npt.assert_allclose(np.sum(solver.qsd), 1.0, atol=1e-10)

        # k_eff should be positive and small
        assert solver.k_eff > 0
        test_logger.debug("QSD k_eff = %.6e, qsd = %s", solver.k_eff, solver.qsd)

    def test_qsd_matches_bisection_moderate(self, test_logger):
        """On a moderately stiff system, QSD and bisection should agree."""

        # Stiffness ~1e3 — both solvers should work
        M = np.array([
            [+1.89645002e-02, -9.48225009e-03, -9.48225009e-03, 0.0],
            [-9.48225009e-03, +1.89645002e-02, -9.48225009e-03, 0.0],
            [-9.48225009e-03, -9.48225009e-03, +1.89645002e-02, 0.0],
            [-2.83934789e-10, -2.83934789e-10, -2.83934789e-10, 0.0],
        ])

        p0 = np.array([1, 0, 0, 0])
        r = 0.5

        bisection = BisectionSolver(M=M, p0=p0, r=r, spectral_decomposition=True)
        res_bisection = bisection.solve()
        assert res_bisection.is_ok()
        t_bisection = res_bisection.ok_value().t_exit

        qsd = QSDSolver(M=M, p0=p0, r=r)
        res_qsd = qsd.solve()
        assert res_qsd.is_ok()
        t_qsd = res_qsd.ok_value().t_exit

        test_logger.debug("Bisection t_exit=%.6e, QSD t_exit=%.6e", t_bisection, t_qsd)

        # QSD assumes instant equilibration so agreement is approximate
        # For moderate stiffness (~1e7) they should be within ~50%
        npt.assert_allclose(t_qsd, t_bisection, rtol=0.5)

    def test_stiffness_detection(self, test_logger):
        """Verify FPTASelector dispatches to QSD when stiffness > threshold."""
        from pykmc.basins.connectivity import BasinStatesConnectivity

        connectivity = BasinStatesConnectivity()
        # 2 transient states (0, 1) + 1 absorbing state (2)
        # Transient rate ~1e-2, absorbing rate ~1e-18 => stiffness ~1e16
        connectivity.add_connectivity(state=0, state_connexion=1, event_connexion=0,
                                      central_atom=0, sym=0, transient=True,
                                      dE_forward=0.1, k_forward=1e-2,
                                      dE_backward=0.1, k_backward=1e-2)
        connectivity.add_connectivity(state=1, state_connexion=0, event_connexion=1,
                                      central_atom=0, sym=0, transient=True,
                                      dE_forward=0.1, k_forward=1e-2,
                                      dE_backward=0.1, k_backward=1e-2)
        connectivity.add_connectivity(state=0, state_connexion=2, event_connexion=2,
                                      central_atom=0, sym=0, transient=False,
                                      dE_forward=2.0, k_forward=1e-18,
                                      dE_backward=0.1, k_backward=1e-2)

        selector = FPTASelector()
        result = selector.select_from_connectivity(connectivity)

        assert result.is_ok(), f"Selector failed: {result.err_value()}"
        assert selector._use_qsd, "Should have used QSD for stiff system"
        test_logger.debug("Stiffness detection: exit_state=%d, t_exit=%.6e",
                          result.ok_value().exit_state, result.ok_value().t_exit)