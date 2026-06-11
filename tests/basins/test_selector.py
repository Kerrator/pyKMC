from pykmc.basins import FPTASelector
import numpy as np
import pandas as pd

class TestSelector :

    def test_solver_dispatch(self, test_logger) :
        """`solver` selects bisection vs QSD; 'auto' switches on stiffness > 1e6."""
        # Stiff reduced generator: transient mixing 1e3, absorbing escape 1e-3.
        stiff = np.array([
            [1.0e3 + 1.0e-3, -1.0e3,          0.0],
            [-1.0e3,          1.0e3 + 1.0e-3, 0.0],
            [-1.0e-3,        -1.0e-3,         0.0],
        ])
        # Non-stiff: comparable transient and absorbing rates.
        mild = np.array([
            [0.3, -0.1, 0.0],
            [-0.1, 0.3, 0.0],
            [-0.2, -0.2, 0.0],
        ])

        # auto -> QSD on the stiff matrix, bisection on the mild one
        sel = FPTASelector(solver="auto"); sel.M_abs_reduced = stiff
        assert sel.get_exit_time().is_ok() and sel._use_qsd is True
        sel = FPTASelector(solver="auto"); sel.M_abs_reduced = mild
        assert sel.get_exit_time().is_ok() and sel._use_qsd is False

        # forced modes ignore stiffness
        sel = FPTASelector(solver="bisection"); sel.M_abs_reduced = stiff
        sel.get_exit_time()
        assert sel._use_qsd is False
        sel = FPTASelector(solver="qsd"); sel.M_abs_reduced = mild
        assert sel.get_exit_time().is_ok() and sel._use_qsd is True


    def test_ftpa(self, test_logger, connectivity_table_Cu) : 

        test_logger.debug("FTPA selector for Copper fake") 
        #Get fake connectivity table (Cu 1 sia 1 vac, remove transition sia event)
        connectivity_table = connectivity_table_Cu

        selector = FPTASelector() 
        result = selector.select_from_connectivity(connectivity_table) 
        
        test_logger.debug("For connectivity table : \n {}".format(connectivity_table.df))
        test_logger.debug("FTPASelector build Generator matrix : \n {}".format(selector.M_abs))
        test_logger.debug("And reduced matrix : \n {}".format(selector.M_abs_reduced))
        test_logger.debug("Got exit time = {} and exit state = {}".format(result.ok_value().t_exit, result.ok_value().exit_state))

    def test_auto_fallback_bisection_to_qsd(self, monkeypatch) :
        """solver='auto': when bisection fails below the stiffness heuristic, the
        QSD solver is tried as a backstop; forced 'bisection' stays strict."""
        from pykmc.result import Err, ErrorInfo, ErrorType
        from pykmc.basins import selection as selection_mod

        mild = np.array([
            [0.3, -0.1, 0.0],
            [-0.1, 0.3, 0.0],
            [-0.2, -0.2, 0.0],
        ])
        failing = Err(ErrorInfo(type=ErrorType.BASIN_TEXIT_NOT_FOUND, message="boom"))

        class _FailingBisection:
            def __init__(self, *a, **k):
                pass
            def solve(self):
                return failing

        monkeypatch.setattr(selection_mod, "BisectionSolver", _FailingBisection)

        sel = FPTASelector(solver="auto"); sel.M_abs_reduced = mild
        result = sel.get_exit_time()
        assert result.is_ok()
        assert sel._use_qsd is True

        sel = FPTASelector(solver="bisection"); sel.M_abs_reduced = mild
        result = sel.get_exit_time()
        assert not result.is_ok()

    def test_excluded_states_never_selected(self, connectivity_table_Cu) :
        """Excluded absorbing states get zero probability; the draw avoids them."""
        selector = FPTASelector()
        result = selector.select_from_connectivity(connectivity_table_Cu)
        assert result.is_ok()
        t_exit = result.ok_value().t_exit
        n_transient = len(selector.M_abs_reduced) - 1
        n_states = len(selector.M_abs)
        absorbing = set(range(n_transient, n_states))

        # exclude one absorbing state: 200 draws must never land on it
        excluded = {n_transient}
        for _ in range(200):
            choice = selector.select_absorbing_state(t_exit, excluded_states=excluded)
            assert choice is not None
            assert choice not in excluded
            assert choice in absorbing

    def test_all_excluded_returns_none_and_err(self, connectivity_table_Cu) :
        """With every absorbing exit excluded the draw returns None and
        select_from_connectivity maps it to Err(BASIN_NO_VIABLE_EXIT)."""
        from pykmc.result import ErrorType

        selector = FPTASelector()
        result = selector.select_from_connectivity(connectivity_table_Cu)
        assert result.is_ok()
        t_exit = result.ok_value().t_exit
        n_transient = len(selector.M_abs_reduced) - 1
        all_absorbing = set(range(n_transient, len(selector.M_abs)))

        assert selector.select_absorbing_state(t_exit, excluded_states=all_absorbing) is None

        result = selector.select_from_connectivity(connectivity_table_Cu, excluded_states=all_absorbing)
        assert not result.is_ok()
        assert result.err_value().type == ErrorType.BASIN_NO_VIABLE_EXIT
