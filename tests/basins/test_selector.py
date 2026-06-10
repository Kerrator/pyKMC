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
