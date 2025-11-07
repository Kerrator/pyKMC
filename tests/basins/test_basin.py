import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pykmc.basins import  StatesConnectivity, BasinStatesConnectivity, BasinGenericEventExplorer, BasinsGenericEvents, FTPASelector, BisectionSolver, solve_master_equation
import logging
from pykmc.enginemanager.lmpi.pool import ManagerFactory
import numpy.testing as npt










class TestSelector : 

    def test_ftpa_bisection(self, test_logger) : 

        #Mock reduced matric 
        M_abs_reduced = np.array([
            [0.8, -0.3, -0.2, -0.3], 
            [-0.05, 0.6, -0.25, -0.3], 
            [-0.1, -0.1, 0.5, -0.3],
            [-0.02, -0.18, -0.2, 0.4]]) 
        
        p0 = np.array([1,0,0,0])
        t0 = 1.0/np.diag(M_abs_reduced)[-1]
        r = 0.5

        test_logger.debug("Solve Master Equation for : ")
        test_logger.debug("M = \n {}".format(M_abs_reduced))
        test_logger.debug("p0 = \n {}".format(p0))
        test_logger.debug("t = {}".format(t0))
        p = solve_master_equation(M_abs_reduced, t0, p0, True)
        test_logger.debug("found p = \n {}".format(p))

        #computed with gnu octave
        res_expected = np.array([0.172586, 0.054168, 0.071085, 0.042176])
        test_logger.debug("Expected p = \n {}".format(res_expected))

        npt.assert_allclose(p, res_expected, rtol=1e-4)
         
        


class TestBasin : 

    #@mpi_test(nproc=8)
    def test_connectivity_table_construction(self, test_logger, config_Cu, reference_table_Cu_fake, system_Cu, visited_environments_Cu) : 

        factory = ManagerFactory(n_sessions=config_Cu.control.n_sessions, use_rank_0=True)
        manager = factory.launch()
        if manager is not None:
            manager.initialize_sessions(config_Cu, system_Cu)
            f = manager.minimize_with_results(config=config_Cu)
            p, e = f.result()
            print(e)
            self.basin = BasinsGenericEvents(config=config_Cu, reference_table=reference_table_Cu_fake, known_environments=visited_environments_Cu, manager = None)
            self.basin.manager = manager

            self.basin._initialize(system=system_Cu)
            self.basin.construct_connexion_table()
            mapping = self.basin.connectivity_table.reorder_states_index()
            self.basin.states = {mapping[old]: val for old, val in self.basin.states.items()}
            print(self.basin.states)
            print("END")
            print(self.basin.connectivity_table.get_table())
            self.basin.connectivity_table.save()
            exit_state = self.basin.selector.select_from_connectivity(self.basin.connectivity_table)
            manager.close_all()
