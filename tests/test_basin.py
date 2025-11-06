import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import os
import copy
from pykmc.basins import  StatesConnectivity, BasinStatesConnectivity, BasinGenericEventExplorer, BasinsGenericEvents, FTPASelector, BisectionSolver
import logging
from pykmc.enginemanager.lmpi.pool import ManagerFactory
from .conftest import mpi_test
import numpy.testing as npt


logger = logging.getLogger("tests")


class TestStatesConnectivity : 

    def test_add_connectivity(self, test_logger) : 

        state_connectivity = StatesConnectivity()
        state_connectivity.add_connectivity(state=0, state_connexion=1, event_connexion=11, central_atom=342, sym=0, transient=True)

        test_logger.debug(state_connectivity.get_table())

    def test_get_transition_to_state(self, test_logger, mock_statesconnectivity) : 

        res = mock_statesconnectivity.get_transition_to_state(1, as_tuples = True, return_all = False)
        test_logger.debug(res)

    def test_merge_connectivity_table(self, test_logger, mock_basinstatesconnectivity) : 

        conn1 = mock_basinstatesconnectivity
        conn2 = mock_basinstatesconnectivity

        test_logger.debug("Before merging : df = {} \n".format(conn1.get_table()))
        conn1.merge(conn2)
        test_logger.debug("After merging : df = {}\n".format(conn1.get_table()))

    def test_update_state_index(self, test_logger, mock_basinstatesconnectivity) : 

        conn1 = mock_basinstatesconnectivity 

        test_logger.debug("Before update state index : df = {} \n".format(conn1.get_table()))
        test_logger.debug("Changing state 2 to state 1")
        conn1.change_state_index(2,1)
        test_logger.debug("After update state index : df = {} \n".format(conn1.get_table()))


class TestBasinExplorer : 

    def test_generic_event_explorer(self, test_logger, mock_config, reference_table_Ni_4000at_monovacancy_sia, mock_state_data ):
        test_logger.debug(":=> Running basin exploration with generic event.")

        basin_explorer = BasinGenericEventExplorer(mock_config, reference_table_Ni_4000at_monovacancy_sia ) 
        basin_explorer.explore(state=mock_state_data)

        test_logger.debug("connexion_table : \n{}".format(basin_explorer.get_connectivity_table()))


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
        r = 0.5

        solver = BisectionSolver(M=M_abs_reduced, p0=p0, r=r, spectral_decomposition=False) 

        test_logger.debug("Solve Master Equation for : ")
        test_logger.debug("M = \n {}".format(M_abs_reduced))
        test_logger.debug("p0 = \n {}".format(p0))
        test_logger.debug("t = {}".format(t0))
        p = solver.solve_master_equation(t0)
        test_logger.debug("found p = \n {}".format(p))

        #computed with gnu octave
        res_expected = np.array([0.869014, 0.041671, 0.070828, 0.018488])
        test_logger.debug("Expected p = \n {}".format(res_expected))

        npt.assert_allclose(p, res_expected, rtol=1e-4)
        
        solver = BisectionSolver(M=M_abs_reduced, p0=p0, r=r, spectral_decomposition=True) 
        test_logger.debug("Solve Master Equation Using Sprectral Decomposition for : ")
        test_logger.debug("M = \n {}".format(M_abs_reduced))
        test_logger.debug("p0 = \n {}".format(p0))
        test_logger.debug("t = {}".format(t0))
        p = solver.solve_master_equation(t0)
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

        solver = BisectionSolver(M=M_abs_reduced, p0=p0, r=r) 

        test_logger.debug("Solve Master Equation for : ")
        test_logger.debug("M = \n {}".format(M_abs_reduced))
        test_logger.debug("p0 = \n {}".format(p0))
        test_logger.debug("t = {}".format(t0))
        p = solver.solve_master_equation(t0)
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
            exit_state = self.basin.selector.select_from_connectivity(self.basin.connectivity_table)
            manager.close_all()
