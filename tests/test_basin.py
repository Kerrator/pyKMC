import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import os
import copy
from pykmc.basins import  StatesConnectivity, BasinStatesConnectivity, BasinGenericEventExplorer, BasinsGenericEvents
import logging
from pykmc.enginemanager.lmpi.pool import ManagerFactory
from .conftest import mpi_test


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
