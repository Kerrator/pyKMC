import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import os
import copy
from pykmc.basins import  StatesConnectivity, BasinStatesConnectivity
import logging



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


