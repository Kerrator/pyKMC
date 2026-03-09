from pykmc.basins import StatesConnectivity, BasinStatesConnectivity
import logging

logger = logging.getLogger("tests")


class TestLazyMaterialization:

    def test_row_buffer_materializes_correctly(self):
        """Row buffer should produce identical DataFrame to eager concat."""
        sc = StatesConnectivity()
        for i in range(100):
            sc.add_connectivity(state=0, state_connexion=i+1, event_connexion=i,
                              central_atom=0, sym=0, transient=True,
                              dE_forward=0.5, k_forward=1.0,
                              dE_backward=0.3, k_backward=2.0)
        assert len(sc.df) == 100
        assert sc.df.iloc[0]["state_connexion"] == 1
        assert sc.df.iloc[99]["state_connexion"] == 100

    def test_empty_check_with_buffer(self):
        """df.empty should work correctly with row buffer."""
        sc = StatesConnectivity()
        assert sc.df.empty
        sc.add_connectivity(state=0, state_connexion=1, event_connexion=0,
                          central_atom=0, sym=0, transient=True,
                          dE_forward=0.5, k_forward=1.0,
                          dE_backward=0.3, k_backward=2.0)
        assert not sc.df.empty

    def test_clear_resets_buffer(self):
        """clear() should reset both row buffer and DataFrame."""
        sc = StatesConnectivity()
        sc.add_connectivity(state=0, state_connexion=1, event_connexion=0,
                          central_atom=0, sym=0, transient=True,
                          dE_forward=0.5, k_forward=1.0,
                          dE_backward=0.3, k_backward=2.0)
        assert len(sc.df) == 1
        sc.clear()
        assert sc.df.empty

    def test_merge_uses_buffer(self):
        """merge() should efficiently combine two tables via row buffer."""
        sc1 = BasinStatesConnectivity()
        sc2 = BasinStatesConnectivity()
        for i in range(5):
            sc1.add_connectivity(state=0, state_connexion=i+1, event_connexion=i,
                               central_atom=0, sym=0, transient=True,
                               dE_forward=0.5, k_forward=1.0,
                               dE_backward=0.3, k_backward=2.0)
            sc2.add_connectivity(state=1, state_connexion=i+10, event_connexion=i,
                               central_atom=1, sym=0, transient=True,
                               dE_forward=0.6, k_forward=1.1,
                               dE_backward=0.4, k_backward=2.1)
        sc1.merge(sc2)
        assert len(sc1.df) == 10

    def test_add_connectivity_batch(self):
        """add_connectivity_batch should add multiple rows at once."""
        sc = StatesConnectivity()
        rows = [{'state': 0, 'state_connexion': i, 'event_connexion': i,
                 'central_atom': 0, 'sym': 0, 'transient': True,
                 'dE_forward': 0.5, 'k_forward': 1.0,
                 'dE_backward': 0.3, 'k_backward': 2.0}
                for i in range(50)]
        sc.add_connectivity_batch(rows)
        assert len(sc.df) == 50

    def test_df_setter_backward_compat(self):
        """Direct df assignment should work for backward compatibility."""
        import pandas as pd
        sc = StatesConnectivity()
        sc.add_connectivity(state=0, state_connexion=1, event_connexion=0,
                          central_atom=0, sym=0, transient=True,
                          dE_forward=0.5, k_forward=1.0,
                          dE_backward=0.3, k_backward=2.0)
        # Direct assignment should override buffer
        new_df = pd.DataFrame({'state': [99], 'state_connexion': [100],
                               'event_connexion': [0], 'central_atom': [0],
                               'sym': [0], 'transient': [True],
                               'dE_forward': [0.1], 'k_forward': [0.2],
                               'dE_backward': [0.3], 'k_backward': [0.4]})
        sc.df = new_df
        assert len(sc.df) == 1
        assert sc.df.iloc[0]["state"] == 99

    def test_inplace_mutation_after_materialization(self):
        """In-place .loc mutations should work after materialization."""
        sc = BasinStatesConnectivity()
        sc.add_connectivity(state=0, state_connexion=1, event_connexion=0,
                          central_atom=0, sym=0, transient=True,
                          dE_forward=0.5, k_forward=1.0,
                          dE_backward=0.3, k_backward=2.0)
        sc.change_state_index(1, 99)
        assert sc.df.iloc[0]["state_connexion"] == 99


class TestStatesConnectivity :

    def test_add_connectivity(self, test_logger) : 

        state_connectivity = StatesConnectivity()
        test_logger.debug("From States Connectivity DataFrame : \n {}".format(state_connectivity.df))
        test_logger.debug("Add a new connexion between state 0 and 1")
        state_connectivity.add_connectivity(state=0, state_connexion=1, event_connexion=11, central_atom=342, sym=0, transient=True, dE_forward=0.1, k_backward=9, dE_backward=0.3, k_forward=6)
        test_logger.debug("New States Connectivity DataFrame : \n {}".format(state_connectivity.df))

    def test_get_transition_to_state(self, test_logger, mock_statesconnectivity) : 

        test_logger.debug("From States Connectivity DataFrame : \n {}".format(mock_statesconnectivity.df))
        res = mock_statesconnectivity.get_transition_to_state(1, as_tuples = True, return_all = False)
        test_logger.debug("Get first transition to state 1 : \n {}".format(res))

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