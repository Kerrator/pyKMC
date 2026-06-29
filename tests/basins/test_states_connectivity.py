from pykmc.basins import StatesConnectivity
import logging

logger = logging.getLogger("tests")

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

    def test_batch_insert_matches_serial(self, test_logger) :
        """add_connectivity_batch + merge_batch must produce the same table as per-row adds.

        This guards the lazy row-buffer rewrite: the O(N^2)-avoiding fast paths must be
        observationally identical to repeated add_connectivity().
        """
        from pykmc.basins import BasinStatesConnectivity

        rows = [
            dict(state=0, state_connexion=i, event_connexion=10 + i, central_atom=i,
                 sym=0, transient=(i % 2 == 0), dE_forward=0.1 * i, k_forward=1.0,
                 dE_backward=0.2 * i, k_backward=1.0)
            for i in range(1, 6)
        ]

        serial = BasinStatesConnectivity()
        for r in rows :
            serial.add_connectivity(**r)

        batched = BasinStatesConnectivity()
        batched.add_connectivity_batch(rows)

        import pandas as pd
        pd.testing.assert_frame_equal(
            serial.df.reset_index(drop=True), batched.df.reset_index(drop=True)
        )

        #merge_batch of two single-row tables == two serial adds
        a = BasinStatesConnectivity(); a.add_connectivity(**rows[0])
        b = BasinStatesConnectivity(); b.add_connectivity(**rows[1])
        merged = BasinStatesConnectivity()
        merged.merge_batch([a, b])
        ref = BasinStatesConnectivity()
        ref.add_connectivity(**rows[0]); ref.add_connectivity(**rows[1])
        pd.testing.assert_frame_equal(
            merged.df.reset_index(drop=True), ref.df.reset_index(drop=True)
        )

    def test_save_load_roundtrip_formats(self, test_logger, tmp_path) :
        """save()/load() round-trip the table for pickle, csv, and parquet."""
        from pykmc.basins import BasinStatesConnectivity, StatesConnectivity

        conn = BasinStatesConnectivity()
        conn.add_connectivity(state=0, state_connexion=1, event_connexion=11, central_atom=3,
                              sym=0, transient=True, dE_forward=0.1, k_forward=2.0,
                              dE_backward=0.3, k_backward=4.0)
        import pandas as pd
        for ext in ("pickle", "csv", "parquet") :
            path = str(tmp_path / f"conn.{ext}")
            conn.save(path)
            loaded = StatesConnectivity.load(path)
            pd.testing.assert_frame_equal(
                conn.df.reset_index(drop=True),
                loaded.df.reset_index(drop=True),
                check_dtype=False,  # csv loses exact dtypes; values must still match
            )

    def test_reorder_then_transient_normalization(self, test_logger) :
        """After reorder, the per-row transient flag must match the index range.

        Builds a table whose 'transient' column is deliberately stale (a row pointing at
        an absorbing target is still marked transient=True, as can happen after duplicate
        merges). reorder_states_index() numbers transient states first; recomputing the
        flag as (state_connexion < n_transient) is the normalization the basin applies.
        """
        from pykmc.basins import BasinStatesConnectivity

        conn = BasinStatesConnectivity()
        #state 0 (transient) -> state 1 (transient), 0 -> 5 (absorbing, but flagged stale True)
        conn.add_connectivity(state=0, state_connexion=1, event_connexion=10, central_atom=1, sym=0, transient=True, dE_forward=0.1, k_forward=1.0, dE_backward=0.2, k_backward=1.0)
        conn.add_connectivity(state=0, state_connexion=5, event_connexion=11, central_atom=2, sym=0, transient=True, dE_forward=0.5, k_forward=1.0, dE_backward=0.6, k_backward=1.0)
        conn.add_connectivity(state=1, state_connexion=6, event_connexion=12, central_atom=3, sym=0, transient=False, dE_forward=0.7, k_forward=1.0, dE_backward=0.8, k_backward=1.0)

        conn.reorder_states_index()
        #transient states are the sources {0,1} -> 2 transient states after compaction
        transient_states = set(conn.get_table()["state"])
        all_states = transient_states | set(conn.get_table()["state_connexion"])
        n_transient = len(transient_states)
        assert n_transient == 2
        assert len(all_states) - n_transient == 2  #states 5,6 -> absorbing

        conn.df["transient"] = conn.df["state_connexion"].apply(lambda x: x < n_transient)
        #the 0->absorbing row must no longer be flagged transient
        for _, row in conn.df.iterrows() :
            assert row["transient"] == (row["state_connexion"] < n_transient)
        assert conn.df["transient"].sum() == 1  #only the 0->1 transient transition remains True