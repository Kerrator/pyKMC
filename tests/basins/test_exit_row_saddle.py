"""Exit-row / saddle pairing when several connectivity rows reach the same exit state."""

import logging

import numpy as np
import numpy.testing as npt

from pykmc.basins.basin import BasinsGenericEvents
from pykmc.basins.connectivity import BasinStatesConnectivity
from pykmc.result import ErrorType


def _basin_with_two_rows_into_one_exit() -> BasinsGenericEvents:
    """Two rows (different central atoms / events) into exit state 2, from states 0 and 1."""
    basin = object.__new__(BasinsGenericEvents)  # no __init__: no manager/engine needed
    table = BasinStatesConnectivity()
    # (state, state_connexion, event, central_atom, sym, transient, dE_f, k_f, dE_b, k_b)
    table.add_connectivity(0, 1, 35, 8257, 0, True, 0.626, 2.7e-10, 0.626, 2.6e-10)
    table.add_connectivity(1, 2, 16, 2199, 0, False, 0.7916, 5.0e-14, 0.7987, 8.9e-13)
    table.add_connectivity(0, 2, 14, 2219, 0, False, 0.7911, 5.1e-14, 0.7987, 8.5e-13)
    basin.connectivity_table = table
    basin.absorbing_saddle_positions = {}
    return basin


class TestExitRowSaddle:
    """_select_exit_row / _transplant_saddles / _reconcile_refine_exclusions."""

    def test_saddle_follows_the_selected_row(self, test_logger: logging.Logger) -> None:
        """The saddle handed to KMC belongs to the row whose from-state/central atom is returned."""
        basin = _basin_with_two_rows_into_one_exit()
        sad_2219 = np.full((5, 3), 1.0)
        sad_2199 = np.full((7, 3), 2.0)
        basin.absorbing_saddle_positions[basin._saddle_key(0, 2, 2219, 14, 0)] = sad_2219
        basin.absorbing_saddle_positions[basin._saddle_key(1, 2, 2199, 16, 0)] = sad_2199

        res = basin._select_exit_row(2)
        assert res.is_ok()
        from_state, event_idx, central_atom, sym, dE, saddle = res.ok_value()
        # smallest from-state first, like get_transition_to_state
        assert (from_state, event_idx, central_atom, sym) == (0, 14, 2219, 0)
        npt.assert_allclose(dE, 0.7911)
        assert saddle is sad_2219

    def test_row_without_saddle_is_skipped(self, test_logger: logging.Logger) -> None:
        """A row whose refinement left no saddle must not be paired with another row's saddle."""
        basin = _basin_with_two_rows_into_one_exit()
        sad_2199 = np.full((7, 3), 2.0)
        basin.absorbing_saddle_positions[basin._saddle_key(1, 2, 2199, 16, 0)] = sad_2199

        res = basin._select_exit_row(2)
        assert res.is_ok()
        from_state, event_idx, central_atom, sym, dE, saddle = res.ok_value()
        assert (from_state, event_idx, central_atom, sym) == (1, 16, 2199, 0)
        npt.assert_allclose(dE, 0.7916)
        assert saddle is sad_2199

    def test_no_saddle_at_all_is_an_error(self, test_logger: logging.Logger) -> None:
        """No row into the exit state carries a saddle -> BASIN_NO_VIABLE_EXIT."""
        basin = _basin_with_two_rows_into_one_exit()
        res = basin._select_exit_row(2)
        assert not res.is_ok()
        assert res.err_value().type == ErrorType.BASIN_NO_VIABLE_EXIT

    def test_transplant_keeps_row_identity(self, test_logger: logging.Logger) -> None:
        """Lazy-merge re-keying moves every row's saddle to the new exit index without overwriting."""
        basin = _basin_with_two_rows_into_one_exit()
        a = np.zeros((3, 3))
        b = np.ones((3, 3))
        c = np.full((3, 3), 5.0)
        basin.absorbing_saddle_positions[basin._saddle_key(0, 7, 2219, 14, 0)] = a
        basin.absorbing_saddle_positions[basin._saddle_key(1, 7, 2199, 16, 0)] = b
        basin.absorbing_saddle_positions[basin._saddle_key(0, 2, 2219, 14, 0)] = c  # pre-existing, must survive

        basin._transplant_saddles(7, 2)
        assert basin.absorbing_saddle_positions[basin._saddle_key(0, 2, 2219, 14, 0)] is c
        assert basin.absorbing_saddle_positions[basin._saddle_key(1, 2, 2199, 16, 0)] is b
        # the old keys are left in place (harmless; the old index is no longer referenced)
        assert basin.absorbing_saddle_positions[basin._saddle_key(0, 7, 2219, 14, 0)] is a

    def test_creating_row_is_preferred(self, test_logger: logging.Logger) -> None:
        """The row whose reconstruction produced the exit geometry wins over table order."""
        basin = _basin_with_two_rows_into_one_exit()
        sad_2219 = np.full((5, 3), 1.0)
        sad_2199 = np.full((7, 3), 2.0)
        basin.absorbing_saddle_positions[basin._saddle_key(0, 2, 2219, 14, 0)] = sad_2219
        basin.absorbing_saddle_positions[basin._saddle_key(1, 2, 2199, 16, 0)] = sad_2199
        basin._creating_row = {2: (1, 2199, 16, 0)}  # state 2 was reconstructed from state 1 via 2199

        from_state, event_idx, central_atom, sym, dE, saddle = basin._select_exit_row(2).ok_value()
        assert (from_state, event_idx, central_atom, sym) == (1, 16, 2199, 0)
        assert saddle is sad_2199

    def test_refine_exclusion_reconciled(self, test_logger: logging.Logger) -> None:
        """A state excluded by one row's failed refinement is re-admitted if another row has a saddle."""
        basin = _basin_with_two_rows_into_one_exit()
        basin._exit_excluded_states = {2, 9}
        basin._refine_excluded = {2, 9}
        basin._failed_exit_states = set()
        basin.absorbing_saddle_positions[basin._saddle_key(1, 2, 2199, 16, 0)] = np.zeros((3, 3))
        basin._reconcile_refine_exclusions()
        assert basin._exit_excluded_states == {9}
        # a failed reconstruction stays excluded even if a saddle exists
        basin._exit_excluded_states = {2}
        basin._refine_excluded = {2}
        basin._failed_exit_states = {2}
        basin._reconcile_refine_exclusions()
        assert basin._exit_excluded_states == {2}
