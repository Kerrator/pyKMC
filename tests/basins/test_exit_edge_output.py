"""The selected basin exit transition survives into the KMC hand-off and the log.

Companion to ``test_exit_edge_consumers.py`` (real ``execute`` on seeded
graphs): these units pin the consumer seams in isolation, the events-log
marker by row label and the explicit failure for a selected transition that
was never refined.
"""

import types

import numpy as np
import pandas as pd
import pytest

from pykmc.basins import BasinsGenericEvents, BasinStatesConnectivity
from pykmc.info_simulation import info_basin_events
from pykmc.result import BasinSelectorOutput, ErrorType


def connectivity(edges, labels=None):
    """Rows (source, destination, rate); ``transient`` follows the destination."""
    table = BasinStatesConnectivity()
    sources = {source for source, _, _ in edges}
    for index, (source, destination, rate) in enumerate(edges):
        table.add_connectivity(
            state=source,
            state_connexion=destination,
            event_connexion=17 + index * 7,
            central_atom=index,
            sym=0,
            transient=destination in sources,
            dE_forward=0.0,
            k_forward=float(rate),
            dE_backward=0.0,
            k_backward=0.0,
        )
    if labels is not None:
        table.df.index = list(labels)
    return table


class TestConsumers:
    def _reference(self, ids):
        return types.SimpleNamespace(
            table=pd.DataFrame(
                {
                    "idx_ref": ids,
                    "idx_backward": ids,
                    "dra": [0.1 * (i + 1) for i in range(len(ids))],
                    "energy_barrier": [0.2 * (i + 1) for i in range(len(ids))],
                }
            )
        )

    def test_info_basin_events_marks_the_selected_row_not_the_first_to_its_destination(
        self,
    ):
        table = connectivity(
            [(0, 1, 1.0), (1, 0, 2.0), (0, 2, 1.0), (1, 2, 1.0)], [91, 3, 104, 8]
        )
        reference = self._reference([17, 24, 31, 38])
        system_types = ["Si", "Ge", "Si", "Ge"]
        position, info = info_basin_events(system_types, reference, table, 8)
        assert position == 1
        assert list(info.reference_events) == [31, 38]
        assert list(info.central_atom) == [2, 3]
        position, _ = info_basin_events(system_types, reference, table, 104)
        assert position == 0
        with pytest.raises(KeyError):
            info_basin_events(system_types, reference, table, 91)

    def test_exit_output_refuses_a_selected_transition_without_a_refined_saddle(self):
        basin = BasinsGenericEvents.__new__(BasinsGenericEvents)
        basin.connectivity_table = connectivity([(0, 1, 1.0), (0, 2, 1.0)], [104, 8])
        basin.absorbing_saddle_positions = {104: np.zeros((1, 3))}
        basin.states = {}
        result = basin._exit_output(
            BasinSelectorOutput(t_exit=1.0, exit_state=2, exit_row=8, from_state=0)
        )
        assert not result.is_ok()
        error = result.err_value()
        assert error.type is ErrorType.BASIN_EXIT_NOT_REFINED
        assert error.variables == {
            "exit_row": 8,
            "from_state": 0,
            "exit_state": 2,
            "idx_ref": 24,
        }
        result = basin._exit_output(
            BasinSelectorOutput(
                t_exit=1.0, exit_state=2, exit_row=None, from_state=None
            )
        )
        assert result.err_value().type is ErrorType.BASIN_EXIT_NOT_REFINED
