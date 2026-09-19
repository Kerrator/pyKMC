"""Nonreciprocal catalogue aliases retain their declared reverse consumer."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
from pykmc.basins.detection import DetectorThreshold
from pykmc.basins.exploration import BasinGenericEventExplorer


def test_nonreciprocal_alias_uses_its_declared_reverse_in_both_consumers():
    # Accepted catalogue semantics permit 61 -> 42 while 42 -> 17. The
    # detector/explorer must not require a link back to 61 or substitute 17.
    frame = pd.DataFrame(
        {
            "idx_ref": [17, 42, 61],
            "idx_backward": [42, 17, 42],
            "event_id": ["same", "same", "same"],
            "id_final": ["same", "same", "same"],
            "energy_barrier": [0.1, 0.7, 0.5],
            "k": [7.0, 0.25, 3.0],
            "sym_matrix": [[np.eye(3)] for _ in range(3)],
        },
        index=[7, 103, 9],
    )
    before = frame.copy(deep=True)
    matrices_before = [matrix[0].copy() for matrix in frame.sym_matrix]
    selected = frame.loc[9].copy()
    observed = DetectorThreshold().detect(selected, frame, 0.6)
    calls = []

    def applicable(topologies):
        calls.append(tuple(topologies))
        return frame.loc[[9]].copy()

    def atoms(topology):
        assert topology == "same"
        return [5]

    catalogue = SimpleNamespace(table=frame, has_id_subset_table=applicable)
    environment = SimpleNamespace(
        atomic_environment_list=["same"], get_atoms_with_id=atoms
    )
    explorer = BasinGenericEventExplorer(
        SimpleNamespace(basin=SimpleNamespace(energy_thr=0.6)), catalogue
    )
    explorer.explore(
        SimpleNamespace(environment=environment), state_index=2, start_index=11
    )
    edges = explorer.get_connectivity_table()

    assert calls == [("same",)]
    assert len(edges) == 1
    edge = edges.iloc[0]
    assert int(edge.event_connexion) == 61
    assert int(edge.central_atom) == 5 and int(edge.sym) == 0
    assert int(edge.state) == 2 and int(edge.state_connexion) == 11
    assert edge.dE_forward == 0.5 and edge.k_forward == 3.0
    assert edge.dE_backward == 0.7 and edge.k_backward == 0.25
    assert not bool(observed)
    assert not bool(edge.transient)
    # Neither consumer may rewrite the nonreciprocal catalogue to make it pass.
    pd.testing.assert_frame_equal(frame, before)
    for expected, matrices in zip(matrices_before, frame.sym_matrix, strict=True):
        np.testing.assert_array_equal(matrices[0], expected)
