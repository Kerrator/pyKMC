"""Protocol fixtures extend R2 through refined rows, logical ids and explorer.

No native engine/MPI pool. Expected barriers are directly specified data.
Logical ids (17,42) intentionally differ from pandas labels (104,8).
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pykmc.basins.detection import DetectorThreshold
from pykmc.basins.exploration import BasinGenericEventExplorer


def table(reverse_barrier=0.7):
    return pd.DataFrame(
        [
            {
                "idx_ref": 17,
                "idx_backward": 42,
                "event_id": "same",
                "id_final": "same",
                "energy_barrier": 0.5,
                "k": 3.0,
                "sym_matrix": [np.eye(3)],
            },
            {
                "idx_ref": 42,
                "idx_backward": 17,
                "event_id": "same",
                "id_final": "same",
                "energy_barrier": reverse_barrier,
                "k": 0.25,
                "sym_matrix": [np.eye(3)],
            },
        ],
        index=[104, 8],
    )


@pytest.mark.parametrize("refined", [False, True])
def test_threshold_follows_logical_reverse_not_same_topology_minimum(refined):
    frame = table()
    selected = (
        pd.Series({"energy_barrier": 0.5, "num_reference_event": 17})
        if refined
        else frame.loc[104]
    )
    detected = DetectorThreshold().detect(selected, frame, 0.6, is_refined=refined)
    print(
        {
            "is_refined": refined,
            "logical_forward": 17,
            "logical_reverse": 42,
            "barriers": [0.5, 0.7],
            "threshold": 0.6,
            "observed_transient": bool(detected),
        }
    )
    assert not detected


@pytest.mark.parametrize("refined", [False, True])
@pytest.mark.parametrize("threshold,expected", [(0.4, False), (0.8, True)])
def test_linked_reverse_threshold_controls(refined, threshold, expected):
    frame = table()
    selected = (
        pd.Series({"energy_barrier": 0.5, "num_reference_event": 17})
        if refined
        else frame.loc[104]
    )
    assert (
        bool(DetectorThreshold().detect(selected, frame, threshold, is_refined=refined))
        == expected
    )


def test_missing_logical_reverse_is_not_hidden_by_same_topology():
    frame = table().loc[[104]]
    with pytest.raises(ValueError, match="17.*42|42.*17"):
        DetectorThreshold().detect(frame.loc[104], frame, 0.6)


@pytest.mark.parametrize("threshold,expected", [(0.6, False), (0.8, True)])
def test_explorer_flag_agrees_with_recorded_reverse_barrier(threshold, expected):
    frame = table()
    reference = SimpleNamespace(
        table=frame,
        has_id_subset_table=lambda ids: frame,
    )
    environment = SimpleNamespace(
        atomic_environment_list=["same"],
        get_atoms_with_id=lambda topology: [5],
    )
    explorer = BasinGenericEventExplorer(
        SimpleNamespace(basin=SimpleNamespace(energy_thr=threshold)),
        reference,
    )
    explorer.explore(SimpleNamespace(environment=environment))
    forward = explorer.connectivity_table.df.query("event_connexion == 17").iloc[0]
    assert forward["dE_backward"] == 0.7
    assert forward["k_backward"] == 0.25
    print(
        {
            "threshold": threshold,
            "forward_barrier": forward["dE_forward"],
            "linked_reverse_barrier": forward["dE_backward"],
            "observed_transient": bool(forward["transient"]),
        }
    )
    assert bool(forward["transient"]) == expected
