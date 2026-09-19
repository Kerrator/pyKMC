"""Logical reverse identity at the public basin detector/explorer boundaries.

The literal pair and units are inherited unchanged from the archived reverse
consumer oracle. A recording catalogue subset is the only consumer seam; the
detector, explorer, and connectivity storage are real. No native physics or
first-passage sampling is claimed.
"""

import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pykmc.basins.detection import DetectorThreshold
from pykmc.basins.exploration import BasinGenericEventExplorer


def reference_pair():
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
                "energy_barrier": 0.7,
                "k": 0.25,
                "sym_matrix": [np.eye(3)],
            },
        ],
        index=[104, 8],
    )


def independent_copy(frame):
    """Pandas deep copies do not copy ndarray objects in object columns."""
    result = frame.copy(deep=True)
    result["sym_matrix"] = frame["sym_matrix"].map(
        lambda matrices: [matrix.copy() for matrix in matrices]
    )
    return result


def explorer_for(frame, subset, threshold):
    calls = []

    def applicable(ids):
        calls.append(tuple(ids))
        return subset

    def atoms(topology):
        assert topology == "same"
        return [5]

    reference = SimpleNamespace(table=frame, has_id_subset_table=applicable)
    environment = SimpleNamespace(
        atomic_environment_list=["same"], get_atoms_with_id=atoms
    )
    explorer = BasinGenericEventExplorer(
        SimpleNamespace(basin=SimpleNamespace(energy_thr=threshold)), reference
    )
    return explorer, SimpleNamespace(environment=environment), calls


def assert_single_edge(explorer, *, transient, backward_barrier, backward_rate):
    edges = explorer.get_connectivity_table()
    assert len(edges) == 1
    edge = edges.iloc[0]
    assert int(edge.state) == 2
    assert int(edge.state_connexion) == 11
    assert int(edge.event_connexion) == 17
    assert int(edge.central_atom) == 5
    assert int(edge.sym) == 0
    assert edge.dE_forward == 0.5
    assert edge.k_forward == 3.0
    assert edge.dE_backward == backward_barrier
    assert edge.k_backward == backward_rate
    assert bool(edge.transient) is transient


def test_copied_forward_uses_catalogue_link_not_stale_series_link():
    frame = reference_pair()
    before = independent_copy(frame)
    selected = frame.loc[104].copy()
    # A copied candidate still identifies logical 17; it cannot redefine its
    # catalogue reverse as itself. The authoritative pair remains 17 -> 42.
    selected["idx_backward"] = 17
    selected.name = 8  # Also collides with the parent's reverse row label.
    assert not DetectorThreshold().detect(selected, frame, 0.6)
    assert selected["idx_backward"] == 17
    assert selected.name == 8
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("subset_label", [901, 8], ids=["new-label", "colliding-label"])
def test_explorer_uses_logical_pair_when_subset_is_relabelled(subset_label):
    frame = reference_pair()
    subset = independent_copy(frame.loc[[104]])
    subset.index = [subset_label]
    before, subset_before = independent_copy(frame), independent_copy(subset)
    explorer, state, calls = explorer_for(frame, subset, 0.6)
    explorer.explore(state, state_index=2, start_index=11)
    assert calls == [("same",)]
    assert_single_edge(
        explorer, transient=False, backward_barrier=0.7, backward_rate=0.25
    )
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(subset, subset_before)


@pytest.mark.parametrize(
    "active_barrier,expected",
    [(0.5, True), (0.6, False)],
    ids=["active-low", "active-equal"],
)
def test_refined_forward_barrier_is_not_replaced_by_generic_barrier(
    active_barrier, expected
):
    frame = reference_pair()
    # Only this explicit threshold control differs from the archived 0.5/0.7
    # pair: the refined site's forward is lower than its generic forward.
    frame.loc[104, "energy_barrier"] = 0.9
    frame.loc[8, "energy_barrier"] = 0.4
    before = independent_copy(frame)
    active = pd.Series({"num_reference_event": 17, "energy_barrier": active_barrier})
    active_before = active.copy(deep=True)
    observed = DetectorThreshold().detect(active, frame, 0.6, is_refined=True)
    assert bool(observed) is expected
    pd.testing.assert_series_equal(active, active_before)
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize(
    "threshold", [0.5, 0.7], ids=["forward-equal", "reverse-equal"]
)
def test_equality_at_either_linked_barrier_is_not_transient(threshold):
    frame = reference_pair()
    before = independent_copy(frame)
    assert not DetectorThreshold().detect(frame.loc[104], frame, threshold)
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("refined", [False, True], ids=["reference", "refined-active"])
@pytest.mark.parametrize(
    "defect,required_ids",
    [
        ("missing-forward", (17,)),
        ("missing-reverse", (17, 42)),
        ("duplicate-forward", (17,)),
        ("duplicate-reverse", (17, 42)),
    ],
    ids=[
        "missing-forward",
        "missing-reverse",
        "duplicate-forward",
        "duplicate-reverse",
    ],
)
def test_missing_or_ambiguous_logical_identity_is_explicit(
    defect, required_ids, refined
):
    frame = reference_pair()
    selected = (
        pd.Series({"num_reference_event": 17, "energy_barrier": 0.5})
        if refined
        else frame.loc[104].copy()
    )
    if defect == "missing-forward":
        frame = frame.loc[[8]].copy()
    elif defect == "missing-reverse":
        frame = frame.loc[[104]].copy()
    else:
        duplicate_label = 104 if defect == "duplicate-forward" else 8
        extra = independent_copy(frame.loc[[duplicate_label]])
        extra.index = [777]
        frame = pd.concat([frame, extra])
    before = independent_copy(frame)
    with pytest.raises(ValueError) as captured:
        DetectorThreshold().detect(selected, frame, 0.6, is_refined=refined)
    for idx_ref in required_ids:
        assert re.search(rf"\b{idx_ref}\b", str(captured.value)), str(captured.value)
    pd.testing.assert_frame_equal(frame, before)


def test_explorer_requires_reverse_even_when_forward_exceeds_threshold():
    frame = reference_pair().loc[[104]].copy()
    frame.loc[104, "energy_barrier"] = 0.9
    before = independent_copy(frame)
    subset = independent_copy(frame)
    explorer, state, calls = explorer_for(frame, subset, 0.6)
    with pytest.raises(ValueError) as captured:
        explorer.explore(state, state_index=2, start_index=11)
    assert re.search(r"\b17\b", str(captured.value))
    assert re.search(r"\b42\b", str(captured.value))
    assert calls == [("same",)]
    assert explorer.get_connectivity_table().empty
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(subset, before)


def test_legitimate_self_link_uses_its_own_reverse_metadata():
    frame = reference_pair()
    frame.loc[104, "idx_backward"] = 17
    frame.loc[8, "idx_backward"] = 42
    # Row 42 remains a distinct same-topology row with the archived 0.7/.25
    # values; the declared self-link resolves to row 17's own 0.5/3.0 instead.
    before = independent_copy(frame)
    assert DetectorThreshold().detect(frame.loc[104], frame, 0.6)
    subset = independent_copy(frame.loc[[104]])
    explorer, state, calls = explorer_for(frame, subset, 0.6)
    explorer.explore(state, state_index=2, start_index=11)
    assert calls == [("same",)]
    assert_single_edge(
        explorer, transient=True, backward_barrier=0.5, backward_rate=3.0
    )
    pd.testing.assert_frame_equal(frame, before)
