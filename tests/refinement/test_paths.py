"""Exercise the existing ordinary and basin consumers with scripted native data."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pykmc.basins.basin import BasinsGenericEvents, StateData
from pykmc.constants import KB
from pykmc.event_recycling import DistanceRecycling
from pykmc.psr import PSRError
from pykmc.result import Err, ErrorInfo, ErrorType
from .conftest import (
    BASIN,
    BULK,
    CELL,
    DE,
    FINAL_X,
    MIN,
    ORDER,
    PREF,
    RAW_X,
    SAVED_MIN,
    TYPES,
    coords,
)


def assert_match(call):
    args, kw = call
    assert kw == {"candidate1": 1, "candidate2": 0}
    assert args[1] == ["Fe", "Ni", "Fe"]
    assert args[4] == ["Ni", "Fe", "Fe"]
    np.testing.assert_allclose(args[2], coords([-0.5, 0.25, 1.0]))
    np.testing.assert_allclose(args[5][:, 0], [0, -0.75, 0.75])


def assert_request(kw, sym, crop, basin=False):
    np.testing.assert_array_equal(kw["saddle_idx"], ORDER)
    assert kw["central_atom_idx"] == 3
    source = SAVED_MIN if basin else MIN
    if not basin or crop:
        np.testing.assert_array_equal(kw["types"], TYPES)
    else:
        # Characterize the omission at the pin; this is not a freeze contract.
        assert "types" not in kw
    if crop:
        np.testing.assert_allclose(kw["positions"], source, atol=1e-12, rtol=0)
        np.testing.assert_allclose(kw["cell"], CELL)
        np.testing.assert_allclose(
            kw["saddle_positions"], coords(RAW_X[sym], wrapped=not basin)
        )
    else:
        expected = source.copy()
        expected[ORDER] = coords(RAW_X[sym], wrapped=True)
        np.testing.assert_allclose(kw["positions"], expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("crop", [False, True], ids=["cell", "crop"])
def test_ordinary_refinement_order_energy_and_next_step(crop, make_case):
    sim, match = make_case(crop)
    worker = sim.manager
    worker.batch = 2
    outcome = sim.execute_refinements(sim.reference_table.table)
    assert worker.completed == [1, 0]
    assert_match(match.call_args)
    rows = outcome.get_successes_results()
    assert len(rows) == 2
    for sym, (kw, row) in enumerate(zip(worker.calls, rows, strict=True)):
        assert_request(kw, sym, crop)
        assert row.refined == "T" and row.dE_forward == DE
        assert row.E_saddle == (DE if crop else BULK + DE)
        np.testing.assert_array_equal(row.neighbors, ORDER)
        np.testing.assert_allclose(
            row.saddle_positions, coords(RAW_X[sym], wrapped=True, y=2.02)
        )
        np.testing.assert_allclose(
            row.min2_positions, coords(FINAL_X[sym], wrapped=True)
        )
    np.testing.assert_allclose(sim.system.positions, MIN)
    table = sim.add_active_events(rows)
    np.testing.assert_allclose(
        table.table.k, PREF * np.exp(-DE / (KB * 900.0)), rtol=1e-12, atol=0
    )
    # A later alias compares by atom id, not local array order; retain the
    # earlier row and do not collapse its genuinely different symmetry sibling.
    alias = table.table.iloc[1].copy()
    alias["num_reference_event"] = 999
    for field in ("neighbors", "saddle_positions", "final_positions"):
        alias[field] = alias[field][::-1].copy()
    table.add(alias)
    table.remove_duplicates(CELL, sim.neighbors_list)
    assert table.table.num_reference_event.tolist() == [37, 37]
    # Next step: the old pair bypasses BOTH registration and solving. Reconstruct
    # the selected symmetry on its original ids despite a rebuilt neighbour order.
    # Execute a remote spectator event and prune through the actual recycler.
    spectator = table.table.iloc[0].copy()
    spectator["atom_index"] = 0
    table.add(spectator)
    table.recycler = DistanceRecycling(0.1, 1.0)
    original = sim.system.positions.copy()
    sim.system.positions[0, 2] += 0.15
    table.prune_for_recycling(2, sim.system, original)
    assert table.table.refined.tolist() == ["RT", "RT"]
    sim.neighbors_list.get_neighbors.return_value = ORDER[::-1].copy()
    assert (
        sim.execute_refinements(
            sim.reference_table.table, existing_pairs=table.existing_pairs()
        ).results
        == []
    )
    assert match.call_count == 1 and len(worker.calls) == 2
    initial = sim.system.positions.copy()
    final = initial.copy()
    final[ORDER] = coords(FINAL_X[1], wrapped=True)
    # Keep independent snapshots: reconstruct mutates/reuses its scratch array.
    pushes = []
    answers = iter([(initial.copy(), BULK), (final.copy(), BULK + 0.1)])

    def group(*, config, positions, types):
        pushes.append(positions.copy())
        return next(answers)

    worker.group_minimizer = group
    out = sim._reconstruction_active_event(1, table).ok_value()
    np.testing.assert_allclose(out.min2_positions, final)
    assert len(pushes) == 2
    frac = sim.config.reconstruction.push_fraction
    for submitted, target in zip(
        pushes, [np.array([-0.5, 0.25, 1.0]), FINAL_X[1]], strict=True
    ):
        expected = initial.copy()
        expected[ORDER] = coords(
            RAW_X[1] + frac * (target - RAW_X[1]), wrapped=True, y=2.02 - frac * 0.02
        )
        np.testing.assert_allclose(submitted, expected)
    np.testing.assert_allclose(sim.system.positions, initial)


@pytest.mark.parametrize("crop", [False, True])
def test_policy_and_matching_failures(crop, make_case):
    sim, match = make_case(crop)
    table = sim.reference_table.table
    sim.config.control.refine_thr = 0
    outcome = sim.execute_refinements(table)
    assert match.call_count == 1 and sim.manager.calls == []
    assert [r.refined for r in outcome.get_successes_results()] == ["F", "F"]
    assert all(r.dE_forward == DE for r in outcome.get_successes_results())
    table.loc[0, "energy_barrier"] = 0.5
    sim.config.control.refine_thr = 1.0
    outcome = sim.execute_refinements(table)
    assert [r.err_value().type for r in outcome.results] == [
        ErrorType.REFINEMENT_INVALID_ENERGY_BARRIER
    ] * 2
    match.return_value = Err(
        ErrorInfo(type=PSRError.NO_MATCH_FOUND, message="controlled")
    )
    outcome = sim.execute_refinements(table)
    (err,) = outcome.results
    assert err.err_value().type == PSRError.NO_MATCH_FOUND
    assert (
        err.err_value().variables == {"n_ref_event": 37} and len(sim.manager.calls) == 2
    )


def basin_for(sim):
    sim.manager.full_min = BASIN
    basin = BasinsGenericEvents(sim.config, sim.reference_table, set(), sim.manager)
    saved = deepcopy(sim.system)
    saved.update_positions(SAVED_MIN.copy())
    basin.states = {8: StateData(saved, sim.atomic_environment, sim.neighbors_list)}
    basin.connectivity_table = SimpleNamespace(
        df=pd.DataFrame(
            [
                dict(
                    state=8,
                    state_connexion=12,
                    event_connexion=37,
                    central_atom=3,
                    sym=1,
                    transient=False,
                    dE_forward=-1.0,
                    k_forward=-1.0,
                )
            ]
        )
    )
    return basin


@pytest.mark.parametrize("crop", [False, True], ids=["cell", "crop"])
def test_basin_selected_symmetry_own_minimum_and_rate(crop, make_case):
    sim, match = make_case(crop)
    sim.reference_table.table.loc[0, "energy_barrier"] = 0.5  # ordinary would reject
    basin = basin_for(sim)
    assert basin.refine_absorbing(sim.system).is_ok()
    np.testing.assert_allclose(sim.manager.min_calls, [SAVED_MIN], atol=1e-12, rtol=0)
    assert_match(match.call_args)
    (kw,) = sim.manager.calls
    assert_request(kw, 1, crop, basin=True)
    row = basin.connectivity_table.df.iloc[0]
    assert row.dE_forward == DE
    assert row.k_forward == pytest.approx(
        PREF * np.exp(-DE / (KB * 900.0)), rel=1e-12, abs=0
    )
    np.testing.assert_allclose(
        basin.absorbing_saddle_positions[8, 12], coords(RAW_X[1], wrapped=True, y=2.02)
    )


@pytest.mark.parametrize("style", ["global", "global/reconstruction"])
def test_basin_state_geometry_is_separate_from_exit_refinement(style, make_case):
    sim, match = make_case()
    basin = basin_for(sim)
    sim.config.basin.style = style
    final = SAVED_MIN.copy()
    final[ORDER] = coords(FINAL_X[1], wrapped=True)
    pushed = []
    answers = iter([final] if style == "global" else [SAVED_MIN, final])

    def group(**kw):
        pushed.append(kw["positions"].copy())
        return next(answers).copy(), BASIN

    sim.manager.group_minimizer = group
    state = basin.system_from_state(8, 37, 3, 1).ok_value()
    assert_match(match.call_args)
    np.testing.assert_allclose(state.positions, final)
    if style == "global":
        np.testing.assert_allclose(pushed, [final])
    else:
        assert len(pushed) == 2
        np.testing.assert_allclose(np.array(pushed)[:, [0, 2]], [SAVED_MIN[[0, 2]]] * 2)
        # Midpoints are between known wrapped saddle and distinct wrapped endpoints.
        frac = sim.config.reconstruction.push_fraction
        np.testing.assert_allclose(
            pushed[0][ORDER, 0],
            np.mod(RAW_X[1] + frac * (np.array([-0.5, 0.25, 1.0]) - RAW_X[1]), 10),
        )
        np.testing.assert_allclose(
            pushed[1][ORDER, 0], np.mod(RAW_X[1] + frac * (FINAL_X[1] - RAW_X[1]), 10)
        )
    assert not sim.manager.calls
    np.testing.assert_allclose(sim.system.positions, MIN)


@pytest.mark.parametrize("sym", [0, 1], ids=["identity-sym", "mirror-sym"])
def test_basin_deforms_in_rotated_psr_frame(sym, make_case):
    """Exercise composition order with a transverse displacement, not just translation."""
    sim, match = make_case(crop=True)
    basin = basin_for(sim)
    basin.connectivity_table.df.at[0, "sym"] = sym
    registration = match.return_value.ok_value()
    # Rotate 90 degrees about x. This leaves the initial x-aligned cloud fixed,
    # but the catalogue mover's positive y displacement must appear in lab z.
    registration.rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    record = sim.reference_table.table
    for name, offset in (("saddle_positions", 0.25), ("final_positions", 0.5)):
        guess = record.at[0, name].copy()
        guess[0, 1] += offset
        record.at[0, name] = guess
    calls = []

    def minimize(**kw):
        calls.append(kw["positions"].copy())
        return kw["positions"].copy(), BASIN

    sim.manager.group_minimizer = minimize
    sim.config.basin.style = "global"
    expected_final = SAVED_MIN.copy()
    expected_final[ORDER] = coords(FINAL_X[sym], wrapped=True)
    expected_final[3, 2] += 0.5
    out = basin.system_from_state(8, 37, 3, sym).ok_value()
    np.testing.assert_allclose(calls, [expected_final], atol=1e-12, rtol=0)
    np.testing.assert_allclose(out.positions, expected_final)
    assert basin.refine_absorbing(sim.system).is_ok()
    (kw,) = sim.manager.calls
    expected_saddle = coords(RAW_X[sym]).astype(float)
    expected_saddle[1, 2] += 0.25
    np.testing.assert_allclose(
        kw["saddle_positions"], expected_saddle, atol=1e-12, rtol=0
    )


def test_basin_registration_failure_is_returned_without_solving(make_case):
    """Both basin mapping consumers stop rather than substituting another event."""
    sim, match = make_case()
    basin = basin_for(sim)
    failure = Err(
        ErrorInfo(type=PSRError.NO_MATCH_FOUND, message="controlled basin miss")
    )
    match.return_value = failure
    connectivity = basin.connectivity_table.df.copy(deep=True)
    assert basin.system_from_state(8, 37, 3, 1) is failure
    assert sim.manager.min_calls == []
    assert basin.refine_absorbing(sim.system) is failure
    np.testing.assert_allclose(sim.manager.min_calls, [SAVED_MIN], atol=1e-12, rtol=0)
    assert match.call_count == 2
    assert sim.manager.calls == []
    assert basin.absorbing_saddle_positions == {}
    pd.testing.assert_frame_equal(basin.connectivity_table.df, connectivity)
    np.testing.assert_allclose(basin.states[8].system.positions, SAVED_MIN)
