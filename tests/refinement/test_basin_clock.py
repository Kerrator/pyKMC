"""One real KMC.run step: do not mistake a successful ordinary fallback for a basin exit."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import pykmc.kmc as loop
from pykmc.basins.basin import BasinsGenericEvents, StateData
from pykmc.basins.connectivity import BasinStatesConnectivity
from pykmc.constants import KB
from pykmc.result import BasinSelectorOutput, Err, ErrorInfo, ErrorType, Ok
from .conftest import (
    BASIN,
    BULK,
    DE,
    FINAL_X,
    MIN,
    ORDER,
    PREF,
    RAW_X,
    SAVED_MIN,
    coords,
)


@pytest.mark.parametrize(
    "path",
    [
        "accepted",
        "exploration-error",
        "refinement-error",
        "selection-error",
        "wrong-first-min",
        "wrong-second-min",
    ],
)
def test_selected_event_coordinates_rate_and_clock(
    path, make_case, monkeypatch, tmp_path
):
    sim, _ = make_case()
    sim.config.control.n_steps = 1
    sim.config.control.basin = True
    worker = sim.manager
    worker.broadcast = Mock()
    initial = MIN.copy()
    ordinary, exit_pos = initial.copy(), SAVED_MIN.copy()
    ordinary[ORDER] = coords(FINAL_X[0], wrapped=True)
    exit_pos[ORDER] = coords(FINAL_X[1], wrapped=True)
    record = sim.reference_table.table
    exit_record = record.copy(deep=True)
    exit_record.loc[0, ["idx_ref", "energy_barrier", "k_pref"]] = [81, 0.5, 3 * PREF]
    sim.reference_table.table = pd.concat([record, exit_record], ignore_index=True)
    sim.reference_table.has_id_subset_table = lambda _ids: record
    sim.visited_environments = set()
    sim.atomic_environment.atomic_environment_list = ["env"]
    sim.neighbors_list.neighbors_list = {"rnei": [], "rcut": []}
    monkeypatch.setattr(loop, "NeighborsList", lambda *args: sim.neighbors_list)
    monkeypatch.setattr(
        loop, "AtomicEnvironment", lambda *args, **kw: sim.atomic_environment
    )
    monkeypatch.setattr(loop.DetectorThreshold, "detect", lambda *args: True)
    monkeypatch.setattr("pykmc.algorithms.random.random", iter([0.1, 0.5]).__next__)
    for name in [
        "minimize_system",
        "get_new_environments",
        "central_atoms_research",
        "add_reference_events",
        "_save",
        "_append_snapshot_to_trajectory",
        "_save_restart_file",
        "_close",
    ]:
        monkeypatch.setattr(sim, name, Mock(return_value=[]))
    sim.execute_event_searches = Mock(
        return_value=SimpleNamespace(results=[], get_successes_results=lambda: [])
    )
    # Isolate presentation only, not control flow: observe the actual executed KMC line below.
    pretty = Mock()
    pretty.output_msg.return_value = (0, "")
    monkeypatch.setattr(loop, "info_active_events", lambda *a: pretty)
    monkeypatch.setattr(loop, "info_basin_events", lambda *a: (0, pretty))
    monkeypatch.setattr(loop, "KMCLoopInfo", Mock())
    for name in [
        "get_info_atomic_environments",
        "get_info_reference_event_searches",
        "get_info_is_valid_reference_events",
        "get_info_refinements",
    ]:
        monkeypatch.setattr(sim, name, Mock())

    stage = []

    def failure():
        return Err(ErrorInfo(type=ErrorType.EVENT_NOT_FOUND, message="scripted stage"))

    class PrescribedBasin(BasinsGenericEvents):
        def _initialize(self, system):
            worker.full_min = BASIN
            self.connectivity_table = BasinStatesConnectivity()
            self.connectivity_table.add_connectivity(
                0, 1, 37, 3, 0, True, 0.75, 1.0, 0.75, 1.0
            )
            self.connectivity_table.add_connectivity(
                1, 2, 81, 3, 1, False, -1.0, -1.0, -1.0, -1.0
            )
            saved = StateData(
                deepcopy(system), sim.atomic_environment, sim.neighbors_list
            )

            saved.system.update_positions(SAVED_MIN.copy())

            # Reuse this fixture's prescribed local ordering when the state is rehydrated.
            def rehydrate(_cfg):
                saved.neighbors_list = sim.neighbors_list
                saved.environment = sim.atomic_environment

            saved.ensure_full_state = rehydrate
            self.states = {
                0: StateData(deepcopy(system), None, None),
                1: saved,
                2: StateData(deepcopy(system), None, None),
            }
            self.states[2].system.update_positions(exit_pos.copy())

            def select(_table):
                stage.append("select")
                return (
                    failure()
                    if path == "selection-error"
                    else Ok(BasinSelectorOutput(t_exit=7.0, exit_state=2))
                )

            self.selector = SimpleNamespace(select_from_connectivity=select)

        def construct_connexion_table(self):
            stage.append("explore")
            if path == "refinement-error":
                worker.trace = [(3, 0.0)]
            return failure() if path == "exploration-error" else Ok(None)

    monkeypatch.setattr(loop, "BasinsGenericEvents", PrescribedBasin)
    bad = SAVED_MIN.copy()
    bad[ORDER, 2] += 1.0
    answers = [(initial, BULK), (ordinary, -17.0)]
    if path in {"accepted", "wrong-first-min", "wrong-second-min"}:
        answers.append((bad if path == "wrong-first-min" else SAVED_MIN, BASIN))
        if path != "wrong-first-min":
            answers.append((bad if path == "wrong-second-min" else exit_pos, -35.0))
    calls = []

    def minimize(**kw):
        calls.append(kw["positions"].copy())
        pos, e = answers[len(calls) - 1]
        return pos.copy(), e

    worker.group_minimizer = minimize
    monkeypatch.chdir(tmp_path)
    sim.run()
    assert len(calls) == len(answers)
    assert len(worker.calls) == (2 if path == "exploration-error" else 3)
    if path == "exploration-error":
        assert worker.min_calls == []
    else:
        np.testing.assert_allclose(worker.min_calls, [SAVED_MIN], atol=1e-12, rtol=0)
    assert stage == (
        ["explore"]
        if path in {"exploration-error", "refinement-error"}
        else ["explore", "select"]
    )
    exit_accepted = path == "accepted"
    rate = PREF * np.exp(-DE / (KB * 900.0))
    dt = 7.0 if exit_accepted else -np.log(0.5) / (2 * rate)
    line = sim.loggers.table_line_info_kmc.call_args.args
    assert line[:2] == ("output", 1)
    assert line[2:4] == pytest.approx([dt * 1e-12] * 2, rel=1e-12, abs=0)
    assert line[4:7] == (
        -35.0 if exit_accepted else -17.0,
        DE,
        81 if exit_accepted else 37,
    )
    np.testing.assert_allclose(
        line[7:9],
        [3 * rate, 3 * rate] if exit_accepted else [rate, 2 * rate],
        rtol=1e-12,
        atol=0,
    )
    np.testing.assert_allclose(
        sim.system.positions, exit_pos if exit_accepted else ordinary
    )
    assert sim.loggers.events_basin_info_line.call_count == int(exit_accepted)
    # Verify each group request, even when the wrong endpoint makes it fail.
    frac = sim.config.reconstruction.push_fraction
    for i, submitted in enumerate(calls):
        sym = i // 2
        target = FINAL_X[sym] if i % 2 else np.array([-0.5, 0.25, 1.0])
        expected = (SAVED_MIN if sym else initial).copy()
        expected[ORDER] = coords(
            RAW_X[sym] + frac * (target - RAW_X[sym]),
            wrapped=True,
            y=2.02 - frac * 0.02,
        )
        np.testing.assert_allclose(submitted, expected)
