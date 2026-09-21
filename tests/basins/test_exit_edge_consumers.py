"""A basin exit must carry the sampled physical edge through existing output.

Promoted from the frozen external copy
``repair-execution/R11/consumer-preparation-01/test_exit_edge_consumers_v1.py``
(review directory ``docs/reviews/2026-09-19-htst-overnight``); assertions unchanged.

Completed graphs/states are explicit inputs, not a graph-discovery claim.
Real execute, reordering, absorbing refinement, rate conversion and FPTA run;
PSR and manager futures are recording external-work boundaries. No native
stationarity, pARTn convergence, or new selector-output field is asserted.
"""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pykmc.basins import BasinsGenericEvents, BasinStatesConnectivity, FPTASelector
from pykmc.config import Config, RateConstantConfig
from pykmc.neighbors_list import NeighborsList
from pykmc.result import EventRefinementOutput, Ok, PSROutput
from pykmc.system import System

import pykmc

KB_T = 8.6173303e-5 * 300.0
ONE_RATE_BARRIER = KB_T * math.log(4.0)
THREE_RATE_BARRIER = KB_T * math.log(4.0 / 3.0)
SOURCE_IDS = np.array([42, 8])
ROTATE_X_TO_Y = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def positions(y):
    return np.array([[10.0, y, 10.0], [12.0, y, 10.0]])


def make_system(points):
    return System(
        positions=points.copy(),
        types=np.array(["Si", "Ge"]),
        cell=np.eye(3) * 100.0,
        pbc=(False, False, False),
        index=SOURCE_IDS.copy(),
    )


class DeclaredState:
    """Already-characterized state; only real spatial neighbors are needed."""

    def __init__(self, points):
        self.system = make_system(points)
        self.neighbors_list = None

    def ensure_full_state(self, config):
        if self.neighbors_list is None:
            self.neighbors_list = NeighborsList(self.system, rnei=3.0, rcut=5.0)
        for center in (0, 1):
            assert self.neighbors_list.get_neighbors("rcut", center) == [0, 1]

    def release_heavy_objects(self):
        self.neighbors_list = None


class ObservedFuture:
    def __init__(self, value):
        self.value = value
        self.reads = 0

    def result(self):
        self.reads += 1
        return self.value


def setup_case(monkeypatch, *, shared_absorber):
    source_root = Path(pykmc.__file__).resolve().parent.parent
    config = Config.from_ini_file(str(source_root / "tests/data/input.in"))
    config.control.active_volume = False
    config.control.reference_table = None
    config.frozen_atoms = None
    config.rateconstant = RateConstantConfig(style="constant", k0=4.0, T=300.0)

    second_source = 1 if shared_absorber else 0
    destination = 2 if shared_absorber else 1
    source_y = 11.0 if shared_absorber else 10.0
    state_points = {0: positions(10.0), destination: positions(12.0)}
    if shared_absorber:
        state_points[1] = positions(11.0)
    declared = [
        {
            "reference": 17,
            "source": 0,
            "center": 0,
            "sym": 0,
            "rate": 1.0,
            "barrier": ONE_RATE_BARRIER,
            "generic_saddle": np.array([[10.1, 10.0, 10.0], [12.0, 10.0, 10.0]]),
            "proposed": np.array([[10.1, 10.0, 10.0], [12.0, 10.0, 10.0]]),
            "returned": np.array([[10.1, 10.0, 10.01], [12.0, 10.0, 10.0]]),
        },
        {
            "reference": 42,
            "source": second_source,
            "center": 1,
            "sym": 1,
            "rate": 1.0 if shared_absorber else 3.0,
            "barrier": ONE_RATE_BARRIER if shared_absorber else THREE_RATE_BARRIER,
            # The actual basin symmetry path rotates this +x displacement into +y.
            "generic_saddle": np.array(
                [[10.0, source_y, 10.0], [12.4, source_y, 10.0]]
            ),
            "proposed": np.array(
                [[10.0, source_y, 10.0], [12.0, source_y + 0.4, 10.0]]
            ),
            "returned": np.array(
                [[10.0, source_y, 10.0], [12.0, source_y + 0.4, 10.02]]
            ),
        },
    ]
    reference = SimpleNamespace(
        table=pd.DataFrame(
            [
                {
                    "idx_ref": item["reference"],
                    "initial_positions": state_points[item["source"]].copy(),
                    "saddle_positions": item["generic_saddle"].copy(),
                    "sym_matrix": [np.eye(3), ROTATE_X_TO_Y.copy()],
                    "sym_perm": [np.arange(2), np.arange(2)],
                }
                for item in declared
            ],
            index=[444, 55],
        )
    )
    observations = {"psr": [], "minimum": [], "refinement": [], "futures": []}

    class RecordingPSR:
        def __init__(self, cfg, system, row, neighbors, center):
            item = next(item for item in declared if item["reference"] == row.idx_ref)
            assert cfg is config and center == item["center"]
            np.testing.assert_array_equal(
                system.positions, state_points[item["source"]]
            )
            observations["psr"].append((item["reference"], center))

        def match(self):
            return Ok(
                PSROutput(
                    rotation_matrix=np.eye(3),
                    translation_matrix=np.zeros(3),
                    permutation_matrix=np.arange(2),
                    matching_score=0.0,
                )
            )

    class RecordingManager:
        def get_total_energy(self, *, positions):
            item = declared[len(observations["minimum"])]
            expected = state_points[item["source"]]
            np.testing.assert_array_equal(positions, expected)
            observations["minimum"].append(positions.copy())
            future = ObservedFuture(0.0)
            observations["futures"].append(future)
            return future

        def partn_refine(self, **kwargs):
            item = declared[len(observations["refinement"])]
            assert kwargs["config"] is config
            assert kwargs["central_atom_idx"] == item["center"]
            np.testing.assert_allclose(
                kwargs["positions"], item["proposed"], rtol=0, atol=1e-14
            )
            np.testing.assert_array_equal(kwargs["saddle_idx"], [0, 1])
            assert kwargs["constraints"].fixed_ids == ()
            assert kwargs["user_constraints"].fixed_ids == ()
            observations["refinement"].append(item["reference"])
            future = ObservedFuture(
                Ok(
                    EventRefinementOutput(
                        central_atom_index=item["center"],
                        saddle_positions=item["returned"].copy(),
                        E_saddle=item["barrier"],
                        min2_positions=state_points[destination].copy(),
                        num_reference_event=item["reference"],
                        refined="T",
                    )
                )
            )
            observations["futures"].append(future)
            return future

    class SeededGraphBasin(BasinsGenericEvents):
        def _initialize(self, system):
            # The completed graph is the input boundary. Real construction's
            # empty work queue returns immediately; no discovery is simulated.
            np.testing.assert_array_equal(system.positions, state_points[0])
            self.states = {
                key: DeclaredState(value) for key, value in state_points.items()
            }
            self.states_to_explore = []
            self.selector = FPTASelector()
            self.connectivity_table = BasinStatesConnectivity()
            edges = (
                [(0, 1, 1.0, 101, 0, 0, True), (1, 0, 2.0, 102, 0, 0, True)]
                if shared_absorber
                else []
            )
            edges += [
                (
                    item["source"],
                    destination,
                    item["rate"],
                    item["reference"],
                    item["center"],
                    item["sym"],
                    False,
                )
                for item in declared
            ]
            for source, target, rate, ref, center, sym, transient in edges:
                self.connectivity_table.add_connectivity(
                    state=source,
                    state_connexion=target,
                    event_connexion=ref,
                    central_atom=center,
                    sym=sym,
                    transient=transient,
                    dE_forward=0.0,
                    k_forward=rate,
                    dE_backward=0.0,
                    k_backward=0.0,
                )
            self.connectivity_table.df.index = (
                [91, 3, 104, 8] if shared_absorber else [104, 8]
            )

    monkeypatch.setattr("pykmc.basins.basin.PointSetRegistration", RecordingPSR)
    basin = SeededGraphBasin(config, reference, set(), RecordingManager())
    return basin, make_system(state_points[0]), state_points, declared, observations


def execute_case(monkeypatch, *, shared_absorber, channel_draw):
    basin, source, state_points, declared, observations = setup_case(
        monkeypatch, shared_absorber=shared_absorber
    )
    draws = iter([0.5, channel_draw])
    consumed = []

    def random():
        value = next(draws)
        consumed.append(value)
        return value

    monkeypatch.setattr(np.random, "random", random)
    result = basin.execute(source)
    assert result.is_ok(), result.err_value() if not result.is_ok() else None
    assert consumed == [0.5, channel_draw]
    assert observations["psr"] == [(17, 0), (42, 1)]
    assert observations["refinement"] == [17, 42]
    assert len(observations["minimum"]) == 2
    assert all(future.reads == 1 for future in observations["futures"])
    for item in declared:
        edge = basin.connectivity_table.df.loc[
            basin.connectivity_table.df["event_connexion"] == item["reference"]
        ]
        assert len(edge) == 1
        assert float(edge.iloc[0].dE_forward) == pytest.approx(
            item["barrier"], rel=1e-12, abs=0
        )
        assert float(edge.iloc[0].k_forward) == pytest.approx(
            item["rate"], rel=1e-12, abs=0
        )
    np.testing.assert_array_equal(source.positions, state_points[0])
    np.testing.assert_array_equal(source.index, SOURCE_IDS)
    return basin, result.ok_value(), state_points, declared


def assert_edge_output(output, item, state_points, destination):
    assert output.exit_state == destination
    assert output.from_state == item["source"]
    assert output.num_reference_event == item["reference"]
    assert output.central_atom == item["center"]
    assert output.energy_barrier == pytest.approx(item["barrier"], rel=1e-12, abs=0)
    np.testing.assert_array_equal(output.neighbors, [0, 1])
    np.testing.assert_array_equal(
        output.initial_system_positions, state_points[item["source"]]
    )
    np.testing.assert_array_equal(output.saddle_positions, item["returned"])
    np.testing.assert_array_equal(output.final_positions, state_points[destination])


def test_shared_absorber_execute_returns_instantaneously_selected_source_edge(
    monkeypatch,
):
    basin, output, state_points, declared = execute_case(
        monkeypatch, shared_absorber=True, channel_draw=0.76
    )
    np.testing.assert_allclose(
        basin.selector.M_abs,
        [[2.0, -2.0, 0.0], [-1.0, 3.0, 0.0], [-1.0, -1.0, 0.0]],
        rtol=1e-12,
        atol=0,
    )
    assert output.t_exit == pytest.approx(math.log(2.0), rel=5e-4)
    # At the median time, source0 has conditional weight17/24; .76 chooses source1.
    assert 17 / 24 < 0.76 < 1
    assert_edge_output(output, declared[1], state_points, destination=2)


@pytest.mark.parametrize(
    "draw,chosen", [(0.1, 0), (0.76, 1)], ids=["first-edge", "second-edge"]
)
def test_parallel_exit_execute_keeps_selected_event_symmetry_saddle_and_barrier(
    monkeypatch, draw, chosen
):
    basin, output, state_points, declared = execute_case(
        monkeypatch, shared_absorber=False, channel_draw=draw
    )
    np.testing.assert_allclose(
        basin.selector.M_abs, [[4.0, 0.0], [-4.0, 0.0]], rtol=1e-12, atol=0
    )
    assert output.t_exit == pytest.approx(math.log(2.0) / 4.0, rel=5e-4)
    # Both edges end in state1, but rates1:3 give distinct physical-edge weights.
    assert (draw < 0.25) is (chosen == 0)
    assert_edge_output(output, declared[chosen], state_points, destination=1)
