"""Pure restart-entry regression: real KMC and downstream barrier arithmetic.

The group backend is an explicit recording fake with two different geometries
and a declared scalar energy function. No native operation is simulated as
scientific evidence. Native saved-frame confirmation remains a separate gate.
"""

from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pykmc.kmc import KMC
from pykmc.refinement import Refinement
from pykmc.result import EventRefinementOutput, Ok
from pykmc.system import System

SAVED = np.array([[1.0, 1.2, 4.0], [2.0, 1.2, 4.0]])
RELAXED = SAVED.copy()
RELAXED[0, 0] = 0.25
CELL = np.diag([8.0, 2.4, 8.0])
PBC = (True, False, True)
BARRIER = 0.5


def energy(positions):
    # Saved=-9.4375 eV; relaxed=-10 eV. This is an independent test function,
    # not a force-field approximation or a pyKMC-generated expected value.
    return -10.0 + (float(positions[0, 0]) - 0.25) ** 2


class RecordingGroup:
    def __init__(self):
        self.live = SAVED.copy()
        self.calls = []

    def group_minimize_with_results(self, *, config, positions=None, types=None):
        self.calls.append(
            ("minimize", None if positions is None else np.array(positions, copy=True))
        )
        if positions is not None:
            self.live = np.array(positions, copy=True)
        self.live = RELAXED.copy()
        return self.live.copy(), energy(self.live)

    def _evaluate(self, operation, positions, recompute):
        self.calls.append(
            (
                operation,
                None if positions is None else np.array(positions, copy=True),
                recompute,
            )
        )
        if positions is not None:
            self.live = np.array(positions, copy=True)
        return energy(self.live)

    def group_get_total_energy(self, positions=None, recompute=True):
        return self._evaluate("total_energy", positions, recompute)

    def group_get_potential_energy(self, positions=None, recompute=True):
        return self._evaluate("potential_energy", positions, recompute)


def simulation(restart_file):
    sim = KMC.__new__(KMC)
    sim.config = SimpleNamespace(
        control=SimpleNamespace(restart_file=restart_file, active_volume=False),
        eventsearch=SimpleNamespace(refined_energy_thr=1e-12),
    )
    sim.system = System(
        types=["Si", "Si"],
        positions=SAVED.copy(),
        cell=CELL.copy(),
        pbc=PBC,
        index=np.array([0, 1]),
    )
    sim.manager = RecordingGroup()
    sim.loggers = SimpleNamespace(
        info=lambda *args: None, progress_bar=lambda *args: None
    )
    sim.global_constraints = object()  # Sentinel authority; entry must not replace it.
    sim.neighbors_list = None
    sim.atomic_environment = SimpleNamespace(get_atoms_with_id=lambda event_id: [0])
    sim.total_energy = None
    sim.potential_energy = None
    return sim


def assert_real_refinement_uses_entry_energy(sim, expected_minimum_energy, monkeypatch):
    """Keep real execute_refinements/Refinement.execute/check_refinement_energy.

    Stub only scheduling/search, supplying a declared absolute saddle value.
    Therefore the actual non-AV consumer must subtract the correct entry energy.
    This is a pure arithmetic/transport control, not native pARTn evidence.
    """
    monkeypatch.setattr(
        Refinement, "get_total_refinements_todo", lambda self, table: (1, 1.0)
    )
    monkeypatch.setattr(
        Refinement, "get_energy_thr_refine", lambda self, table, rate: 1.0
    )
    received_energies = []

    def schedule(self, at_idx, row, total_energy, context, threshold):
        received_energies.append(total_energy)
        future = Future()
        future.set_result(
            Ok(
                EventRefinementOutput(
                    central_atom_index=at_idx,
                    saddle_positions=sim.system.positions.copy(),
                    E_saddle=expected_minimum_energy + BARRIER,
                    refined="T",
                )
            )
        )
        context[future] = dict(
            min2_positions=sim.system.positions.copy(),
            num_reference_event=17,
            reference_energy_barrier=BARRIER,
            neighbors=np.array([0, 1]),
            estimate=dict(
                nu0_hz=None, nu0_status=None, nu0_reason=None, nu0_source=None
            ),
        )
        return [future]

    monkeypatch.setattr(Refinement, "refine_single", schedule)
    table = pd.DataFrame(
        [dict(idx_ref=17, event_id="fixed-test-site", energy_barrier=BARRIER)]
    )
    refinement = sim.execute_refinements(table)
    assert received_energies == [expected_minimum_energy]
    assert len(refinement.results) == 1 and refinement.results[0].is_ok()
    assert refinement.results[0].ok_value().dE_forward == pytest.approx(
        BARRIER, abs=1e-12, rel=0
    )


def test_restart_evaluates_saved_geometry_without_minimization(tmp_path, monkeypatch):
    restart = tmp_path / "restart_2.npz"
    np.savez(restart, last_step=2, last_time=0.125)
    restart_bytes = restart.read_bytes()
    sim = simulation(str(restart))
    before = sim.system.positions.copy()
    authority = sim.global_constraints

    sim.minimize_system()

    # The saved Python and group geometries must name the evaluated state.
    assert sim.total_energy == pytest.approx(-9.4375, abs=1e-12, rel=0)
    assert sim.potential_energy == pytest.approx(-9.4375, abs=1e-12, rel=0)
    np.testing.assert_array_equal(sim.system.positions, before)
    np.testing.assert_array_equal(sim.manager.live, before)
    assert sim.global_constraints is authority
    assert not any(call[0] == "minimize" for call in sim.manager.calls)
    evaluations = [call for call in sim.manager.calls if call[0].endswith("_energy")]
    assert evaluations and any(call[1] is not None for call in evaluations), (
        "explicit saved-source evaluation"
    )
    for _, supplied, _ in evaluations:
        if supplied is not None:
            np.testing.assert_array_equal(supplied, before)
    assert restart.read_bytes() == restart_bytes, (
        "entry energy evaluation does not rewrite saved seconds"
    )
    assert_real_refinement_uses_entry_energy(sim, -9.4375, monkeypatch)


def test_fresh_initialization_adopts_relaxed_geometry_and_matching_energy(monkeypatch):
    sim = simulation(None)
    authority = sim.global_constraints

    sim.minimize_system()

    np.testing.assert_array_equal(sim.system.positions, RELAXED)
    np.testing.assert_array_equal(sim.manager.live, RELAXED)
    assert sim.total_energy == pytest.approx(-10.0, abs=1e-12, rel=0)
    assert sim.potential_energy == pytest.approx(-10.0, abs=1e-12, rel=0)
    assert sim.global_constraints is authority
    assert sum(call[0] == "minimize" for call in sim.manager.calls) == 1
    assert_real_refinement_uses_entry_energy(sim, -10.0, monkeypatch)
