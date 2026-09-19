"""Rebuild-before-selection consumer oracle for the stationary coupled polynomial.

U(q,r)=(q²−1)²[1+(r−1)²q²]+16(r−1)²(r−2)²
       +(y0²+z0²+y1²+z1²)/2.
q=x0−10, r=x1−10; transverse coordinates are measured from10.
At r=1,2 and q=±1,0 the gradient vanishes. Minimum curvatures are
[8(1+a),1,1,32,1,1], saddle [-4+2a,1,1,32,1,1], a=(r−1)².
Thus changing the noncentral free atom r1→r2 changes nu by sqrt(2).

The completed other event is injected, as in the archived oracle. A labeled
refine_single seam supplies its analytically known current stationary saddle;
it does not claim an actual pARTn search. Refinement.execute, crop/estimate
handoff, ActiveEventTable, actual analytic HTST kernel, site service and BKL
are real. The archived engine config is metadata for this analytic worker;
no LAMMPS calculation or native stationarity claim is made.
"""

from concurrent.futures import Future
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import pykmc
from pykmc.algorithms import rejection_free
from pykmc.config import Config, RateConstantConfig
from pykmc.event_recycling import DistanceRecycling
from pykmc.event_table import ActiveEventTable
from pykmc.htst import compute_event_prefactors
from pykmc.neighbors_list import NeighborsList
from pykmc.physics import resolve_event_constraints
from pykmc.rate_constant import create_rate_constant
from pykmc.rate_constant.prefactors import PrefactorService
from pykmc.refinement import Refinement
from pykmc.result import EventRefinementOutput, Ok
from pykmc.system import System

METHOD = "fd"  # Match the archived analytic callback's kernel method label.
SOURCE_IDS = (17, 23, 91)


def stationary_energy(positions):
    """Independent exact stationarity/energy check at the declared triplet."""
    points = np.asarray(positions)
    q, r = points[0, 0] - 10.0, points[1, 0] - 10.0
    a = (r - 1.0) ** 2
    assert q in (-1.0, 0.0, 1.0) and r in (1.0, 2.0)
    transverse = points[:2, 1:] - 10.0
    gradient_q = 4 * q * (q * q - 1) * (1 + a * q * q) + 2 * a * q * (q * q - 1) ** 2
    gradient_r = 2 * (r - 1) * q * q * (q * q - 1) ** 2 + 32 * (r - 1) * (r - 2) * (
        2 * r - 3
    )
    assert gradient_q == 0.0 and gradient_r == 0.0
    np.testing.assert_array_equal(transverse, np.zeros((2, 2)))
    return (q * q - 1) ** 2 * (1 + a * q * q) + 16 * (r - 1) ** 2 * (r - 2) ** 2


class CoupledWorker:
    """The original analytic Hessian oracle, with actual producing records kept."""

    def __init__(self):
        self.requests = []
        self.results = []

    def submit(self, operation, *, request, compute_backward=True):
        assert operation == "compute_event_prefactors"
        self.requests.append(request)

        def hessian(positions, free):
            assert list(free) == [0, 1]
            q, r = positions[0, 0] - 10, positions[1, 0] - 10
            assert r == 1.0 or r == 2.0
            a = (r - 1) ** 2
            assert q == 0.0 or abs(q) == 1.0
            curvature = -4 + 2 * a if q == 0.0 else 8 * (1 + a)
            return np.diag([curvature, 1.0, 1.0, 32.0, 1.0, 1.0]) / request.masses[0]

        assert stationary_energy(request.min1_positions) == 0.0
        assert stationary_energy(request.saddle_positions) == 1.0
        if compute_backward:
            assert stationary_energy(request.min2_positions) == 0.0
        result = compute_event_prefactors(
            request, hessian, method=METHOD, compute_backward=compute_backward
        )
        self.results.append(result)
        future = Future()
        future.set_result(result)
        return future

    def partn_refine(self, **kwargs):
        raise AssertionError("The declared known-saddle seam must isolate native pARTn")


def setup():
    root = Path(pykmc.__file__).resolve().parent.parent
    cfg = Config.from_ini_file(str(root / "tests/data/input.in"))
    cfg = cfg.model_copy(
        update={
            "rateconstant": RateConstantConfig(style="htst", k0=1.0, free_radius=6.0),
            "control": cfg.control.model_copy(update={"reference_table": None}),
            "atomicenvironment": cfg.atomicenvironment.model_copy(
                update={"rnei": 0.4, "rcut": 0.5}
            ),
        }
    )
    system = System(
        types=["Si"] * 3,
        positions=np.array([[9.0, 10.0, 10.0], [11.0, 10.0, 10.0], [30.0, 10.0, 10.0]]),
        cell=100 * np.eye(3),
        pbc=[False] * 3,
        index=np.array(SOURCE_IDS),
    )
    manager = CoupledWorker()
    svc = PrefactorService(
        cfg,
        manager,
        create_rate_constant(cfg.rateconstant),
        species_masses=(("Si",), (28.0855,)),
        method=METHOD,
    )
    return cfg, system, manager, svc


def full_saddle(system):
    points = system.positions.copy()
    points[0, 0] = 10.0
    return points


def constraints_for(cfg, system):
    return resolve_event_constraints(
        cfg, system.positions, system.types, system.cell, system.pbc, 0, system.index
    )


def add_candidate(table, system):
    saddle = full_saddle(system)
    output = EventRefinementOutput(
        central_atom_index=0,
        saddle_positions=saddle[[0]],
        E_saddle=1.0,
        min2_positions=np.array([[11.0, 10.0, 10.0]]),
        dE_forward=1.0,
        num_reference_event=47,
        refined="T",
        nu0_status="pending",
        full_saddle_positions=saddle,
        constraints=constraints_for(table.config, system),
    )
    # Assignment also permits the pre-repair dataclass to reach the scientific
    # stale-site assertion rather than failing with an unexpected keyword.
    output.crop_atom_ids = (int(system.index[0]),)
    table.add_events(output)


def initial_active():
    cfg, system, manager, svc = setup()
    active = ActiveEventTable(
        cfg, recycler=DistanceRecycling(0.02, 10.0), prefactor_service=svc
    )
    add_candidate(active, system)
    executed = EventRefinementOutput(
        central_atom_index=2,
        saddle_positions=np.array([[30.05, 10.0, 10.0]]),
        E_saddle=0.1,
        min2_positions=np.array([[30.1, 10.0, 10.0]]),
        dE_forward=0.1,
        num_reference_event=101,
        refined="F",
        nu0_status="pending",
    )
    executed.crop_atom_ids = (int(system.index[2]),)
    active.add_events(executed)
    neighbors = NeighborsList(system, 0.4, 0.5)
    assert neighbors.get_neighbors("rcut", 0) == [0]
    source = system.positions.copy()
    assert active.request_site_prefactors(system, neighbors) == {
        "attempted": 1,
        "ok": 1,
        "rejected": 0,
        "no_geometry": 0,
    }
    np.testing.assert_array_equal(system.positions, source)
    assert manager.results[0].forward.ok
    return cfg, system, manager, svc, active, active.table.iloc[0].copy()


def assert_no_stale_site(active):
    """Drop or explicit invalidation is allowed, but not selectable old site ν."""
    if len(active.table):
        stale = (active.table.nu0_source == "site") & (active.table.nu0_status == "ok")
        assert not stale.any(), (
            "Changed full source must invalidate old accepted site physics before selection"
        )
    assert (0, 47) not in active.existing_pairs(), (
        "An invalid site cannot suppress the ordinary refinement dispatcher"
    )


def execute_known_current_refinement(
    active, system, manager, *, omit_full_geometry=False
):
    """Run real execute; only the native/registration producer is a declared seam."""
    cfg = active.config
    neighbors = NeighborsList(system, 0.4, 0.5)
    assert neighbors.get_neighbors("rcut", 0) == [0]
    before = system.positions.copy()
    calls = []
    reference = pd.DataFrame(
        [
            {
                "idx_ref": 47,
                "event_id": "candidate",
                "sym_matrix": [np.eye(3)],
                "k": create_rate_constant(cfg.rateconstant).compute_rate(1.0).rate,
                "energy_barrier": 1.0,
                "nu0": np.nan,
                "nu0_status": "pending",
                "nu0_reason": "no accepted reference estimate",
            }
        ]
    )
    refinement = Refinement(
        cfg,
        SimpleNamespace(info=lambda *a: None, progress_bar=lambda *a: None),
        system,
        neighbors,
        SimpleNamespace(get_atoms_with_id=lambda event: [0]),
        manager,
    )

    def known_stationary_saddle(
        self, at_idx, dfevent, total_energy, future_context, e_thr
    ):
        assert at_idx == 0 and int(dfevent.idx_ref) == 47 and total_energy == 0.0
        calls.append((at_idx, int(dfevent.idx_ref)))
        saddle = full_saddle(self.system)
        assert stationary_energy(self.system.positions) == 0.0
        assert stationary_energy(saddle) == 1.0
        final = self.system.positions.copy()
        final[0, 0] = 11.0
        assert stationary_energy(final) == 0.0
        selected = self.neighbors_list.get_neighbors("rcut", at_idx).copy()
        future = Future()
        future.set_result(
            Ok(
                EventRefinementOutput(
                    central_atom_index=at_idx,
                    saddle_positions=saddle.copy(),
                    E_saddle=1.0,
                    refined="T",
                    constraints=constraints_for(self.config, self.system),
                )
            )
        )
        future_context[future] = {
            "min2_positions": final[selected].copy(),
            "num_reference_event": 47,
            "reference_energy_barrier": 1.0,
            "neighbors": selected.copy(),
            "estimate": self._inherited_estimate(dfevent),
        }
        return future

    refinement.refine_single = MethodType(known_stationary_saddle, refinement)
    refinement.execute(
        reference, total_energy=0.0, existing_pairs=active.existing_pairs()
    )
    assert calls == [(0, 47)], (
        "Invalidated pair must be rebuilt through real Refinement.execute"
    )
    outputs = refinement.get_successes_results()
    assert len(outputs) == 1
    result = outputs[0]
    np.testing.assert_array_equal(result.full_saddle_positions, full_saddle(system))
    np.testing.assert_array_equal(result.saddle_positions, full_saddle(system)[[0]])
    assert result.dE_forward == 1.0 and result.num_reference_event == 47
    assert getattr(result, "crop_atom_ids", None) == (SOURCE_IDS[0],), (
        "Real dispatcher must retain source atom IDs in crop order"
    )
    if omit_full_geometry:
        # Explicit information-loss fault at the producer boundary: never use
        # the crop alone to manufacture the absent full stationary source.
        result.full_saddle_positions = None
    active.add_events(outputs)
    np.testing.assert_array_equal(system.positions, before)
    return neighbors, result


def site_record(active, label):
    getter = getattr(active, "site_calculation", None)
    assert callable(getter), (
        "ActiveEventTable needs site_calculation(row_label) for the actual immutable producer"
    )
    return getter(label)


def assert_current_site(active, system, worker):
    assert len(active.table) == 1, (
        "Rebuilding cannot leave a second stale selectable copy"
    )
    row = active.table.iloc[0]
    assert row.nu0_status == "ok" and row.nu0_source == "site"
    assert row.k_prefactor != active.config.rateconstant.k0
    assert bool(row.nu0_site_attempted)
    assert tuple(row.crop_atom_ids) == (SOURCE_IDS[0],)
    record = site_record(active, active.table.index[0])
    assert record is not None and record.direction == "forward" and record.estimate.ok
    assert record == worker.results[-1].calculation("forward")
    record.validate()
    assert record.provenance.method == METHOD
    assert record.provenance.free_indices == (0, 1)
    assert record.provenance.zone_indices == (0, 1, 2)
    assert record.identity.free_ids == SOURCE_IDS[:2]
    assert record.estimate.nu0_hz == row.nu0
    for snapshot in (record.provenance.source, record.provenance.produced):
        assert snapshot.is_complete
        np.testing.assert_array_equal(snapshot.min1_positions, system.positions)
        np.testing.assert_array_equal(snapshot.saddle_positions, full_saddle(system))
        assert snapshot.constraints.source_ids == SOURCE_IDS
        assert snapshot.constraints.atom_ids == SOURCE_IDS
        assert snapshot.types == ("Si",) * 3 and snapshot.masses == (28.0855,)
        assert snapshot.pbc == (False, False, False)
        np.testing.assert_array_equal(snapshot.cell, system.cell)
        assert (
            snapshot.descriptor.descriptor_id
            == active.prefactor_service.descriptor_for(system.types).descriptor_id
        )
    with pytest.raises((FrozenInstanceError, AttributeError)):
        record.direction = "backward"
    assert not np.shares_memory(worker.requests[-1].min1_positions, system.positions)
    return row.copy(), record


def recycle_and_fresh(changed_neighbor):
    cfg, system, manager, svc, active, old_row = initial_active()
    old_record = manager.results[0].calculation("forward")
    before = system.positions.copy()
    system.positions[2, 0] += 0.1
    if changed_neighbor:
        system.positions[1, 0] = 12.0
    advanced = system.positions.copy()
    active.prune_for_recycling(executed_idx=1, system=system, positions_pre=before)
    # Remote-only motion may conservatively trigger the same full rebuild;
    # equality of its final spectrum is the unchanged-local numerical control.
    assert_no_stale_site(active)
    svc.reset_step_counters()
    neighbors, _ = execute_known_current_refinement(active, system, manager)
    summary = active.request_site_prefactors(system, neighbors)
    assert summary == {"attempted": 1, "ok": 1, "rejected": 0, "no_geometry": 0}
    assert svc.step_requests == 1 and len(manager.requests) == 2
    rebuilt, current_record = assert_current_site(active, system, manager)
    assert current_record.calculation_id != old_record.calculation_id
    np.testing.assert_array_equal(old_record.provenance.source.min1_positions, before)

    # A fresh table follows the identical dispatcher and actual current full
    # source, not a manually patched scalar frequency or crop-only substitute.
    fresh = ActiveEventTable(cfg, prefactor_service=svc)
    fresh_neighbors, _ = execute_known_current_refinement(fresh, system, manager)
    assert fresh.request_site_prefactors(system, fresh_neighbors)["ok"] == 1
    fresh_row, fresh_record = assert_current_site(fresh, system, manager)
    assert fresh_record.calculation_id == current_record.calculation_id
    np.testing.assert_array_equal(system.positions, advanced)
    return rebuilt, fresh_row, old_row


def draw_clock(row, monkeypatch):
    draws = iter([0.5, 0.5])
    monkeypatch.setattr("pykmc.algorithms.random.random", lambda: next(draws))
    selected, dt, total = rejection_free([float(row.k)])
    assert selected == 0
    assert total == pytest.approx(float(row.k), rel=1e-12, abs=0.0)
    assert dt * total == pytest.approx(np.log(2.0), rel=1e-12, abs=0.0)
    return dt


def test_same_local_geometry_recycles_the_correct_site_value():
    recycled, fresh, old = recycle_and_fresh(False)
    assert recycled.nu0 == pytest.approx(old.nu0)
    assert recycled.nu0 == pytest.approx(fresh.nu0, rel=1e-12)
    assert recycled.k == pytest.approx(fresh.k, rel=1e-12, abs=0.0)


def test_changed_noncentral_free_atom_cannot_keep_stale_site_rate(monkeypatch):
    recycled, fresh, old = recycle_and_fresh(True)
    assert fresh.nu0 / old.nu0 == pytest.approx(np.sqrt(2), rel=1e-12)
    assert recycled.nu0 == pytest.approx(fresh.nu0, rel=1e-12, abs=0.0)
    assert recycled.nu0 != pytest.approx(old.nu0, rel=1e-12, abs=0.0)
    clocks = [draw_clock(row, monkeypatch) for row in [recycled, fresh]]
    print(
        {
            "old_nu0_hz": old.nu0,
            "recycled_nu0_hz": recycled.nu0,
            "fresh_nu0_hz": fresh.nu0,
            "recycled_status": recycled.nu0_status,
            "recycled_source": recycled.nu0_source,
            "recycled_site_attempted": bool(recycled.nu0_site_attempted),
            "clock_ratio": clocks[0] / clocks[1],
            "rate_ratio_fresh_to_recycled": fresh.k / recycled.k,
        }
    )
    assert recycled.k == pytest.approx(fresh.k, rel=1e-12, abs=0.0), (
        "Unmoved central atom does not establish unchanged HTST spectrum of its moved free region"
    )
    assert clocks[0] == pytest.approx(clocks[1], rel=1e-12, abs=0.0)
    assert draw_clock(old, monkeypatch) / clocks[0] == pytest.approx(
        np.sqrt(2), rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize("movement,distance", [(1.0, 10.0), (0.0, 25.0)])
def test_existing_central_movement_and_distance_filters_drop(movement, distance):
    cfg, system, _, svc = setup()
    active = ActiveEventTable(
        cfg, recycler=DistanceRecycling(0.02, distance), prefactor_service=svc
    )
    for atom in [0, 2]:
        active.add_events(
            EventRefinementOutput(
                central_atom_index=atom,
                saddle_positions=system.positions[[atom]],
                E_saddle=1.0,
                dE_forward=1.0,
                num_reference_event=atom,
                refined="F",
                nu0_status="pending",
            )
        )
    before = system.positions.copy()
    system.positions[0, 0] += movement
    active.prune_for_recycling(executed_idx=1, system=system, positions_pre=before)
    assert len(active.table) == 0


def test_invalidated_site_without_full_geometry_is_explicit_k0_at_selection(
    monkeypatch,
):
    _, system, manager, svc, active, old = initial_active()
    before = system.positions.copy()
    system.positions[2, 0] += 0.1
    system.positions[1, 0] = 12.0
    active.prune_for_recycling(executed_idx=1, system=system, positions_pre=before)
    assert_no_stale_site(active)
    svc.reset_step_counters()
    neighbors, _ = execute_known_current_refinement(
        active, system, manager, omit_full_geometry=True
    )
    assert active.request_site_prefactors(system, neighbors) == {
        "attempted": 1,
        "ok": 0,
        "rejected": 0,
        "no_geometry": 1,
    }
    assert svc.step_requests == 0 and len(manager.requests) == 1
    assert len(active.table) == 1
    row = active.table.iloc[0]
    assert row.nu0_source == "k0" and row.nu0_status != "ok"
    assert np.isnan(row.nu0) and row.nu0_reason
    assert row.k_prefactor == svc.config.rateconstant.k0 == 1.0
    assert site_record(active, active.table.index[0]) is None
    expected = np.exp(-1.0 / (8.6173303e-5 * svc.config.rateconstant.T))
    assert row.k == pytest.approx(expected, rel=1e-12, abs=0.0)
    assert row.k != pytest.approx(old.k, rel=1e-12, abs=0.0)
    draw_clock(row, monkeypatch)


def test_unchanged_complete_source_reuses_actual_producer_without_new_work(monkeypatch):
    cfg, system, manager, svc, active, old = initial_active()
    before = system.positions.copy()
    original_record = site_record(active, 0)
    assert original_record == manager.results[0].calculation("forward")
    # This dedicated reuse control omits even the distant executed-atom motion.
    active.prune_for_recycling(executed_idx=1, system=system, positions_pre=before)
    assert active.existing_pairs() == {(0, 47)}
    assert len(active.table) == 1
    neighbors = NeighborsList(system, 0.4, 0.5)
    reference = pd.DataFrame(
        [
            {
                "idx_ref": 47,
                "event_id": "candidate",
                "sym_matrix": [np.eye(3)],
                "k": old.k,
                "energy_barrier": 1.0,
                "nu0": np.nan,
                "nu0_status": "pending",
                "nu0_reason": "no accepted reference estimate",
            }
        ]
    )
    refinement = Refinement(
        cfg,
        SimpleNamespace(info=lambda *a: None, progress_bar=lambda *a: None),
        system,
        neighbors,
        SimpleNamespace(get_atoms_with_id=lambda event: [0]),
        manager,
    )
    svc.reset_step_counters()
    refinement.execute(
        reference, total_energy=0.0, existing_pairs=active.existing_pairs()
    )
    assert refinement.get_successes_results() == []
    assert active.request_site_prefactors(system, neighbors) == {
        "attempted": 0,
        "ok": 0,
        "rejected": 0,
        "no_geometry": 0,
    }
    assert svc.step_requests == 0 and len(manager.requests) == 1
    assert site_record(active, 0) == original_record
    assert active.table.iloc[0].nu0 == old.nu0
    assert active.table.iloc[0].k == old.k
    assert active.table.iloc[0].nu0_source == "site"
    np.testing.assert_array_equal(system.positions, before)
    draw_clock(active.table.iloc[0], monkeypatch)
