"""Fresh opaque results are current, but never verified recycled/cache physics.

The immutable R09 coupled-polynomial worker provides actual analytic HTST
results. No native backend, source mutation, or guard monkeypatch is used.
"""

import numpy as np
import pytest

from pykmc.event_recycling import DistanceRecycling
from pykmc.event_table import ActiveEventTable
from pykmc.kmc import KMC
from pykmc.neighbors_list import NeighborsList
from pykmc.rate_constant import create_rate_constant
from pykmc.rate_constant.prefactors import PrefactorService
from pykmc.result import EventRefinementOutput


from . import test_site_recycling as h

EXPECTED_NU_HZ = 8.343623398872e12


def current_service(cfg):
    worker = h.CoupledWorker()
    service = PrefactorService(
        cfg,
        worker,
        create_rate_constant(cfg.rateconstant),
        species_masses=(("Si",), (28.0855,)),
        method=h.METHOD,
    )
    return service, worker


def fresh_site():
    cfg, system, _, _ = h.setup()
    cfg = cfg.model_copy(
        update={
            "lammps": cfg.lammps.model_copy(
                update={
                    "pair_style": "polynomial/opaque",
                    "pair_coeff": "* * analytic-coupled-polynomial Si",
                }
            )
        }
    )
    service, worker = current_service(cfg)
    assert not service.current_descriptor.reusable
    assert service.current_descriptor.engine.force_model.limitation
    before = system.positions.copy()
    active = ActiveEventTable(
        cfg, recycler=DistanceRecycling(0.02, 10.0), prefactor_service=service
    )
    h.add_candidate(active, system)
    neighbors = NeighborsList(system, 0.4, 0.5)
    assert active.request_site_prefactors(system, neighbors) == {
        "attempted": 1,
        "ok": 1,
        "rejected": 0,
        "no_geometry": 0,
    }
    row, calculation = h.assert_current_site(active, system, worker)
    assert not calculation.reusable
    assert calculation.provenance.source.descriptor == service.current_descriptor
    assert row.nu0 == pytest.approx(EXPECTED_NU_HZ, rel=1e-12, abs=0.0)
    assert len(worker.requests) == 1
    np.testing.assert_array_equal(system.positions, before)
    sim = KMC(cfg, manager=worker)
    sim.system, sim.neighbors_list, sim.prefactor_service = system, neighbors, service
    return sim, active, service, worker, calculation, before


def test_fresh_opaque_site_survives_real_kmc_selection(monkeypatch):
    sim, active, service, worker, calculation, before = fresh_site()
    draws = iter((0.5, 0.5))
    monkeypatch.setattr("pykmc.algorithms.random.random", lambda: next(draws))
    # This is the real guard followed by the real BKL, not a direct BKL bypass.
    selected, dt, total = sim._select_event(active)
    assert selected == 0 and len(active.table) == 1
    expected_rate = EXPECTED_NU_HZ * 1e-12 * np.exp(-1.0 / (8.6173303e-5 * 300.0))
    assert total == pytest.approx(expected_rate, rel=1e-12, abs=0.0)
    assert dt * total == pytest.approx(np.log(2.0), rel=1e-12, abs=0.0)
    assert active.table.iloc[0].nu0_source == "site"
    assert active.table.iloc[0].nu0_status == "ok"
    assert active.site_calculation(0) == calculation
    assert not active.site_calculation(0).reusable
    assert not service.current_descriptor.reusable
    assert len(worker.requests) == 1
    np.testing.assert_array_equal(sim.system.positions, before)


@pytest.mark.parametrize("boundary", ["new-service", "changed-source"])
def test_fresh_allowance_does_not_cross_changed_context(boundary):
    sim, active, service, worker, calculation, before = fresh_site()
    replacement = None
    if boundary == "new-service":
        other, replacement = current_service(sim.config)
        assert other is not service
        assert other.current_descriptor == service.current_descriptor
        active.prefactor_service = sim.prefactor_service = other
    else:
        # A changed interacting neighbour (inside free_radius) breaks the
        # source binding; same-step freshness cannot waive it. Adapted from a
        # distant-atom move: F06 repair invariant / C1 dependency-region
        # contract, motion outside the dependency region keeps the row.
        sim.system.positions[1, 0] = 12.0
    with pytest.raises(
        ValueError, match="No active events with current physical dependencies"
    ):
        sim._select_event(active)
    assert active.table.empty and active.existing_pairs() == set()
    assert len(worker.requests) == 1 and not calculation.reusable
    if replacement is not None:
        assert replacement.requests == []
        np.testing.assert_array_equal(sim.system.positions, before)
    np.testing.assert_array_equal(calculation.provenance.source.min1_positions, before)


def test_prune_revokes_opaque_freshness_even_with_unchanged_source():
    sim, active, service, worker, calculation, before = fresh_site()
    active.add_events(
        EventRefinementOutput(
            central_atom_index=2,
            saddle_positions=np.array([[30.05, 10.0, 10.0]]),
            E_saddle=0.1,
            min2_positions=np.array([[30.1, 10.0, 10.0]]),
            dE_forward=0.1,
            num_reference_event=101,
            refined="F",
            nu0_status="pending",
            crop_atom_ids=(h.SOURCE_IDS[2],),
        )
    )
    # Prove ordinary distance/movement filtering would retain the candidate.
    geometric = active.recycler.select_recyclable(active, 1, sim.system, before)
    assert list(geometric.num_reference_event.astype(int)) == [47]
    active.prune_for_recycling(1, sim.system, before)
    assert active.table.empty and active.existing_pairs() == set()
    with pytest.raises(
        ValueError, match="No active events with current physical dependencies"
    ):
        sim._select_event(active)
    assert len(worker.requests) == 1
    assert not calculation.reusable and not service.current_descriptor.reusable
    np.testing.assert_array_equal(sim.system.positions, before)
