"""Bounded physical-dependency, correspondence and rate-only site controls.

The frozen analytic consumer helper supplies actual polynomial kernel results.
These tests assert physical behavior through public table/service boundaries;
no SiteState signature or private side-store representation is inspected.
"""

import shlex
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pykmc.config import RegionConfig
from pykmc.event_recycling import DistanceRecycling
from pykmc.event_table import ActiveEventTable
from pykmc.neighbors_list import NeighborsList
from pykmc.physics import ResolvedConstraints
from pykmc.rate_constant import create_rate_constant
from pykmc.rate_constant.prefactors import PrefactorService
from pykmc.result import EventRefinementOutput

from . import test_site_recycling as h


def current_service(cfg, system, *, mass=28.0855):
    worker = h.CoupledWorker()
    user = None
    if cfg.frozen_atoms is not None:
        user = ResolvedConstraints.resolve(
            system.positions,
            system.types,
            cfg.frozen_atoms,
            system.index,
            cell=system.cell,
            pbc=system.pbc,
        )
    service = PrefactorService(
        cfg,
        worker,
        create_rate_constant(cfg.rateconstant),
        species_masses=(("Si",), (mass,)),
        global_constraints=user,
        method="fd",
    )
    return service, worker


def seed_site(cfg, system, svc, *, reference=None):
    """Actual full current request; optional accepted reference is an actual result."""
    table = ActiveEventTable(cfg, prefactor_service=svc)
    neighbors = NeighborsList(
        system, cfg.atomicenvironment.rnei, cfg.atomicenvironment.rcut
    )
    crop = list(neighbors.get_neighbors("rcut", 0))
    saddle = h.full_saddle(system)
    final = system.positions.copy()
    final[0, 0] = 11.0
    estimate = (
        {}
        if reference is None
        else {
            "nu0_hz": reference.estimate.nu0_hz,
            "nu0_status": "ok",
            "nu0_source": "reference",
        }
    )
    if reference is not None:
        assert reference.estimate.ok and reference.provenance.source.is_complete
    output = EventRefinementOutput(
        central_atom_index=0,
        saddle_positions=saddle[crop].copy(),
        E_saddle=1.0,
        min2_positions=final[crop].copy(),
        dE_forward=1.0,
        num_reference_event=47,
        refined="T",
        full_saddle_positions=saddle.copy(),
        constraints=h.constraints_for(cfg, system),
        **estimate,
    )
    output.crop_atom_ids = tuple(int(system.index[i]) for i in crop)
    table.add_events(output)
    summary = table.request_site_prefactors(system, neighbors)
    return table, neighbors, summary


def validator(table):
    method = getattr(table, "validate_recycled", None)
    assert callable(method), (
        "ActiveEventTable must validate physical dependencies before existing_pairs/selection"
    )
    return method


def install_current_authority(table, config, service):
    # Replace the public current authorities together. Do not clear a private
    # cached rate facade: temperature-only correctness is part of the boundary.
    table.config = config
    table.prefactor_service = service


def assert_dropped(table):
    assert len(table.table) == 0
    assert table.existing_pairs() == set(), (
        "Invalid dependencies must not skip the next real refinement"
    )
    assert h.site_record(table, 0) is None


@pytest.mark.parametrize(
    "change", ["mass", "potential-bytes", "fd-step", "global-constraints"]
)
def test_changed_producing_context_invalidates_then_rebuilds(change, tmp_path):
    cfg, system, old_worker, old_svc = h.setup()
    active, neighbors, summary = seed_site(cfg, system, old_svc)
    assert summary["ok"] == 1
    original = h.site_record(active, 0)
    source = system.positions.copy()
    mass = 28.0855
    if change == "mass":
        mass = 40.0
        current_cfg = cfg
    elif change == "fd-step":
        current_cfg = cfg.model_copy(
            update={
                "rateconstant": cfg.rateconstant.model_copy(
                    update={"fd_step": cfg.rateconstant.fd_step * 2}
                )
            }
        )
    elif change == "global-constraints":
        # Freeze only the distant source row: the original free [0,1] Hessian
        # oracle stays unchanged, while actual global user authority changes.
        current_cfg = cfg.model_copy(update={"frozen_atoms": RegionConfig(indices=[2])})
    else:
        tokens = shlex.split(cfg.lammps.pair_coeff)
        original_file = Path(tokens[2])
        if not original_file.is_absolute():
            original_file = (
                Path(h.pykmc.__file__).resolve().parent.parent / original_file
            )
        changed = tmp_path / "changed-dependency.eam"
        changed.write_bytes(original_file.read_bytes() + b"\n# dependency-control\n")
        tokens[2] = str(changed)
        current_cfg = cfg.model_copy(
            update={
                "lammps": cfg.lammps.model_copy(
                    update={"pair_coeff": shlex.join(tokens)}
                )
            }
        )
    current, worker = current_service(current_cfg, system, mass=mass)
    assert current.descriptor_for(system.types).descriptor_id != original.descriptor_id
    install_current_authority(active, current_cfg, current)
    validator(active)(system, neighbors)
    assert_dropped(active)
    assert len(old_worker.requests) == 1 and worker.requests == []

    # Re-enter the actual ordinary dispatcher and site operation with current
    # config, authoritative mass/constraint snapshot and complete current source.
    current_neighbors, _ = h.execute_known_current_refinement(active, system, worker)
    assert active.request_site_prefactors(system, current_neighbors)["ok"] == 1
    assert len(worker.requests) == 1 and active.existing_pairs() == {(0, 47)}
    record = h.site_record(active, 0)
    assert record == worker.results[0].calculation("forward")
    assert record.estimate.ok and active.table.iloc[0].nu0_source == "site"
    assert active.table.iloc[0].k_prefactor != current_cfg.rateconstant.k0
    assert record.descriptor_id == current.descriptor_for(system.types).descriptor_id
    assert record.calculation_id != original.calculation_id
    np.testing.assert_array_equal(record.provenance.source.min1_positions, source)
    np.testing.assert_array_equal(
        record.provenance.source.saddle_positions, h.full_saddle(system)
    )
    assert record.provenance.source.masses == (mass,)
    assert record.provenance.source.settings.fd_step == current_cfg.rateconstant.fd_step
    if change == "global-constraints":
        assert record.provenance.source.user_constraints == current.global_constraints
        assert record.provenance.source.constraints.fixed_ids == (h.SOURCE_IDS[2],)
    expected_ratio = np.sqrt(28.0855 / mass)
    assert record.estimate.nu0_hz / original.estimate.nu0_hz == pytest.approx(
        expected_ratio, rel=1e-12, abs=0.0
    )
    np.testing.assert_array_equal(system.positions, source)


def test_identical_rebuilt_current_authority_keeps_actual_producer():
    cfg, system, old_worker, svc = h.setup()
    active, neighbors, _ = seed_site(cfg, system, svc)
    record = h.site_record(active, 0)
    current, worker = current_service(cfg, system)
    install_current_authority(active, cfg, current)
    validator(active)(system, neighbors)
    assert active.existing_pairs() == {(0, 47)} and len(active.table) == 1
    assert h.site_record(active, 0) == record
    assert active.request_site_prefactors(system, neighbors)["attempted"] == 0
    assert worker.requests == [] and len(old_worker.requests) == 1


def test_temperature_only_rebuilds_rate_without_a_new_hessian(monkeypatch):
    cfg, system, old_worker, svc = h.setup()
    active, neighbors, _ = seed_site(cfg, system, svc)
    old = active.table.iloc[0].copy()
    record = h.site_record(active, 0)
    warm = cfg.model_copy(
        update={"rateconstant": cfg.rateconstant.model_copy(update={"T": 600.0})}
    )
    current, worker = current_service(warm, system)
    assert current.descriptor_for(system.types).descriptor_id == record.descriptor_id
    install_current_authority(active, warm, current)
    validator(active)(system, neighbors)
    assert active.existing_pairs() == {(0, 47)} and len(active.table) == 1
    assert h.site_record(active, 0) == record
    row = active.table.iloc[0]
    assert row.nu0 == old.nu0 and row.k_prefactor == old.k_prefactor
    expected = old.nu0 / 1e12 * np.exp(-1.0 / (8.6173303e-5 * 600.0))
    assert row.k == pytest.approx(expected, rel=1e-12, abs=0.0)
    assert row.k > old.k
    h.draw_clock(row, monkeypatch)
    assert active.request_site_prefactors(system, neighbors)["attempted"] == 0
    assert worker.requests == [] and len(old_worker.requests) == 1


@pytest.mark.parametrize("ordering", ["source-order", "crop-order"])
def test_order_only_changes_preserve_stable_atom_correspondence(ordering):
    cfg, system, worker, svc = h.setup()
    cfg = cfg.model_copy(
        update={
            "atomicenvironment": cfg.atomicenvironment.model_copy(update={"rcut": 3.5})
        }
    )
    svc, worker = current_service(cfg, system)
    active, _, summary = seed_site(cfg, system, svc)
    assert summary["ok"] == 1
    record = h.site_record(active, 0)
    assert tuple(active.table.iloc[0].crop_atom_ids) == h.SOURCE_IDS[:2]
    expected_by_id = {
        int(i): p.copy()
        for i, p in zip(system.index, h.full_saddle(system), strict=True)
    }
    if ordering == "source-order":
        order = np.array([1, 0, 2])
        system.positions = system.positions[order].copy()
        system.types = list(np.asarray(system.types)[order])
        system.index = system.index[order].copy()
    else:
        active.table.at[0, "crop_atom_ids"] = tuple(reversed(h.SOURCE_IDS[:2]))
        for field in ("saddle_positions", "final_positions"):
            active.table.at[0, field] = np.asarray(active.table.at[0, field])[
                ::-1
            ].copy()
    neighbors = NeighborsList(
        system, cfg.atomicenvironment.rnei, cfg.atomicenvironment.rcut
    )
    source = system.positions.copy()
    validator(active)(system, neighbors)
    center = list(system.index).index(h.SOURCE_IDS[0])
    assert active.existing_pairs() == {(center, 47)} and len(active.table) == 1
    assert int(active.table.iloc[0].atom_index) == center
    assert h.site_record(active, 0) == record
    resolver = getattr(active, "crop_indices", None)
    assert callable(resolver), "Stored crop IDs need a current-source index resolver"
    indices = resolver(0, system, neighbors)
    assert tuple(int(system.index[i]) for i in indices) == tuple(
        active.table.iloc[0].crop_atom_ids
    )
    overlaid = system.positions.copy()
    overlaid[indices] = active.table.iloc[0].saddle_positions
    np.testing.assert_array_equal(
        overlaid, np.array([expected_by_id[int(i)] for i in system.index])
    )
    assert active.request_site_prefactors(system, neighbors)["attempted"] == 0
    assert len(worker.requests) == 1
    np.testing.assert_array_equal(system.positions, source)


def test_changed_crop_membership_invalidates_before_pair_skip():
    cfg, system, worker, svc = h.setup()
    active, _, _ = seed_site(cfg, system, svc)
    assert tuple(active.table.iloc[0].crop_atom_ids) == (h.SOURCE_IDS[0],)
    changed = cfg.model_copy(
        update={
            "atomicenvironment": cfg.atomicenvironment.model_copy(update={"rcut": 3.5})
        }
    )
    current, next_worker = current_service(changed, system)
    install_current_authority(active, changed, current)
    neighbors = NeighborsList(
        system, changed.atomicenvironment.rnei, changed.atomicenvironment.rcut
    )
    assert set(neighbors.get_neighbors("rcut", 0)) == {0, 1}
    validator(active)(system, neighbors)
    assert_dropped(active)
    assert next_worker.requests == [] and len(worker.requests) == 1


def test_changed_final_crop_cannot_keep_an_unrelated_site_producer():
    cfg, system, worker, svc = h.setup()
    active, neighbors, _ = seed_site(cfg, system, svc)
    original = h.site_record(active, 0)
    altered = np.array(active.table.at[0, "final_positions"], copy=True)
    altered[0, 1] += 0.125
    active.table.at[0, "final_positions"] = altered
    assert h.site_record(active, 0) is None, (
        "Producer access must reject a changed bound event"
    )
    validator(active)(system, neighbors)
    assert_dropped(active)
    assert len(worker.requests) == 1
    assert original == worker.results[0].calculation("forward")


def test_widened_window_retries_rejected_site_inherited_reference_fallback():
    cfg, system, reference_worker, reference_svc = h.setup()
    reference_table, _, _ = seed_site(cfg, system, reference_svc)
    actual_reference = h.site_record(reference_table, 0)
    assert actual_reference.estimate.ok
    system.positions[1, 0] = 12.0
    narrow = cfg.model_copy(
        update={
            "rateconstant": cfg.rateconstant.model_copy(update={"nu0_max_THz": 10.0})
        }
    )
    current, worker = current_service(narrow, system)
    active, neighbors, summary = seed_site(
        narrow, system, current, reference=actual_reference
    )
    assert summary == {
        "attempted": 1,
        "ok": 0,
        "rejected": 1,
        "no_geometry": 0,
        "identityless": 0,
    }
    fallback = active.table.iloc[0]
    assert fallback.nu0_status == "ok" and fallback.nu0_source == "reference"
    assert fallback.nu0 == actual_reference.estimate.nu0_hz
    rejected = h.site_record(active, 0)
    assert rejected == worker.results[0].calculation("forward")
    assert (
        not rejected.estimate.ok
        and rejected.estimate.reason_code.value == "out_of_window"
    )

    wide = narrow.model_copy(
        update={
            "rateconstant": narrow.rateconstant.model_copy(
                update={"nu0_max_THz": 100.0}
            )
        }
    )
    wide_svc, wide_worker = current_service(wide, system)
    assert wide_svc.descriptor_for(system.types).descriptor_id == rejected.descriptor_id
    install_current_authority(active, wide, wide_svc)
    validator(active)(system, neighbors)
    assert_dropped(active)
    current_neighbors, _ = h.execute_known_current_refinement(
        active, system, wide_worker
    )
    assert active.request_site_prefactors(system, current_neighbors)["ok"] == 1
    row = active.table.iloc[0]
    assert row.nu0_status == "ok" and row.nu0_source == "site"
    assert row.nu0 / actual_reference.estimate.nu0_hz == pytest.approx(
        np.sqrt(2), rel=1e-12, abs=0.0
    )
    assert h.site_record(active, 0) == wide_worker.results[0].calculation("forward")
    assert (
        len(reference_worker.requests)
        == len(worker.requests)
        == len(wide_worker.requests)
        == 1
    )


@pytest.mark.parametrize("periodic,retained", [(False, False), (True, True)])
def test_central_motion_filter_uses_actual_axis_pbc(periodic, retained):
    _, system, worker, _ = h.setup()
    system.pbc = (periodic, False, False)
    before = system.positions.copy()
    system.positions[0, 0] += 100.0
    frame = pd.DataFrame({"atom_index": [0, 2], "num_reference_event": [47, 101]})
    active = SimpleNamespace(table=frame)
    selected = DistanceRecycling(0.02, 10.0).select_recyclable(
        active, 1, system, before
    )
    assert len(selected) == int(retained)
    if retained:
        assert int(selected.iloc[0].atom_index) == 0
    pd.testing.assert_frame_equal(frame, active.table)
    assert worker.requests == []
