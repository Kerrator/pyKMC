"""Catalogue provenance, recomputation, migration and active-rate regressions."""

import pickle
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
from pykmc.event_table import ReferenceEventTable
from pykmc.rate_constant import create_rate_constant

from .prefactor_catalogue_helpers import (
    FINAL,
    FIRST,
    MASS_A,
    MASS_B,
    REGISTRY_KEYS,
    SADDLE,
    active_and_draw,
    assert_fallback,
    assert_geometry_unchanged,
    calculate,
    geometric_rows,
    linked_calculation,
    longitudinal_curvature,
    require_schema2,
    row_at,
    service,
    table_for,
    write_potential,
)


def test_mixed_producers_survive_sparse_relabel_and_resave_under_third_service(
    tmp_path, monkeypatch
):
    require_schema2()
    a, b, c = (tmp_path / f"potential-{name}.json" for name in "ABC")
    for path, value in ((a, 0.0), (b, 1.0), (c, 0.5)):
        write_potential(path, a=value)
    sa, wa = service(a)
    sb, wb = service(b)
    sc, wc = service(c, mass=80.0)
    ra, rb = calculate(sa, "A"), calculate(sb, "B")
    h = sa.settings.fd_step
    ratio = np.sqrt((16 + 56 * h * h + 6 * h**4) / (8 + 4 * h * h))
    assert rb.forward.nu0_hz / ra.forward.nu0_hz == pytest.approx(ratio, rel=1e-10)
    records = [
        (17, ra, "forward", 47),
        (47, ra, "backward", 17),
        (61, rb, "forward", 88),
        (88, rb, "backward", 61),
        (700, ra, "forward", None),
        (501, ra, "forward", None),
    ]
    table = table_for(sa, records, unknown=(700, 501))
    table.remove([501])
    table.table = table.table.iloc[::-1].copy()
    table.table.index = [103, 8, 501, 2, 700]
    original_geometry = geometric_rows(table.table)
    old_records = {
        idx: result.calculation(direction)
        for idx, result, direction, _ in records
        if idx in (17, 47, 61, 88)
    }
    original_ids = {idx: record.calculation_id for idx, record in old_records.items()}
    producing_ids = {
        sa.current_descriptor.descriptor_id,
        sb.current_descriptor.descriptor_id,
    }
    assert len(producing_ids | {sc.current_descriptor.descriptor_id}) == 3
    table.config = sc.config
    table.prefactor_service = sc
    table.rate_constant = create_rate_constant(sc.config.rateconstant)
    before = (len(wa.calls), len(wb.calls), len(wc.calls))
    first, second = tmp_path / "mixed-first.pkl", tmp_path / "mixed-second.pkl"
    table.save(str(first))
    table.save(str(second))
    assert (len(wa.calls), len(wb.calls), len(wc.calls)) == before
    for path in (first, second):
        frame = pd.read_pickle(path)
        assert frame.attrs["schema_version"] == 2
        assert frame.attrs["context_role"] == "serialization_policy"
        assert set(frame.attrs["descriptors"]) == producing_ids
        assert_geometry_unchanged(frame, original_geometry)
        for idx, identity in original_ids.items():
            assert frame.attrs["estimate_references"][idx].calculation_id == identity
            stored = frame.attrs["calculations"][identity]
            assert stored == old_records[idx]
            assert len(stored.provenance.source.types) == 2
        assert frame.attrs["estimate_references"].get(700) is None
        assert np.isnan(frame.loc[frame.idx_ref == 700, "nu0"].iloc[0])
    # KMC/Basin supply this real table getter to Refinement, whose inheritance
    # method reads the supplied Series directly. No reference_estimate call is
    # allowed to repair these rows after the getter has returned them.
    live_service, live_worker = service(b)
    table.config = live_service.config
    table.prefactor_service = live_service
    table.rate_constant = create_rate_constant(live_service.config.rateconstant)
    subset = table.has_id_subset_table(["event-17", "event-61", "event-700"])
    assert len(live_worker.calls) == 1
    for _, reference in subset.iterrows():
        idx = int(reference.idx_ref)
        active = active_and_draw(table, idx, monkeypatch, reference_row=reference)
        if idx == 700:
            assert active.nu0_status != "ok" and active.nu0_source == "k0"
            assert np.isnan(active.nu0) and active.nu0_reason
        else:
            assert active.nu0_status == "ok" and active.nu0_source == "reference"
            assert active.nu0 == pytest.approx(rb.forward.nu0_hz, rel=1e-10)
    assert len(live_worker.calls) == 1
    loaded_service, worker = service(b, saved=second)
    loaded = ReferenceEventTable(
        loaded_service.config, prefactor_service=loaded_service
    )
    assert len(worker.calls) == 1, (
        "only A's shared full source needs current B computation"
    )
    assert worker.calls[0][1:3] == (True, True)
    for idx in (17, 47, 61, 88):
        row = active_and_draw(loaded, idx, monkeypatch)
        assert row.nu0_status == "ok"
        assert row.nu0 == pytest.approx(rb.forward.nu0_hz, rel=1e-10)
    assert linked_calculation(loaded, 61).calculation_id == original_ids[61]
    assert linked_calculation(loaded, 88).calculation_id == original_ids[88]
    assert_fallback(loaded, 700, monkeypatch)
    assert len(worker.calls) == 1
    assert_geometry_unchanged(loaded.table, original_geometry)
    assert producing_ids <= set(loaded.prefactor_archive.descriptors)


@pytest.mark.parametrize("missing", ["schema1", "missing", "none"])
def test_unlinked_crop_cannot_acquire_provenance_by_repeated_migration(
    tmp_path, monkeypatch, missing
):
    require_schema2()
    potential, saved = tmp_path / "potential.json", tmp_path / "saved.pkl"
    write_potential(potential)
    svc, _ = service(potential)
    actual = calculate(svc, "original")
    table = table_for(svc, [(700, actual, "forward", None)])
    table.save(str(saved))
    frame = pd.read_pickle(saved)
    original_geometry = geometric_rows(frame)
    original_frequency = float(frame.nu0.iloc[0])
    if missing == "schema1":
        for name in REGISTRY_KEYS:
            frame.attrs.pop(name, None)
        frame.attrs["schema_version"] = 1
        frame.attrs["legacy_note"] = "original unknown producer metadata"
    elif missing == "missing":
        frame.attrs.pop("estimate_references")
    else:
        frame.attrs["estimate_references"][700] = None
    original_calculations = set(frame.attrs.get("calculations", {}))
    frame.to_pickle(saved)
    write_potential(potential, a=1.0)
    path = saved
    for turn, mass in enumerate((MASS_B, MASS_B, MASS_A)):
        current, worker = service(potential, saved=path, mass=mass, forbid=True)
        loaded = ReferenceEventTable(current.config, prefactor_service=current)
        assert not worker.calls
        assert_geometry_unchanged(loaded.table, original_geometry)
        assert_fallback(loaded, 700, monkeypatch)
        archive = loaded.prefactor_archive
        assert archive.references.get(700) is None
        assert set(archive.calculations) == original_calculations
        assert any(entry["nu0"] == original_frequency for entry in archive.history[700])
        if missing == "schema1":
            assert any(
                item.get("legacy_note") == "original unknown producer metadata"
                for item in archive.legacy_metadata
            )
        path = tmp_path / f"resaved-{turn}.pkl"
        loaded.save(str(path))
        assert not worker.calls
        output = pd.read_pickle(path)
        assert output.attrs["schema_version"] == 2
        assert output.attrs["estimate_references"].get(700) is None
        assert np.isnan(output.nu0.iloc[0])


def test_complete_changed_physics_recomputes_both_directions_and_retains_history(
    tmp_path, monkeypatch
):
    require_schema2()
    potential, saved = tmp_path / "same-path.json", tmp_path / "old.pkl"
    old_hash = write_potential(potential, a=0.0, scale=1.0, bias=0.3)
    old_service, _ = service(potential)
    old = calculate(old_service, "old-full-source")
    table = table_for(
        old_service,
        [
            (17, old, "forward", 47),
            (47, old, "backward", 17),
            (700, old, "forward", None),
        ],
        unknown=(700,),
    )
    old_records = {17: old.calculation("forward"), 47: old.calculation("backward")}
    old_barriers = {
        idx: float(row_at(table, idx).energy_barrier) for idx in old_records
    }
    original_geometry = geometric_rows(table.table)
    table.save(str(saved))
    new_hash = write_potential(potential, a=1.0, scale=2.0, bias=0.3)
    assert old_hash != new_hash
    fresh_service, _ = service(potential, mass=MASS_B)
    fresh = calculate(fresh_service, "independent-current")
    for direction, sign in (("forward", -1), ("backward", 1)):
        h = old_service.settings.fd_step
        ratio = np.sqrt(
            2
            * longitudinal_curvature(1, 0.3, sign, h)
            / longitudinal_curvature(0, 0.3, sign, h)
            * MASS_A
            / MASS_B
        )
        assert getattr(fresh, direction).nu0_hz / getattr(
            old, direction
        ).nu0_hz == pytest.approx(ratio, rel=1e-10)
    assert fresh.forward.nu0_hz != fresh.backward.nu0_hz
    current, worker = service(potential, saved=saved, mass=MASS_B)
    # Allocate normally, then expose that same table object to a read-only
    # submit observer before its real initializer eagerly loads the catalogue.
    loaded = ReferenceEventTable.__new__(ReferenceEventTable)

    def before_submit(_request):
        stale = row_at(loaded, 17)
        assert stale.nu0_status != "ok" and np.isnan(stale.nu0)
        assert stale.k_prefactor == pytest.approx(current.config.rateconstant.k0)

    worker.before_submit = before_submit
    ReferenceEventTable.__init__(loaded, current.config, prefactor_service=current)
    assert len(worker.calls) == 1
    request, backwards, energies, recomputed = worker.calls[0]
    assert backwards and energies
    np.testing.assert_array_equal(request.min1_positions, FIRST)
    np.testing.assert_array_equal(request.saddle_positions, SADDLE)
    np.testing.assert_array_equal(request.min2_positions, FINAL)
    assert request.masses == (MASS_B,)
    assert recomputed.provenance.energies == pytest.approx((0.08, 2.0, -0.08))
    assert_geometry_unchanged(loaded.table, original_geometry)
    for idx, direction, barrier in ((17, "forward", 1.92), (47, "backward", 2.08)):
        row = row_at(loaded, idx)
        record = linked_calculation(loaded, idx)
        expected = fresh.calculation(direction)
        assert record.direction == direction
        assert record.calculation_id == expected.calculation_id
        assert record.calculation_id != old_records[idx].calculation_id
        assert record.identity.free_ids == (0,)
        assert record.identity.atom_ids == (0, 1)
        assert record.provenance.method == "fd"
        assert record.provenance.source.is_complete
        assert record.provenance.produced.is_complete
        assert row.energy_barrier == pytest.approx(barrier, rel=1e-14)
        active = active_and_draw(loaded, idx, monkeypatch)
        assert active.nu0_status == "ok" and active.nu0_source == "reference"
        assert active.nu0 == pytest.approx(expected.estimate.nu0_hz, rel=1e-10)
        assert active.k_prefactor == pytest.approx(
            expected.estimate.nu0_hz * 1e-12, rel=1e-13
        )
        assert old_records[idx].calculation_id in loaded.prefactor_archive.calculations
        assert any(
            entry["reference"] is not None
            and entry["reference"].calculation_id == old_records[idx].calculation_id
            and entry["nu0"] == old_records[idx].estimate.nu0_hz
            and entry["energy_barrier"] == old_barriers[idx]
            and entry["reason"]
            for entry in loaded.prefactor_archive.history[idx]
        )
    assert_fallback(loaded, 700, monkeypatch)
    assert len(worker.calls) == 1
    second = tmp_path / "current.pkl"
    loaded.save(str(second))
    again_service, again_worker = service(potential, saved=second, mass=MASS_B)
    again = ReferenceEventTable(again_service.config, prefactor_service=again_service)
    assert not again_worker.calls
    for idx in (17, 47):
        assert (
            linked_calculation(again, idx).calculation_id
            == linked_calculation(loaded, idx).calculation_id
        )
        assert active_and_draw(again, idx, monkeypatch).nu0_status == "ok"
    assert_fallback(again, 700, monkeypatch)
    assert not again_worker.calls


def test_opaque_fresh_result_never_becomes_verified_by_serialization(
    tmp_path, monkeypatch
):
    require_schema2()
    potential, path = tmp_path / "opaque.json", tmp_path / "opaque-first.pkl"
    write_potential(potential, bias=0.3)
    svc, worker = service(potential, opaque=True)
    original = calculate(svc, "actual-opaque")
    assert not svc.current_descriptor.reusable
    assert svc.current_descriptor.engine.force_model.limitation
    assert not original.calculation("forward").reusable
    table = table_for(svc, [(17, original, "forward", None)])
    assert active_and_draw(table, 17, monkeypatch).nu0_status == "ok"
    assert len(worker.calls) == 1
    table.save(str(path))
    assert len(worker.calls) == 1
    original_descriptor = svc.current_descriptor
    for turn in range(2):
        current, observer = service(potential, saved=path, opaque=True)
        loaded = ReferenceEventTable(current.config, prefactor_service=current)
        row = active_and_draw(loaded, 17, monkeypatch)
        if row.nu0_status == "ok":
            assert len(observer.calls) == 1, (
                "accepted opaque reload needs a new actual calculation"
            )
            assert observer.calls[0][1:3] == (True, True)
            assert row.nu0 == pytest.approx(original.forward.nu0_hz, rel=1e-10)
        else:
            assert row.nu0_source == "k0" and np.isnan(row.nu0) and row.nu0_reason
        record = linked_calculation(loaded, 17)
        assert not record.reusable
        restored_descriptor = loaded.prefactor_archive.descriptors[record.descriptor_id]
        assert restored_descriptor == original_descriptor
        assert asdict(restored_descriptor) == asdict(original_descriptor)
        assert restored_descriptor.descriptor_id == original_descriptor.descriptor_id
        assert pickle.loads(pickle.dumps(restored_descriptor)) == original_descriptor
        calls = len(observer.calls)
        path = tmp_path / f"opaque-resaved-{turn}.pkl"
        loaded.save(str(path))
        assert len(observer.calls) == calls
        stored = pd.read_pickle(path)
        assert all(
            not descriptor.reusable
            for descriptor in stored.attrs["descriptors"].values()
        )
