"""A missing descriptor registry entry requires one actual current calculation."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from . import prefactor_catalogue_helpers as helper


@pytest.mark.parametrize("registry_form", ["absent-entry", "explicit-none"])
def test_current_calculation_repairs_missing_descriptor_once(
    tmp_path, monkeypatch, registry_form
):
    helper.require_schema2()
    potential = tmp_path / "unchanged-physics.json"
    helper.write_potential(potential, a=0.0, scale=1.0, bias=0.3)
    original_service, original_worker = helper.service(potential)
    original = helper.calculate(original_service, "actual-original")
    table = helper.table_for(
        original_service,
        [(17, original, "forward", 47), (47, original, "backward", 17)],
    )
    path = tmp_path / "missing-registry.pkl"
    table.save(str(path))
    assert len(original_worker.calls) == 1
    frame = pd.read_pickle(path)
    geometries = helper.geometric_rows(frame)
    original_records = dict(frame.attrs["calculations"])
    original_descriptor = original.provenance.produced.descriptor
    descriptor_id = original_descriptor.descriptor_id
    assert set(frame.attrs["descriptors"]) == {descriptor_id}
    if registry_form == "absent-entry":
        del frame.attrs["descriptors"][descriptor_id]
    else:
        frame.attrs["descriptors"][descriptor_id] = None
    frame.to_pickle(path)
    input_hash = hashlib.sha256(path.read_bytes()).hexdigest()

    current, worker = helper.service(potential, saved=path)
    loaded = helper.ReferenceEventTable.__new__(helper.ReferenceEventTable)

    def before_submit(request):
        # Missing registry evidence must already be unavailable at dispatch.
        row = helper.row_at(loaded, 17)
        assert row.nu0_status != "ok" and np.isnan(row.nu0)
        assert row.k_prefactor == current.config.rateconstant.k0
        np.testing.assert_array_equal(request.min1_positions, helper.FIRST)
        np.testing.assert_array_equal(request.saddle_positions, helper.SADDLE)
        np.testing.assert_array_equal(request.min2_positions, helper.FINAL)

    worker.before_submit = before_submit
    helper.ReferenceEventTable.__init__(
        loaded, current.config, prefactor_service=current
    )
    assert len(worker.calls) == 1, (
        "one complete current computation serves both directions"
    )
    request, backward, energies, computed = worker.calls[0]
    assert backward is True and energies is True
    assert computed.provenance.energies == pytest.approx((0.04, 1.0, -0.04), rel=1e-14)
    assert len(request.types) == 2 and computed.provenance.source.is_complete
    assert loaded.prefactor_archive.descriptors[descriptor_id] == original_descriptor
    assert all(
        loaded.prefactor_archive.calculations[key] == record
        for key, record in original_records.items()
    )
    helper.assert_geometry_unchanged(loaded.table, geometries)
    for idx, direction in ((17, "forward"), (47, "backward")):
        record = helper.linked_calculation(loaded, idx)
        assert record == computed.calculation(direction)
        actual = helper.active_and_draw(loaded, idx, monkeypatch)
        assert actual.nu0_status == "ok" and actual.nu0_source == "reference"
        assert actual.nu0 == pytest.approx(
            getattr(original, direction).nu0_hz, rel=1e-10
        )
    assert any(
        entry["reference"] is not None
        and entry["reference"].calculation_id
        == original.calculation("forward").calculation_id
        and entry["nu0"] == original.forward.nu0_hz
        and entry["reason"]
        for entry in loaded.prefactor_archive.history[17]
    )
    assert len(worker.calls) == 1
    assert hashlib.sha256(path.read_bytes()).hexdigest() == input_hash

    resaved = tmp_path / "repaired-registry.pkl"
    loaded.save(str(resaved))
    assert len(worker.calls) == 1, "serialization cannot submit additional work"
    again_service, again_worker = helper.service(potential, saved=resaved, forbid=True)
    again = helper.ReferenceEventTable(
        again_service.config, prefactor_service=again_service
    )
    assert not again_worker.calls
    for idx in (17, 47):
        assert helper.active_and_draw(again, idx, monkeypatch).nu0_status == "ok"
        assert helper.linked_calculation(again, idx) == helper.linked_calculation(
            loaded, idx
        )
    assert not again_worker.calls
