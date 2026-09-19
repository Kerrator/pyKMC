"""Proven directional merges preserve aliases, producing history and selection.

The declared 5/5.02 THz worker is a transport fixture using the complete actual
submitted request. It is not a numerical Hessian oracle. Full physical mapping,
archive association, save/load, active rate conversion and BKL run in production.
"""

import numpy as np
import pytest

from pykmc.event_table import ReferenceEventTable
from pykmc.rate_constant import create_rate_constant
from pykmc.rate_constant.prefactors import PrefactorService
from tests.lifecycle.conftest import FakeManager, accepted

from .prefactor_catalogue_helpers import active_and_draw
from .test_identity_gate import (
    _insert,
    _links,
    _series,
    _table_with_service,
    _trivial_event,
)


@pytest.mark.parametrize("alias_id", [None, 7], ids=["sole-pair", "incoming-alias"])
def test_proven_merge_preserves_history_high_water_reload_and_bkl(
    htst_config, system_single_type_fcc, tmp_path, monkeypatch, alias_id
):
    table, producer = _table_with_service(
        htst_config, accepted(5.0e12), accepted(5.02e12)
    )
    system = system_single_type_fcc
    event = _trivial_event(system)
    request = table._event_request(event, system.pbc, event_key=(0, 1))
    fwd, bwd = _series(table, system, 0, 0.5, 0.5)
    admission = table._admit_series(fwd, bwd, request=request).ok_value()
    assert len(admission.frame) == 2 and admission.self_reverse_candidate
    table.add(admission.frame)
    assert _links(table) == [(0, 1), (1, 0)]

    result = table.prefactor_service.compute([request])[request.event_key]
    assert len(producer.prefactor_requests) == 1
    assert result.provenance.source.is_complete
    for idx, direction in ((0, "forward"), (1, "backward")):
        table._patch_row(
            idx,
            getattr(result, direction),
            calculation=result.calculation(direction),
            fresh=True,
        )
    backward_record = result.calculation("backward")
    assert backward_record.estimate.nu0_hz == 5.02e12
    if alias_id is not None:
        # This pre-existing alias has the same actual forward source and result;
        # its incoming link points at the row about to be merged by production.
        _insert(table, table.table.iloc[0], alias_id, 1)
        table._patch_row(
            alias_id,
            result.forward,
            calculation=result.calculation("forward"),
            fresh=True,
        )

    assert table.finalize_self_reverse(0, 1, prefactors=result) is True
    links = [(0, 0)] + ([] if alias_id is None else [(alias_id, 0)])
    assert _links(table) == links
    next_id = 2 if alias_id is None else alias_id + 1
    assert table.max_idx_ref() == next_id
    assert table.table.iloc[0].nu0 == 5.0e12  # keep forward, never average
    reference = table.prefactor_archive.references[1]
    assert reference.calculation_id == backward_record.calculation_id
    assert (
        table.prefactor_archive.calculations[reference.calculation_id]
        == backward_record
    )
    history = table.prefactor_archive.history[1]
    assert any(
        entry["reference"] == reference
        and entry["nu0"] == 5.02e12
        and entry["energy_barrier"] == 0.5
        and entry["reason"]
        for entry in history
    )
    assert set(table.table.idx_backward).issubset(set(table.table.idx_ref))

    saved = tmp_path / "proven-merge.pkl"
    table.save(str(saved))
    assert len(producer.prefactor_requests) == 1
    archive_before = table.prefactor_archive.metadata()
    cfg = htst_config.model_copy(
        update={
            "control": htst_config.control.model_copy(
                update={"reference_table": str(saved)}
            )
        }
    )

    def forbidden_calculation(request):
        raise AssertionError(
            "known unchanged physical producer must survive reload without recalculation"
        )

    manager = FakeManager(forbidden_calculation)
    service = PrefactorService(
        cfg, manager, create_rate_constant(cfg.rateconstant), method="fd"
    )
    loaded = ReferenceEventTable(cfg, prefactor_service=service)
    assert _links(loaded) == links
    assert loaded.max_idx_ref() == next_id
    assert loaded.prefactor_archive.metadata() == archive_before
    for idx, _ in links:
        active = active_and_draw(loaded, idx, monkeypatch)
        assert active.nu0_status == "ok" and active.nu0_source == "reference"
        assert active.nu0 == 5.0e12 and active.k_prefactor == 5.0
    assert manager.prefactor_requests == []
    assert set(loaded.table.idx_backward).issubset(set(loaded.table.idx_ref))

    # Appending after reload cannot recycle the removed backward ID or an alias.
    first, second = _series(loaded, system, 0, 1.0, 1.0)
    pending = loaded._admit_series(first, second).ok_value()
    loaded.add(pending.frame)
    np.testing.assert_array_equal(
        loaded.table.idx_ref.to_numpy()[-2:], [next_id, next_id + 1]
    )
    assert loaded.prefactor_archive.references[1] == reference
    assert loaded.prefactor_archive.history[1] == history
    assert manager.prefactor_requests == []
