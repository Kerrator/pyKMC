"""Bookkeeping failures on the active-table step path degrade, never abort.

A row that cannot declare stable crop identities is not a crop-only fallback
(contracts section 7d, F1 covers rows with stable identities and no full
saddle): in htst mode reconstruction resolves the stored identities against
the current source, so such a row cannot be reconstructed and selecting it
would purge its reference from the catalogue. It is therefore dropped before
selection at the point where it is reported (WARNING level, naming the row
and the stage), never counted under ``no_geometry`` and never kept behind a
message that says otherwise; its ``(atom, reference)`` pair is re-refined
next step (ACCEPTANCE N06: a dropped row is unavailable before selection). A
configuration error raised by the prefactor service while validating
recycled rows is reported with the affected row's identity and reason rather
than being absorbed silently.
"""

import logging

import numpy as np
import pytest

from pykmc.event_table import ActiveEventTable
from pykmc.neighbors_list import NeighborsList
from pykmc.result import EventRefinementOutput

from . import test_site_dependencies as deps
from . import test_site_recycling as h

_NOTHING = {
    "attempted": 0,
    "ok": 0,
    "rejected": 0,
    "no_geometry": 0,
    "identityless": 1,
}


def identityless_refined_row(cfg, system, svc, *, refined="T"):
    """A ``refined`` row without crop identities and without a full saddle."""
    table = ActiveEventTable(cfg, prefactor_service=svc)
    saddle = h.full_saddle(system)
    table.add_events(
        EventRefinementOutput(
            central_atom_index=0,
            saddle_positions=saddle[[0]],
            E_saddle=1.0,
            min2_positions=np.array([[11.0, 10.0, 10.0]]),
            dE_forward=1.0,
            num_reference_event=47,
            refined=refined,
            nu0_status="pending",
        )
    )
    neighbors = NeighborsList(system, 0.4, 0.5)
    assert neighbors.get_neighbors("rcut", 0) == [0]
    return table, neighbors


def _warning_text(caplog) -> str:
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "the identity failure must be visible at WARNING level"
    return " ".join(r.getMessage() for r in warnings)


def _assert_dropped_truthfully(table, system, neighbors, manager, summary, text):
    # Nothing was attempted for the row and nothing is left to select.
    assert summary == _NOTHING
    assert manager.requests == []
    assert len(table.table) == 0
    assert table._pending_site_rows == set() and table._site_states == {}
    assert "atom 0" in text and "reference 47" in text
    assert "crop" in text
    # The WARNING states what happened, not the opposite.
    assert "dropped before selection" in text
    assert "keeping" not in text
    # KMC.reconstruction() validates again before _select_event: the drop
    # already happened where it was reported, nothing vanishes quietly here.
    assert table.validate_recycled(system, neighbors, allow_pending=False) == 0
    assert len(table.table) == 0


def test_identityless_refined_row_is_dropped_before_selection(caplog):
    # Adapted from the no_geometry-fallback assertion: an identity-less row is
    # not a 7d F1 crop-only row, it is unreconstructable (N06).
    cfg, system, manager, svc = h.setup()
    table, neighbors = identityless_refined_row(cfg, system, svc)
    # Control: the reconstruction path (crop_indices without capture) cannot
    # resolve this row, so a selected row would fail and purge reference 47.
    with pytest.raises(ValueError):
        table.crop_indices(0, system)
    with caplog.at_level(logging.WARNING, logger="log"):
        summary = table.request_site_prefactors(system, neighbors)
    text = _warning_text(caplog)
    assert "site request" in text
    _assert_dropped_truthfully(table, system, neighbors, manager, summary, text)


def test_identityless_pending_approximation_is_dropped_before_selection(caplog):
    """The fallback-context branch (unrefined pending row) drops truthfully too."""
    cfg, system, manager, svc = h.setup()
    table, neighbors = identityless_refined_row(cfg, system, svc, refined="F")
    assert 0 in table._pending_site_rows
    with caplog.at_level(logging.WARNING, logger="log"):
        summary = table.request_site_prefactors(system, neighbors)
    text = _warning_text(caplog)
    assert "fallback context" in text
    _assert_dropped_truthfully(table, system, neighbors, manager, summary, text)


def test_identityless_drop_keeps_the_other_rows_and_their_context():
    """Dropping relabels the survivors; their pending/site stores follow them."""
    cfg, system, manager, svc = h.setup()
    table, neighbors = identityless_refined_row(cfg, system, svc)
    # A second, healthy crop-only row (stable identities, no full saddle):
    # a 7d F1 fallback that keeps its inherited estimate and stays selectable.
    saddle = h.full_saddle(system)
    table.add_events(
        EventRefinementOutput(
            central_atom_index=0,
            saddle_positions=saddle[[0]],
            E_saddle=1.0,
            min2_positions=np.array([[11.0, 10.0, 10.0]]),
            dE_forward=0.5,
            num_reference_event=48,
            refined="T",
            nu0_status="pending",
            crop_atom_ids=(int(system.index[0]),),
        )
    )
    summary = table.request_site_prefactors(system, neighbors)
    assert summary == {
        "attempted": 1,
        "ok": 0,
        "rejected": 0,
        "no_geometry": 1,
        "identityless": 1,
    }
    assert manager.requests == []
    assert len(table.table) == 1
    survivor = table.table.iloc[0]
    assert int(survivor.num_reference_event) == 48
    assert bool(survivor.nu0_site_attempted) and survivor.nu0_source == "k0"
    assert list(table._site_states) == [0], "the survivor's context follows its label"
    assert table.validate_recycled(system, neighbors, allow_pending=False) == 0
    assert len(table.table) == 1 and table.table.iloc[0].k == survivor.k


def test_identityless_pending_row_does_not_abort_duplicate_removal(caplog):
    cfg, system, manager, svc = h.setup()
    table, neighbors = identityless_refined_row(cfg, system, svc)
    with caplog.at_level(logging.WARNING, logger="log"):
        table.remove_duplicates(system.cell, neighbors)
    assert len(table.table) == 1
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "the identity failure must be visible at WARNING level"
    text = " ".join(r.getMessage() for r in warnings)
    assert "atom 0" in text and "reference 47" in text


def test_validate_recycled_reports_service_request_errors(caplog):
    cfg, system, _, svc = h.setup()
    active, neighbors, summary = deps.seed_site(cfg, system, svc)
    assert summary["ok"] == 1
    # A service whose species map cannot describe the source raises
    # HTSTRequestError from descriptor_for while validating the row.
    wrong_species = svc.__class__(
        cfg,
        h.CoupledWorker(),
        svc.rate_constant,
        species_masses=(("Ni",), (58.6934,)),
        method=h.METHOD,
    )
    deps.install_current_authority(active, cfg, wrong_species)
    with caplog.at_level(logging.WARNING, logger="log"):
        dropped = active.validate_recycled(system, neighbors)
    assert dropped == 1
    deps.assert_dropped(active)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "a service configuration error must not be absorbed silently"
    text = " ".join(r.getMessage() for r in warnings)
    assert "reference 47" in text
    assert "species map" in text


def test_validate_recycled_generic_invalidation_stays_quiet(caplog):
    """Ordinary dependency invalidation keeps its info-level count only."""
    cfg, system, _, svc = h.setup()
    active, neighbors, _ = deps.seed_site(cfg, system, svc)
    system.positions[1, 0] = 12.0
    with caplog.at_level(logging.INFO, logger="log"):
        assert active.validate_recycled(system, neighbors) == 1
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_validate_recycled_without_source_identities_warns_and_drops(caplog):
    """A source without stable identities cannot validate recycled rows.

    ``source_index_map`` needs ``system.index``; when it is unavailable the
    conservative fallback applies (drop the recycled rows, re-refine next
    step) with one WARNING naming the cause, never a TypeError out of the KMC
    step.
    """
    cfg, system, manager, svc = h.setup()
    active, neighbors, summary = deps.seed_site(cfg, system, svc)
    assert summary["ok"] == 1 and len(active.table) == 1
    before = system.positions.copy()
    system.index = None
    with caplog.at_level(logging.WARNING, logger="log"):
        dropped = active.validate_recycled(system, neighbors, allow_pending=False)
    assert dropped == 1
    deps.assert_dropped(active)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, "one WARNING for the whole pass, not one per row"
    text = warnings[0].getMessage()
    assert "identities" in text and "dropp" in text
    assert "NoneType" in text or "index" in text, "the cause must be named"
    assert len(manager.requests) == 1
    np.testing.assert_array_equal(system.positions, before)


def test_identityless_row_warns_once_per_step_and_the_duplicate_stage_is_truthful(
    caplog,
):
    """One WARNING per row per step; the duplicate-removal note does not drop.

    ``remove_duplicates`` only reports the row (it cannot compare it by
    identity); the drop happens in ``request_site_prefactors``. The step order
    is dedup then site requests, so the same row must not be reported twice at
    WARNING level and the first note must say what that stage does.
    """
    cfg, system, manager, svc = h.setup()
    table, neighbors = identityless_refined_row(cfg, system, svc)
    with caplog.at_level(logging.INFO, logger="log"):
        table.remove_duplicates(system.cell, neighbors)
        assert len(table.table) == 1, "duplicate removal itself never drops the row"
        summary = table.request_site_prefactors(system, neighbors)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    about_row = [m for m in warnings if "atom 0" in m and "reference 47" in m]
    assert len(about_row) == 1, about_row
    note = about_row[0]
    assert "duplicate removal" in note
    # True for that stage: not compared by identity here, dropped later.
    assert "is dropped before selection" not in note
    assert "not compared by identity" in note
    assert "request_site_prefactors" in note and "before selection" in note
    # The drop still happens, at INFO, where it is done.
    assert summary == _NOTHING and len(table.table) == 0 and manager.requests == []
    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any("dropped" in m and "before selection" in m for m in infos), infos
    # A later step reports afresh: the once-per-step memory does not persist.
    table2, neighbors2 = identityless_refined_row(cfg, system, svc)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="log"):
        table2.request_site_prefactors(system, neighbors2)
    assert any(
        "site request" in r.getMessage()
        and "is dropped before selection" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


def test_validate_recycled_reports_per_row_validation_errors(caplog, monkeypatch):
    """A row dropped by an exception inside its validation is reported.

    ``SiteState.matches``, ``crop_indices`` and the neighbour lookup may raise
    ValueError/TypeError/KeyError/IndexError/RuntimeError; the blanket handler
    keeps the step alive by dropping the row, but a systematic cause (a bug, a
    shape or type drift in the source) would otherwise nullify recycling every
    step with only an aggregate INFO count. Each such drop names the row, the
    exception type and its message at WARNING, like the service-error branch.
    """
    cfg, system, _, svc = h.setup()
    active, neighbors, summary = deps.seed_site(cfg, system, svc)
    assert summary["ok"] == 1 and len(active.table) == 1

    def broken(label, system, neighbors_list=None, *, capture=False):
        raise RuntimeError("synthetic identity drift")

    monkeypatch.setattr(active, "crop_indices", broken)
    with caplog.at_level(logging.INFO, logger="log"):
        dropped = active.validate_recycled(system, neighbors)
    assert dropped == 1
    deps.assert_dropped(active)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in caplog.records]
    text = warnings[0].getMessage()
    assert "atom 0" in text and "reference 47" in text
    assert "RuntimeError" in text and "synthetic identity drift" in text
    assert "dropped" in text
    # The aggregate count still follows.
    assert any(
        "invalidated 1 rows" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.INFO
    )


def test_identityless_drops_are_counted_in_the_step_summary(caplog):
    """Both identity-less branches count what left the table before selection."""
    cfg, system, manager, svc = h.setup()
    table, neighbors = identityless_refined_row(cfg, system, svc)
    with caplog.at_level(logging.WARNING, logger="log"):
        summary = table.request_site_prefactors(system, neighbors)
    assert summary["identityless"] == 1 and summary["attempted"] == 0
    assert len(table.table) == 0
    # The unrefined pending approximation (fallback-context branch) counts too.
    table, neighbors = identityless_refined_row(cfg, system, svc, refined="F")
    summary = table.request_site_prefactors(system, neighbors)
    assert summary["identityless"] == 1 and summary["attempted"] == 0
    # A healthy table reports zero.
    healthy, neighbors, summary = deps.seed_site(cfg, system, svc)
    assert summary["identityless"] == 0 and summary["ok"] == 1
