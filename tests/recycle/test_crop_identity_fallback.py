"""Bookkeeping failures on the active-table step path degrade, never abort.

A row that cannot declare stable crop identities is a crop-only row without
a usable dependency context: it keeps its inherited estimate, is counted
under ``no_geometry`` (contracts section 7d, F1) and is reported at WARNING
level, instead of raising through ``KMC.run``. A configuration error raised
by the prefactor service while validating recycled rows is reported with the
affected row's identity and reason rather than being absorbed silently.
"""

import logging

import numpy as np

from pykmc.event_table import ActiveEventTable
from pykmc.neighbors_list import NeighborsList
from pykmc.result import EventRefinementOutput

from . import test_site_dependencies as deps
from . import test_site_recycling as h


def identityless_refined_row(cfg, system, svc):
    """A refined row without crop identities and without a full saddle."""
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
            refined="T",
            nu0_status="pending",
        )
    )
    neighbors = NeighborsList(system, 0.4, 0.5)
    assert neighbors.get_neighbors("rcut", 0) == [0]
    return table, neighbors


def test_identityless_refined_row_is_a_no_geometry_fallback(caplog):
    cfg, system, manager, svc = h.setup()
    table, neighbors = identityless_refined_row(cfg, system, svc)
    before = table.table.iloc[0].copy()
    with caplog.at_level(logging.WARNING, logger="log"):
        summary = table.request_site_prefactors(system, neighbors)
    assert summary == {"attempted": 1, "ok": 0, "rejected": 0, "no_geometry": 1}
    assert manager.requests == []
    assert len(table.table) == 1
    row = table.table.iloc[0]
    assert bool(row.nu0_site_attempted)
    assert row.nu0_source == before.nu0_source == "k0"
    assert row.nu0_status == before.nu0_status
    assert row.k == before.k and row.k_prefactor == before.k_prefactor
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "the identity failure must be visible at WARNING level"
    text = " ".join(r.getMessage() for r in warnings)
    assert "atom 0" in text and "reference 47" in text
    assert "crop" in text


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
