"""A stationarity rejection of a site request is visible at run time.

The kernel returns ``NONSTATIONARY_GEOMETRY`` as an ordinary rejection and
the row keeps its inherited estimate (the documented k0/reference fallback).
That fallback must not be silent: the table reports it at WARNING level per
event, naming the atom, the reference and the force tolerance, and keeps a
per-step count of site rejections by reason so a whole run degrading to k0
is visible without post-processing the tables.
"""

import logging
from concurrent.futures import Future

import numpy as np

from pykmc.event_table import ActiveEventTable
from pykmc.htst import compute_event_prefactors
from pykmc.htst.result import PrefactorRejected, PrefactorRejection
from pykmc.neighbors_list import NeighborsList
from pykmc.rate_constant import create_rate_constant
from pykmc.rate_constant.prefactors import PrefactorService

from . import test_site_recycling as h


class NonstationaryWorker(h.CoupledWorker):
    """The analytic worker whose native stationarity check rejects the saddle."""

    def submit(self, operation, *, request, compute_backward=True):
        assert operation == "compute_event_prefactors"
        self.requests.append(request)

        def hessian(positions, free):
            raise PrefactorRejected(
                PrefactorRejection.NONSTATIONARY_GEOMETRY,
                "maximum free-atom force norm 0.031 eV/Å exceeds stationarity "
                f"tolerance {request.settings.force_tol} eV/Å",
            )

        result = compute_event_prefactors(
            request, hessian, method=h.METHOD, compute_backward=compute_backward
        )
        self.results.append(result)
        future = Future()
        future.set_result(result)
        return future


def nonstationary_site():
    cfg, system, _, _ = h.setup()
    worker = NonstationaryWorker()
    svc = PrefactorService(
        cfg,
        worker,
        create_rate_constant(cfg.rateconstant),
        species_masses=(("Si",), (28.0855,)),
        method=h.METHOD,
    )
    active = ActiveEventTable(cfg, prefactor_service=svc)
    h.add_candidate(active, system)
    neighbors = NeighborsList(system, 0.4, 0.5)
    return cfg, system, worker, active, neighbors


def test_nonstationary_site_rejection_is_reported_and_counted(caplog):
    cfg, system, worker, active, neighbors = nonstationary_site()
    before = active.table.iloc[0].copy()
    with caplog.at_level(logging.INFO, logger="log"):
        summary = active.request_site_prefactors(system, neighbors)
    assert summary == {"attempted": 1, "ok": 0, "rejected": 1, "no_geometry": 0}
    assert len(worker.requests) == 1
    assert (
        worker.results[0].forward.reason_code
        is PrefactorRejection.NONSTATIONARY_GEOMETRY
    )
    row = active.table.iloc[0]
    # The documented fallback: the inherited estimate stands.
    assert row.nu0_source == before.nu0_source == "k0"
    assert row.k_prefactor == before.k_prefactor == cfg.rateconstant.k0
    assert bool(row.nu0_site_attempted)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "a stationarity rejection must be visible at WARNING level"
    per_event = [m for m in warnings if "atom 0" in m and "reference 47" in m]
    assert per_event, warnings
    assert "nonstationary_geometry" in per_event[0]
    assert str(cfg.rateconstant.force_tol) in per_event[0]
    # Per-step counter by rejection reason, for the step summary.
    assert active.step_site_rejections == {"nonstationary_geometry": 1}
    assert any("1 site" in m and "nonstationary_geometry" in m for m in warnings), (
        "the per-step count must be summarised at WARNING level"
    )


def test_step_site_rejection_counter_resets_each_call():
    cfg, system, worker, active, neighbors = nonstationary_site()
    assert active.step_site_rejections == {}
    active.request_site_prefactors(system, neighbors)
    assert active.step_site_rejections == {"nonstationary_geometry": 1}
    # Nothing eligible on the next call: the counter describes that call only.
    assert active.request_site_prefactors(system, neighbors)["attempted"] == 0
    assert active.step_site_rejections == {}
    np.testing.assert_array_equal(system.positions, h.setup()[1].positions)
