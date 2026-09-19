"""Synthetic typed workers for table transport and scalar-policy unit tests.

These explicit protocol records bind the actual submitted source and selected
rows. Their hand-specified frequencies are unit-test values, not physical
Hessian measurements. Analytic force/Hessian catalogue tests independently
exercise numerical production and reuse.
"""

from dataclasses import replace

import numpy as np
from pykmc.event_table import ReferenceEventTable
from pykmc.htst.free_region import common_free_indices
from pykmc.htst.provenance import CalculationProvenance
from pykmc.htst.result import EventPrefactors
from pykmc.rate_constant import create_rate_constant
from pykmc.rate_constant.prefactors import PrefactorService

from .conftest import FakeManager


def protocol_event_prefactors(request, forward, backward):
    """Return a synthetic typed response with the submitted request's true map."""
    free = tuple(int(i) for i in common_free_indices(request))
    count = len(free)

    def mapped(estimate):
        return replace(
            estimate,
            n_free=count,
            n_positive_min=None if estimate.skipped else 3 * count,
        )

    return EventPrefactors(
        event_key=request.event_key,
        forward=mapped(forward),
        backward=mapped(backward),
        method="fd",
        n_free=count,
        settings=request.settings,
        provenance=CalculationProvenance.capture(
            request,
            request,
            method="fd",
            free_indices=free,
        ),
    )


def protocol_service(config, manager=None):
    """Build a coherent current authority; unexpected recomputation is an error."""
    if manager is None:

        def forbidden(request):
            raise AssertionError("unit fixture must not dispatch a new calculation")

        manager = FakeManager(forbidden)
    return PrefactorService(
        config,
        manager,
        create_rate_constant(config.rateconstant),
        species_masses=(("Ni",), (58.6934,)),
        method="fd",
    )


def protocol_table(config):
    """Attach current authority for HTST reloads, preserving the constant path."""
    rate = create_rate_constant(config.rateconstant)
    service = (
        protocol_service(config) if rate.backend.requires_event_prefactors else None
    )
    return ReferenceEventTable(config, prefactor_service=service)


def protocol_patch(table, idx_ref, estimate, *, incomplete=False):
    """Patch a test row using its explicitly retained full source fixture."""
    if not estimate.ok or not (table.table["idx_ref"] == idx_ref).any():
        table._patch_row(idx_ref, estimate)
        return
    if table.prefactor_service is None:
        table.prefactor_service = protocol_service(table.config)
    source = table._protocol_source
    request = table.prefactor_service.build_request(
        event_key=("synthetic-unit-row", idx_ref),
        min1_positions=np.array(source.positions, copy=True),
        saddle_positions=np.array(source.positions, copy=True),
        min2_positions=np.array(source.positions, copy=True),
        types=source.types,
        cell=source.cell,
        pbc=source.pbc,
        center_index=0,
    )
    if incomplete:
        # The fixture explicitly lacks one original source row. This known
        # partial input can be recorded, but cannot support full recomputation.
        count = len(request.types)
        request = replace(
            request,
            constraints=replace(
                request.constraints, source_ids=tuple(range(count + 1))
            ),
            user_constraints=None,
        )
    result = protocol_event_prefactors(request, estimate, estimate)
    table._patch_row(
        idx_ref,
        result.forward,
        calculation=result.calculation("forward"),
        fresh=True,
    )
    return result


def archived_frequency(table, idx_ref, frequency):
    """Check that an unavailable historical number survives outside rate fields."""
    return any(
        entry["nu0"] == frequency
        for entry in table.prefactor_archive.history.get(idx_ref, ())
    )
