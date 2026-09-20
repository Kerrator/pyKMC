"""Site requests carry the user constraints only, never the active-volume shell.

Contracts 7f policy 5: the AV outer shell (atoms beyond ``activevolume.rmov``)
is a crop/transport restriction held by ``fix setforce`` during a search, not
a fixed-coordinate contract. An HTST/RPA request built for an active row must
therefore receive the USER constraint set (``config.frozen_atoms``) so the
Vineyard free region stays ``free_radius`` about the mover minus user-fixed
atoms; the shell never shrinks it. This holds on both request paths of
``ActiveEventTable``: the constraints attached by the producer and the
fallback resolution when the output carries none.

The recycled-row binding (``SiteState``) compares the user authority only: a
moved shell atom outside the dependency region keeps the row, a changed user
authority ends it.
"""

from dataclasses import replace

import numpy as np

from pykmc.config import ActiveVolume, RegionConfig
from pykmc.event_table import ActiveEventTable
from pykmc.htst.free_region import common_free_indices, select_free_indices
from pykmc.neighbors_list import NeighborsList
from pykmc.physics import ResolvedConstraints, resolve_event_constraints
from pykmc.rate_constant import create_rate_constant
from pykmc.rate_constant.prefactors import PrefactorService
from pykmc.result import EventRefinementOutput
from pykmc.system import System

from . import test_site_dependencies as deps
from . import test_site_recycling as h

SOURCE_IDS = (17, 23, 91, 5)
CENTER, FREE_NEIGHBOUR, SHELL, USER_FIXED = 0, 1, 2, 3


def av_setup():
    """AV+htst config with one user-frozen atom and one shell-only atom.

    Rows: the centre (x = 9), its free neighbour (x = 11, inside ``free_radius``
    6.0 but beyond ``rmov`` 1.0), a shell-only atom (x = 30) and a user-frozen
    atom (x = 50). The coupled analytic oracle asserts the free set ``[0, 1]``.
    """
    cfg, _, manager, _ = h.setup()
    cfg = cfg.model_copy(
        update={
            "control": cfg.control.model_copy(update={"active_volume": True}),
            "activevolume": ActiveVolume(rmov=1.0, ract=3.0),
            "frozen_atoms": RegionConfig(indices=[USER_FIXED]),
        }
    )
    system = System(
        types=["Si"] * 4,
        positions=np.array(
            [
                [9.0, 10.0, 10.0],
                [11.0, 10.0, 10.0],
                [30.0, 10.0, 10.0],
                [50.0, 10.0, 10.0],
            ]
        ),
        cell=100 * np.eye(3),
        pbc=[False] * 3,
        index=np.array(SOURCE_IDS),
    )
    user = ResolvedConstraints.resolve(
        system.positions,
        system.types,
        cfg.frozen_atoms,
        system.index,
        cell=system.cell,
        pbc=system.pbc,
    )
    assert user.fixed_ids == (SOURCE_IDS[USER_FIXED],)
    svc = PrefactorService(
        cfg,
        manager,
        create_rate_constant(cfg.rateconstant),
        species_masses=(("Si",), (28.0855,)),
        global_constraints=user,
        method=h.METHOD,
    )
    return cfg, system, manager, svc, user


def fallback_candidate(table, system):
    """A refined output with its full saddle but without producer constraints."""
    saddle = h.full_saddle(system)
    output = EventRefinementOutput(
        central_atom_index=CENTER,
        saddle_positions=saddle[[CENTER]],
        E_saddle=1.0,
        min2_positions=np.array([[11.0, 10.0, 10.0]]),
        dE_forward=1.0,
        num_reference_event=47,
        refined="T",
        nu0_status="pending",
        full_saddle_positions=saddle,
        constraints=None,
    )
    output.crop_atom_ids = (SOURCE_IDS[CENTER],)
    table.add_events(output)
    assert table._full_saddle_constraints.get(0) is None, (
        "the fallback path must be taken"
    )
    return saddle


def test_fallback_site_request_carries_the_user_view_only():
    cfg, system, manager, svc, user = av_setup()
    table = ActiveEventTable(cfg, prefactor_service=svc)
    saddle = fallback_candidate(table, system)
    union = resolve_event_constraints(
        cfg,
        system.positions,
        system.types,
        system.cell,
        system.pbc,
        CENTER,
        system.index,
        user_constraints=user,
    )
    # Control: the AV union really does hold the shell (rows 1, 2) and the
    # user atom (row 3); without policy 5 the request would carry it.
    assert union.fixed_ids == tuple(SOURCE_IDS[i] for i in (1, 2, 3))
    assert union.rmov == 1.0 and union.center_id == SOURCE_IDS[CENTER]

    request = table._site_request(0, system, saddle)

    assert request.constraints == union.user_view()
    assert request.constraints.fixed_ids == (SOURCE_IDS[USER_FIXED],)
    assert request.constraints.center_id is None and request.constraints.rmov is None
    assert request.user_constraints == user
    sphere = select_free_indices(
        request.saddle_positions
        if svc.settings.free_region_center == "saddle"
        else request.min1_positions,
        CENTER,
        svc.settings.free_radius,
        request.cell,
        request.pbc,
    )
    expected = [i for i in sphere.tolist() if i != USER_FIXED]
    assert expected == [CENTER, FREE_NEIGHBOUR]
    assert common_free_indices(request).tolist() == expected, (
        "the rmov shell must not shrink the Vineyard free region"
    )
    assert manager.requests == []


def test_av_shell_motion_keeps_the_recycled_site_row_and_user_change_ends_it():
    cfg, system, manager, svc, user = av_setup()
    table = ActiveEventTable(cfg, prefactor_service=svc)
    fallback_candidate(table, system)
    neighbors = NeighborsList(system, 0.4, 0.5)
    assert neighbors.get_neighbors("rcut", CENTER) == [CENTER]
    assert table.request_site_prefactors(system, neighbors) == {
        "attempted": 1,
        "ok": 1,
        "rejected": 0,
        "no_geometry": 0,
    }
    assert len(manager.requests) == 1
    assert manager.requests[0].constraints.fixed_ids == (SOURCE_IDS[USER_FIXED],)
    record = h.site_record(table, 0)
    assert record is not None and record.estimate.ok
    assert record.provenance.free_indices == (CENTER, FREE_NEIGHBOUR)
    assert record.provenance.source.constraints.fixed_ids == (SOURCE_IDS[USER_FIXED],)
    assert record.provenance.source.user_constraints == user
    row = table.table.iloc[0].copy()
    assert row.nu0_source == "site" and row.nu0_status == "ok"

    # The shell-only atom moves: outside the stored crop and the free sphere,
    # so no site dependency changed. Its AV reference coordinate did, which
    # must not count (the shell is not a user constraint).
    system.positions[SHELL, 0] += 0.1
    assert table.validate_recycled(system, neighbors) == 0
    assert len(table.table) == 1 and table.existing_pairs() == {(CENTER, 47)}
    assert h.site_record(table, 0) == record
    assert table.table.iloc[0].nu0 == row.nu0 and table.table.iloc[0].k == row.k

    # Control: a changed user authority under the same policy and physics
    # (a hand-narrowed payload) ends the binding and drops the row.
    narrowed = replace(user, fixed_ids=(), fixed_positions=(), user_fixed_ids=())
    assert narrowed != user and narrowed.user_policy == user.user_policy
    other = PrefactorService(
        cfg,
        h.CoupledWorker(),
        create_rate_constant(cfg.rateconstant),
        species_masses=(("Si",), (28.0855,)),
        global_constraints=narrowed,
        method=h.METHOD,
    )
    assert (
        other.descriptor_for(system.types).descriptor_id
        == svc.descriptor_for(system.types).descriptor_id
    )
    deps.install_current_authority(table, cfg, other)
    assert table.validate_recycled(system, neighbors) == 1
    deps.assert_dropped(table)
    assert len(manager.requests) == 1
