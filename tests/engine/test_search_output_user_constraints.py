"""Search/refine outputs carry the USER constraints only, never the AV mask.

The engine attaches the resolved user constraint set to every successful
``EventSearchOutput`` / ``EventRefinementOutput``; the active-volume shell
(atoms beyond ``rmov``) is a crop restriction that must not shrink the Vineyard
free region of the HTST request built from that output (contracts 7f policy 5).
Result validation likewise checks user rows only: a refined saddle that keeps a
placed shell overlay is a valid constrained event.

Real public wrappers, payload resolution and result validation run; the pARTn
implementation and the full-system restore are recording seams.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from pykmc.config import RateConstantConfig, RegionConfig
from pykmc.engine.lammps import FullSystem, LammpsEngine
from pykmc.htst.free_region import common_free_indices, select_free_indices
from pykmc.physics import ResolvedConstraints, resolve_event_constraints
from pykmc.rate_constant import create_rate_constant
from pykmc.rate_constant.prefactors import PrefactorService
from pykmc.result import EventRefinementOutput, EventSearchOutput, Ok


IDS = (42, 17, 8, 91, 63)
TYPES = ("Ni",) * 5
CELL = np.diag([10.0, 12.0, 14.0])
PBC = (True, False, True)
CENTER = 1  # pARTn central atom, source ID 17
MOVER = 0  # HTST free-region centre, source ID 42
USER_ROW = 2  # source ID 8 lies in the frozen plane y >= 3.3
SHELL_ROW = 4  # source ID 63: 2.0 A from the centre (> rmov), 3.25 A from the mover
FREE_RADIUS = 4.0


def config():
    return SimpleNamespace(
        lammps=SimpleNamespace(
            pair_style="zero 10.0",
            pair_coeff="* *",
            min_style="cg",
            frz_min="0.0 1e-12 100 1000",
            minimize="0.0 1e-12 100 1000",
            verbosity=0,
        ),
        frozen_atoms=RegionConfig(
            region_type="plane", normal="y", side="above", threshold=3.3
        ),
        control=SimpleNamespace(active_volume=True),
        activevolume=SimpleNamespace(rmov=1.6, ract=5.0, AV_debug=False),
        eventsearch=SimpleNamespace(delr_thr=0.1),
        psr=SimpleNamespace(matching_score_thr=1e-7),
        rateconstant=RateConstantConfig(
            style="htst", k0=7, free_radius=FREE_RADIUS, nu0_min_THz=0.01
        ),
    )


def geometries():
    source = np.array(
        [
            [2.25, 3.0, 3.0],
            [3.5, 3.0, 3.0],
            [3.5, 3.5, 3.0],
            [8.0, 3.0, 3.0],
            [5.5, 3.0, 3.0],
        ]
    )
    saddle, final = source.copy(), source.copy()
    saddle[MOVER, 0] += 0.4
    final[MOVER, 0] += 0.8
    return source, saddle, final


def user_snapshot(cfg, source):
    user = ResolvedConstraints.resolve(
        source, TYPES, cfg.frozen_atoms, IDS, cell=CELL, pbc=PBC
    )
    assert user.fixed_ids == (IDS[USER_ROW],)
    return user


def execution_payload(cfg, source, user):
    """The user+AV union a caller resolves from its source before dispatch."""
    execution = resolve_event_constraints(
        cfg, source, TYPES, CELL, PBC, CENTER, IDS, user_constraints=user
    )
    assert set(execution.fixed_ids) == {8, 91, 63}
    return execution


def engine_for(cfg, monkeypatch):
    engine = LammpsEngine(cfg.lammps, comm=None)
    engine.full_system = FullSystem(
        types=TYPES, species=("Ni",), masses=(58.6934,), cell=CELL.copy(), pbc=PBC
    )
    engine._is_orthorhombic = True
    engine.lmp = SimpleNamespace()
    monkeypatch.setattr(engine, "ensure_full_system", lambda positions=None: True)
    return engine


def search_output(triplet):
    return EventSearchOutput(
        central_atom_index=CENTER,
        move_atom_index=MOVER,
        dE_forward=0.2,
        dE_backward=0.2,
        min1_positions=triplet[0].copy(),
        saddle_positions=triplet[1].copy(),
        min2_positions=triplet[2].copy(),
        cell=CELL.copy(),
        types=np.array(TYPES),
    )


def test_search_output_constraints_are_user_only_and_keep_the_free_radius(
    monkeypatch,
):
    cfg = config()
    triplet = geometries()
    user = user_snapshot(cfg, triplet[0])
    engine = engine_for(cfg, monkeypatch)
    seen = []

    def impl(config, center, positions, cell, types, constraints, user_constraints):
        # Transport still carries the full user+AV union into the search.
        assert set(constraints.fixed_ids) == {8, 91, 63}
        seen.append(constraints)
        return Ok(search_output(triplet))

    monkeypatch.setattr(engine, "_partn_search_impl", impl)
    result = engine.partn_search(
        cfg,
        CENTER,
        positions=triplet[0].copy(),
        cell=CELL,
        types=TYPES,
        constraints=execution_payload(cfg, triplet[0], user),
        user_constraints=user,
    )
    assert result.is_ok() and len(seen) == 1
    attached = result.ok_value().constraints
    assert isinstance(attached, ResolvedConstraints)
    assert attached.fixed_ids == (IDS[USER_ROW],)
    assert attached.center_id is None and attached.rmov is None
    assert attached.source_ids == IDS and attached.atom_ids == IDS
    attached.require_preserves(user, cell=CELL, pbc=PBC)

    service = PrefactorService(
        cfg,
        SimpleNamespace(),
        create_rate_constant(cfg.rateconstant),
        species_masses=(("Ni",), (58.6934,)),
        global_constraints=user,
    )
    request = service.build_request(
        event_key=("search", 1),
        min1_positions=triplet[0],
        saddle_positions=triplet[1],
        min2_positions=triplet[2],
        types=TYPES,
        cell=CELL,
        pbc=PBC,
        center_index=MOVER,
        constraints=attached,
    )
    sphere = select_free_indices(triplet[1], MOVER, FREE_RADIUS, CELL, PBC)
    assert SHELL_ROW in sphere and USER_ROW in sphere
    free = common_free_indices(request)
    # The Vineyard free region is free_radius minus the user-frozen atoms only.
    np.testing.assert_array_equal(free, [i for i in sphere if i != USER_ROW])


@pytest.mark.parametrize("moved", ["shell", "user"])
def test_refine_result_validation_checks_user_rows_only(monkeypatch, moved):
    cfg = config()
    source, saddle, _ = geometries()
    user = user_snapshot(cfg, source)
    engine = engine_for(cfg, monkeypatch)
    refined = saddle.copy()
    refined[SHELL_ROW if moved == "shell" else USER_ROW, 0] += 0.2

    def impl(*args, **kwargs):
        return Ok(
            EventRefinementOutput(
                central_atom_index=CENTER,
                saddle_positions=refined.copy(),
                E_saddle=0.2,
                refined="T",
            )
        )

    monkeypatch.setattr(engine, "_partn_refine_impl", impl)
    call = dict(
        config=cfg,
        central_atom_idx=CENTER,
        positions=source.copy(),
        cell=CELL,
        types=TYPES,
        saddle_idx=np.array([MOVER]),
        saddle_positions=saddle[[MOVER]].copy(),
        constraints=execution_payload(cfg, source, user),
        user_constraints=user,
    )
    if moved == "user":
        with pytest.raises(ValueError, match="incompatible constrained event"):
            engine.partn_refine(**call)
        return
    # A placed shell overlay held by setforce is a valid constrained result.
    result = engine.partn_refine(**call)
    assert result.is_ok()
    output = result.ok_value()
    np.testing.assert_array_equal(output.saddle_positions, refined)
    assert output.constraints.fixed_ids == (IDS[USER_ROW],)
    assert output.constraints.center_id is None
