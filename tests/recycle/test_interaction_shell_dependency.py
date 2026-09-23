"""The interaction shell belongs to a recycled site row's dependency region.

The ``changed`` case at unit level: with ``free_radius`` 0.25 Å
the free sphere holds the centre alone, the four anchors sit at r0 = 2^(1/6) Å,
inside the 2.5 Å ``lj/cut`` range but outside the free sphere and the one-row
crop, and the executed centre is 20 Å away so the recycler's distance filter
keeps the row. A fixed atom within the interaction range of a free atom enters
the free-region Hessian through the second derivative of their pair energy, so
moving an anchor changes the spectrum and the row must be invalidated. Motion
beyond the free sphere grown by the interaction range keeps the row (the
``same`` control); a zone crop bounds the region by the zone, the whole set of
atoms the scratch calculation ever saw.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pykmc.config import RateConstantConfig
from pykmc.htst import HTSTSettings
from pykmc.htst.provenance import RequestSnapshot
from pykmc.htst.request import HTSTEventRequest
from pykmc.htst.site_state import (
    DEFAULT_INTERACTION_RANGE,
    DEPENDENCY_POSITION_TOL,
    SiteState,
    dependency_radius,
    source_index_map,
)
from pykmc.physics import ResolvedConstraints
from pykmc.system import System


R0 = 2 ** (1 / 6)
FREE_RADIUS = 0.25
LJ_CUTOFF = 2.5
SOURCE_IDS = (42, 7, 8, 9, 10, 11)
REF = 47
CELL = 100.0 * np.eye(3)
PBC = (False, False, False)


def geometry(a: float, *, remote_shift: float = 0.0) -> np.ndarray:
    """Six-atom LJ case: the centre, four anchors at r0 and a remote atom at 20 Å."""
    b = math.sqrt(R0 * R0 - a * a)
    return (
        np.array(
            [
                [0.0, 0.0, 0.0],
                [b, a, 0.0],
                [b, -a, 0.0],
                [b, 0.0, a],
                [b, 0.0, -a],
                [20.0 + remote_shift, 0.0, 0.0],
            ]
        )
        + 10.0
    )


def source_snapshot(settings: HTSTSettings) -> RequestSnapshot:
    positions = geometry(1.0)
    b = math.sqrt(R0 * R0 - 1.0)
    saddle, final = positions.copy(), positions.copy()
    saddle[0, 0] += b
    final[0, 0] += 2 * b
    types = ("Ni",) * 6
    constraints = ResolvedConstraints.resolve(
        positions, types, None, SOURCE_IDS, cell=CELL, pbc=PBC
    )
    request = HTSTEventRequest(
        event_key=(),
        min1_positions=positions,
        saddle_positions=saddle,
        min2_positions=final,
        types=types,
        species=("Ni",),
        masses=(58.6934,),
        cell=CELL,
        pbc=PBC,
        center_index=0,
        settings=settings,
        constraints=constraints,
        user_constraints=constraints,
    )
    return RequestSnapshot.capture(request)


def site_state(settings: HTSTSettings | None = None, **kwargs) -> SiteState:
    source = source_snapshot(settings or HTSTSettings(free_radius=FREE_RADIUS))
    # The crop is the centre alone (rcut = 0.25 Å).
    signature = (
        SOURCE_IDS[0],
        REF,
        (SOURCE_IDS[0],),
        None,
        None,
        0.2,
        "T",
        "ok",
        "site",
        "2e12",
        True,
    )
    return SiteState(source, "lammps_eskm", signature, **kwargs)


def unchanged(state: SiteState, positions: np.ndarray) -> bool:
    system = System(
        types=np.array(["Ni"] * 6),
        positions=np.asarray(positions, dtype=float),
        cell=CELL.copy(),
        pbc=list(PBC),
        index=np.array(SOURCE_IDS),
    )
    return state.dependency_unchanged(system, source_index_map(system))


def test_default_interaction_range_is_the_config_default() -> None:
    """The site-state default and the rate-constant key default are one value."""
    field = RateConstantConfig.model_fields["interaction_range"]
    assert field.default == DEFAULT_INTERACTION_RANGE == 13.0
    assert (
        RateConstantConfig(style="htst").interaction_range == DEFAULT_INTERACTION_RANGE
    )


def test_frozen_changed_case_anchor_geometry() -> None:
    """The anchors are outside the free sphere and the crop, inside the LJ range."""
    source, moved = geometry(1.0), geometry(0.95, remote_shift=0.1)
    anchor_distance = np.linalg.norm(source[1:5] - source[0], axis=1)
    assert np.allclose(anchor_distance, R0)
    assert FREE_RADIUS < R0 < LJ_CUTOFF
    # a = 1.0 -> 0.95 moves both anchor coordinates: about 0.10 A per anchor.
    shift = np.linalg.norm(moved[1:5] - source[1:5], axis=1)
    assert shift.min() > DEPENDENCY_POSITION_TOL
    assert np.allclose(shift, shift[0]) and 0.05 < shift[0] < 0.15
    np.testing.assert_array_equal(moved[0], source[0])


def test_moved_anchor_inside_the_interaction_range_invalidates_the_site_row() -> None:
    """``changed``: anchors move about 0.10 Å and the centre stays.

    The remote atom is 20 Å away.
    """
    state = site_state()
    assert unchanged(state, geometry(1.0))
    assert not unchanged(state, geometry(0.95, remote_shift=0.1)), (
        "an anchor inside the interaction range of the free centre moved: the "
        "free-region Hessian changed, the recycled site row is stale"
    )


def test_motion_beyond_the_interaction_shell_keeps_the_site_row() -> None:
    """``same`` control: only the remote executed centre, 20 Å away, moves."""
    state = site_state()
    assert unchanged(state, geometry(1.0, remote_shift=0.1))


@pytest.mark.parametrize("interaction_range", [LJ_CUTOFF, 4.0])
def test_dependency_region_is_the_free_sphere_grown_by_the_range(
    interaction_range: float,
) -> None:
    """Without a zone the region is ``free_radius + interaction_range``, inclusive."""
    state = site_state(interaction_range=interaction_range)
    edge = FREE_RADIUS + interaction_range
    assert dependency_radius(state.source.settings, interaction_range) == edge
    assert set(state.dependency_ids) == set(SOURCE_IDS[:5])
    assert not unchanged(state, geometry(0.95))
    entered = geometry(1.0)
    entered[5, 0] = 10.0 + edge - 0.01
    assert not unchanged(state, entered), "an atom entered the dependency region"
    beyond = geometry(1.0)
    beyond[5, 0] = 10.0 + edge + 0.01
    assert unchanged(state, beyond), "motion outside the region keeps the row"


def test_zone_crop_bounds_the_dependency_region_by_the_zone() -> None:
    """A zone-cropped scratch never saw atoms beyond the zone, whatever the range."""
    settings = HTSTSettings(free_radius=FREE_RADIUS, zone_radius=5.0)
    state = site_state(settings, interaction_range=12.0)
    assert dependency_radius(settings, 12.0) == 5.0
    assert set(state.dependency_ids) == set(SOURCE_IDS[:5])
    assert not unchanged(state, geometry(0.95))
    outside_zone = geometry(1.0)
    outside_zone[5, 0] = 16.0
    assert unchanged(state, outside_zone)
    entered_zone = geometry(1.0)
    entered_zone[5, 0] = 14.9
    assert not unchanged(state, entered_zone)
