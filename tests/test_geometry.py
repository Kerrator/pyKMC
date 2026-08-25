"""Tests for PBC-aware helpers in pykmc.utils.geometry."""

import numpy as np
import pytest

from pykmc.utils.geometry import (
    event_contained,
    event_movers,
    minimum_image_distance,
    per_atom_displacement,
    reconstruction_matches,
)

CELL = np.diag([10.0, 10.0, 10.0])


def test_minimum_image_distance_no_wrap() -> None:
    """Pair well inside the box: plain Euclidean distance."""
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([4.0, 5.0, 1.0])
    assert minimum_image_distance(a, b, CELL) == pytest.approx(5.0)


def test_minimum_image_distance_wraps_across_boundary() -> None:
    """Pair straddling the boundary: wrapped distance beats the naive one."""
    a = np.array([0.5, 5.0, 5.0])
    b = np.array([9.5, 5.0, 5.0])
    naive = float(np.linalg.norm(b - a))
    wrapped = minimum_image_distance(a, b, CELL)
    assert wrapped == pytest.approx(1.0)
    assert wrapped < naive


def test_minimum_image_distance_matches_per_atom_displacement() -> None:
    """Single-pair helper agrees with the vectorized one on a (1, 3) pair."""
    a = np.array([0.5, 9.7, 2.0])
    b = np.array([9.5, 0.3, 2.4])
    expected = per_atom_displacement(a[None, :].copy(), b[None, :].copy(), CELL)[0]
    assert minimum_image_distance(a, b, CELL) == pytest.approx(float(expected))


def test_push_towards_scalar_pbc_does_not_crash() -> None:
    """Scalar bool pbc must behave like the equivalent per-dimension vector."""
    from pykmc.utils.geometry import push_towards

    current = np.array([[1.0, 1.0, 1.0]])
    target = np.array([[9.5, 1.0, 1.0]])

    for scalar, vector in ((False, np.array([False, False, False])),
                           (True, np.array([True, True, True]))):
        got = push_towards(current.copy(), target.copy(), fraction=0.5, cell=CELL, pbc=scalar)
        ref = push_towards(current.copy(), target.copy(), fraction=0.5, cell=CELL, pbc=vector)
        assert np.allclose(got, ref)


def test_compute_delr_scalar_pbc_does_not_crash() -> None:
    """compute_delr always loops over dimensions, so scalar pbc must normalize."""
    from pykmc.utils.geometry import compute_delr

    pos1 = np.array([[0.5, 5.0, 5.0]])
    pos2 = np.array([[9.5, 5.0, 5.0]])

    # Periodic: minimum image across the boundary -> 1.0
    assert compute_delr(pos1, pos2, CELL, pbc=True) == pytest.approx(1.0)
    # Non-periodic: naive distance -> 9.0
    assert compute_delr(pos1, pos2, CELL, pbc=False) == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# event_movers: adaptive participant set (finding #10 / design decision 1)
# ---------------------------------------------------------------------------
def test_event_movers_keeps_all_participants_above_threshold() -> None:
    """Every atom above matching_thr is a mover, not just the top n_movers.

    Doc scenario: 5 real movers with n_movers=3 -> all 5 tight-checked, so a
    genuine 4th/5th participant can no longer slip through the loose shell bound.
    """
    disp = np.array([1.5, 1.4, 1.3, 1.2, 1.1])
    movers = event_movers(disp, n_movers=3, matching_thr=0.1)
    assert sorted(movers.tolist()) == [0, 1, 2, 3, 4]


def test_event_movers_ignores_static_atoms_when_participants_exist() -> None:
    """A single participant with static neighbours yields only that participant.

    The top-n_movers value is a FLOOR for the sub-threshold case, not a cap that
    drags near-static peripheral atoms into the tight check (preserves the
    peripheral-tolerance behaviour).
    """
    disp = np.array([0.0, 1.0, 0.0])
    assert event_movers(disp, n_movers=3, matching_thr=0.1).tolist() == [1]


def test_event_movers_floor_when_no_participant() -> None:
    """A sub-threshold event keeps the top-n_movers floor (degenerate fallback)."""
    disp = np.array([0.05, 0.02, 0.01])
    assert sorted(event_movers(disp, n_movers=3, matching_thr=0.1).tolist()) == [0, 1, 2]
    # Fewer atoms than the floor -> keep them all, never crash.
    assert sorted(event_movers(np.array([0.05, 0.02]), 3, 0.1).tolist()) == [0, 1]


def test_event_movers_empty_returns_empty_not_valueerror() -> None:
    """An empty displacement array returns an empty index set, not a ValueError."""
    out = event_movers(np.array([]), n_movers=3, matching_thr=0.1)
    assert out.size == 0
    assert out.dtype == int


def test_reconstruction_matches_empty_rejects_gracefully() -> None:
    """Empty discrepancy or empty movers -> graceful (False, inf, inf), no crash."""
    assert reconstruction_matches(np.array([]), np.array([], dtype=int), 0.1, 1.0) == (
        False,
        float("inf"),
        float("inf"),
    )
    assert reconstruction_matches(np.array([0.0, 0.0]), np.array([], dtype=int), 0.1, 1.0)[0] is False


# ---------------------------------------------------------------------------
# event_contained: whole-path guard (finding #7 / design decision 2)
# ---------------------------------------------------------------------------
# A generously large box so the radii used in the containment tests stay well
# under half the box length and are never minimum-image-wrapped (the containment
# math shares minimum_image_distance with the acceptance metric).
BIG_CELL = np.diag([40.0, 40.0, 40.0])


def test_event_contained_inward_event_is_contained() -> None:
    """A mover well inside rcut at every step is contained."""
    min1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    saddle = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    min2 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    contained, r, limit = event_contained(
        0, [0, 1], np.array([1]), min1, saddle, min2, BIG_CELL, rcut=6.5, containment_margin=1.0
    )
    assert contained is True
    assert r == pytest.approx(2.0)  # max over the path is the min2 radius
    assert limit == pytest.approx(5.5)


def test_event_contained_outward_event_trips_on_min2() -> None:
    """A mover inside rcut-margin at min1 but past it at min2 is NOT contained.

    Measuring min1 alone would pass; the whole-path max catches the outward
    excursion (finding #7).
    """
    limit_r = 6.5 - 1.0  # 5.5
    min1 = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])  # inside
    saddle = np.array([[0.0, 0.0, 0.0], [5.5, 0.0, 0.0]])
    min2 = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]])  # past the limit
    contained, r, limit = event_contained(
        0, [0, 1], np.array([1]), min1, saddle, min2, BIG_CELL, rcut=6.5, containment_margin=1.0
    )
    assert contained is False
    assert r > limit_r
    assert r == pytest.approx(6.0)  # whole-path max is the min2 radius
    # min1-only would have passed (5.0 <= 5.5); the guard now sees the min2 excursion.
    assert minimum_image_distance(min1[0], min1[1], BIG_CELL) <= limit_r


def test_event_contained_absent_central_row_rejects() -> None:
    """A central id missing from neighbours fails closed (not-contained), not skip.

    The guard is the only geometric sanity check; a corrupted/permuted neighbours
    column that dropped the central id must reject rather than bypass the check
    (finding #7 fail-open hole).
    """
    p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    contained, r, _ = event_contained(
        99, [0, 1], np.array([1]), p, p, p, CELL, rcut=6.5, containment_margin=1.0
    )
    assert contained is False
    assert r == float("inf")


def test_event_contained_none_central_disabled() -> None:
    """central_atom=None disables the guard (historical no-op)."""
    p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    contained, r, _ = event_contained(
        None, [0, 1], np.array([1]), p, p, p, CELL, rcut=6.5, containment_margin=1.0
    )
    assert contained is True
    assert r == pytest.approx(0.0)


def test_event_contained_none_central_with_rcut_none_no_typeerror() -> None:
    """Disabled guard must not touch rcut: central_atom=None with rcut=None is a no-op.

    ``atomicenvironment.rcut`` is Optional; when the guard is disabled
    (``central_atom is None``) the helper must return cleanly without computing
    ``float(rcut)``, which would raise TypeError on ``None``.
    """
    p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    contained, r, limit = event_contained(
        None, [0, 1], np.array([1]), p, p, p, CELL, rcut=None, containment_margin=1.0
    )
    assert contained is True
    assert r == pytest.approx(0.0)
    assert limit == pytest.approx(0.0)


def test_event_contained_empty_movers_rejects() -> None:
    """Empty movers -> not contained (graceful), never a max()-over-empty crash."""
    p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    contained, r, _ = event_contained(
        0, [0, 1], np.array([], dtype=int), p, p, p, CELL, rcut=6.5, containment_margin=1.0
    )
    assert contained is False
    assert r == float("inf")


# ---------------------------------------------------------------------------
# adaptive_push_fraction: the reconstruction push rule (residual exit misses)
# ---------------------------------------------------------------------------
# Geometry taken verbatim from the two basin exit-failure dumps of the
# 2026-08-24 alloy campaign (analysis/push_geom.py):
#   nicr_global/basin_exit_fail_171.npz -- mover atom 5458, saddle at 0.59 of
#       the hop from min1; the 0.15 push toward min1 lands at 0.50 of the hop
#       -> RECONSTRUCTION_INVALID_MIN1 (step 51 is the same channel/geometry).
#   nife_global/basin_exit_fail_29.npz  -- mover atom 6377, saddle at 0.43 of
#       the hop from min1; the 0.15 push toward min2 lands at 0.49 of the hop
#       from min2 -> RECONSTRUCTION_INVALID_MIN2.
# Only the five rcut-shell rows with the largest min1->min2 displacement are
# kept: row 0 is the event mover (2.56 A hop), rows 1-4 are representative
# shell atoms whose own hop is <= 0.035 A. That degeneracy (min1 ~= min2 for
# the shell) is why the rule must scale the *saddle->minimum* push rather than
# place atoms at a fixed fraction of the min1->min2 segment. Literals are the
# dump coordinates to 6 d.p. so the test does not depend on /data.
PUSH_CELL = np.diag([54.820575, 54.820575, 75.240000])

NICR_MIN1 = np.array([
    [31.116434, 21.181185, 33.387510],
    [36.173676, 21.182257, 33.386432],
    [31.138723, 23.687441, 33.391101],
    [33.643252, 23.705551, 33.379339],
    [33.643062, 18.656106, 33.388786],
])
NICR_SADDLE = np.array([
    [32.617887, 21.202238, 33.569028],
    [36.094384, 21.182465, 33.386043],
    [31.186067, 23.559634, 33.376001],
    [33.639714, 23.582347, 33.374345],
    [33.632711, 18.790147, 33.384916],
])
NICR_MIN2 = np.array([
    [33.681452, 21.180625, 33.380952],
    [36.145714, 21.182474, 33.395516],
    [31.154823, 23.707405, 33.383042],
    [33.659003, 23.686424, 33.386811],
    [33.656726, 18.676091, 33.394793],
])

NIFE_MIN1 = np.array([
    [36.160886, 26.173553, 33.326364],
    [38.629664, 26.167711, 33.341033],
    [36.144033, 23.661212, 33.340817],
    [33.632757, 23.640282, 33.333332],
    [31.111802, 26.176432, 33.323765],
])
NIFE_SADDLE = np.array([
    [35.080127, 26.181017, 33.498397],
    [38.585226, 26.167354, 33.333216],
    [36.118675, 23.770824, 33.327040],
    [33.658581, 23.779156, 33.324122],
    [31.167431, 26.175993, 33.320490],
])
NIFE_MIN2 = np.array([
    [33.598327, 26.171305, 33.334039],
    [38.663800, 26.167063, 33.333451],
    [36.129855, 23.634690, 33.334631],
    [33.618089, 23.662605, 33.341761],
    [31.138124, 26.176432, 33.332898],
])

#(dump, target minimum, other minimum, saddle, legacy landing fraction)
#"landing fraction" = mover distance from the minimum being sought, divided by
#the min1->min2 hop. The reconstruct+minimise gate needs <= ~0.35 to reconnect.
PUSH_CASES = [
    ("nicr_5458->min1", NICR_MIN1, NICR_MIN2, NICR_SADDLE, 0.5012),
    ("nicr_5458->min2", NICR_MIN2, NICR_MIN1, NICR_SADDLE, 0.3580),
    ("nife_6377->min1", NIFE_MIN1, NIFE_MIN2, NIFE_SADDLE, 0.3630),
    ("nife_6377->min2", NIFE_MIN2, NIFE_MIN1, NIFE_SADDLE, 0.4945),
]

LEGACY_PUSH_FRACTION = 0.15
TARGET_LANDING_FRACTION = 0.35


def _mover_landing_fraction(
    pushed: np.ndarray, target_min: np.ndarray, other_min: np.ndarray
) -> float:
    """Mover distance from ``target_min`` after the push, over the min1<->min2 hop."""
    hop = per_atom_displacement(target_min.copy(), other_min.copy(), PUSH_CELL)[0]
    dist = per_atom_displacement(target_min.copy(), pushed.copy(), PUSH_CELL)[0]
    return float(dist / hop)


@pytest.mark.parametrize(("name", "target", "other", "saddle", "expected"), PUSH_CASES)
def test_push_rule_legacy_pins_the_campaign_exit_misses(
    name: str,
    target: np.ndarray,
    other: np.ndarray,
    saddle: np.ndarray,
    expected: float,
) -> None:
    """Regression pin of today's fixed 0.15 push on the two exit-failure dumps.

    Reproduces ``push_geom.py``: the mover is left at roughly half the hop from
    the minimum being sought in every direction, i.e. still on the saddle side
    of the barrier where the minimise can fall back the wrong way. All four
    cases sit above the 0.35 the reconstruction gate needs, the two observed
    failures (nicr toward min1, nife toward min2) grossly so.
    """
    from pykmc.utils.geometry import push_towards

    pushed = push_towards(
        saddle.copy(),
        target.copy(),
        fraction=LEGACY_PUSH_FRACTION,
        cell=PUSH_CELL,
        pbc=[True, True, True],
    )
    landed = _mover_landing_fraction(pushed, target, other)
    assert landed == pytest.approx(expected, abs=0.002), name
    assert landed > TARGET_LANDING_FRACTION, name


@pytest.mark.parametrize(("name", "target", "other", "saddle", "expected"), PUSH_CASES)
def test_adaptive_push_fraction_lands_mover_inside_target(
    name: str,
    target: np.ndarray,
    other: np.ndarray,
    saddle: np.ndarray,
    expected: float,
) -> None:
    """With the knob set, the mover lands within the requested share of the hop.

    Same four (dump x direction) cases as the legacy pin above; the adaptive
    fraction must bring every one of them to <= mover_landing_fraction.
    """
    from pykmc.utils.geometry import adaptive_push_fraction, push_towards

    fraction = adaptive_push_fraction(
        saddle,
        target,
        other,
        push_fraction=LEGACY_PUSH_FRACTION,
        mover_landing_fraction=TARGET_LANDING_FRACTION,
        cell=PUSH_CELL,
        pbc=[True, True, True],
    )
    assert fraction >= LEGACY_PUSH_FRACTION  # never pushes less than the legacy rule
    pushed = push_towards(
        saddle.copy(),
        target.copy(),
        fraction=fraction,
        cell=PUSH_CELL,
        pbc=[True, True, True],
    )
    landed = _mover_landing_fraction(pushed, target, other)
    assert landed <= TARGET_LANDING_FRACTION + 1e-9, (name, landed)
    assert landed < expected  # strictly closer to the sought minimum than legacy


def test_adaptive_push_fraction_keeps_shell_saddle_relaxation() -> None:
    """Shell atoms keep a proportional share of their saddle offset.

    The alternative rule -- place every atom at a fixed fraction of its own
    min1->min2 segment -- degenerates for the rcut shell, whose min1 and min2
    coincide to <= 0.035 A: it would drop the shell exactly onto the minimum
    and the minimise would no longer have to find its way back, gutting the
    min1/min2 connectivity check the push+minimise exists for. The adaptive
    fraction is uniform, so the shell keeps most of its saddle displacement.
    """
    from pykmc.utils.geometry import adaptive_push_fraction, push_towards

    fraction = adaptive_push_fraction(
        NICR_SADDLE,
        NICR_MIN1,
        NICR_MIN2,
        push_fraction=LEGACY_PUSH_FRACTION,
        mover_landing_fraction=TARGET_LANDING_FRACTION,
        cell=PUSH_CELL,
        pbc=[True, True, True],
    )
    pushed = push_towards(
        NICR_SADDLE.copy(),
        NICR_MIN1.copy(),
        fraction=fraction,
        cell=PUSH_CELL,
        pbc=[True, True, True],
    )
    offsets = per_atom_displacement(NICR_MIN1.copy(), pushed.copy(), PUSH_CELL)
    saddle_offsets = per_atom_displacement(
        NICR_MIN1.copy(), NICR_SADDLE.copy(), PUSH_CELL
    )
    # Every shell atom (rows 1-4) retains (1 - fraction) of its saddle offset,
    # which is >= half of it here -- not zero as a segment-fraction rule gives.
    retained = offsets[1:] / saddle_offsets[1:]
    assert np.all(retained > 0.5)
    np.testing.assert_allclose(retained, 1.0 - fraction, rtol=1e-6)


def test_adaptive_push_fraction_default_config_is_legacy_bit_identical() -> None:
    """The default ReconstructionConfig keeps the historical push exactly.

    ``mover_landing_fraction`` defaults to None, so every existing input file
    reproduces the previous push bit-for-bit -- the new rule is opt-in.
    """
    from pykmc.config import ReconstructionConfig
    from pykmc.utils.geometry import adaptive_push_fraction, push_towards

    cfg = ReconstructionConfig()
    assert cfg.mover_landing_fraction is None
    assert cfg.push_fraction == LEGACY_PUSH_FRACTION

    fraction = adaptive_push_fraction(
        NICR_SADDLE,
        NICR_MIN1,
        NICR_MIN2,
        push_fraction=cfg.push_fraction,
        mover_landing_fraction=cfg.mover_landing_fraction,
        cell=PUSH_CELL,
        pbc=[True, True, True],
    )
    assert fraction == cfg.push_fraction

    new = push_towards(
        NICR_SADDLE.copy(), NICR_MIN1.copy(), fraction=fraction,
        cell=PUSH_CELL, pbc=[True, True, True],
    )
    old = push_towards(
        NICR_SADDLE.copy(), NICR_MIN1.copy(), fraction=LEGACY_PUSH_FRACTION,
        cell=PUSH_CELL, pbc=[True, True, True],
    )
    assert np.array_equal(new, old)


def test_adaptive_push_fraction_degenerate_geometry_falls_back_to_legacy() -> None:
    """Degenerate inputs must not inflate the push toward 1.0 (land-on-minimum).

    A vanishing hop or a saddle further from the target than the whole hop is
    not a saddle sitting between two minima; the adaptive formula would demand
    an arbitrarily large fraction there, so those cases keep the legacy push
    and let the containment/acceptance gates reject the event.
    """
    from pykmc.utils.geometry import adaptive_push_fraction

    kwargs = {
        "push_fraction": LEGACY_PUSH_FRACTION,
        "mover_landing_fraction": TARGET_LANDING_FRACTION,
        "cell": PUSH_CELL,
        "pbc": [True, True, True],
    }
    flat = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]])
    # min1 == min2: no hop at all.
    assert adaptive_push_fraction(flat + 0.1, flat, flat, **kwargs) == LEGACY_PUSH_FRACTION
    # Saddle further from the target than the hop itself (s > 1).
    min1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    min2 = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    saddle = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    assert adaptive_push_fraction(saddle, min1, min2, **kwargs) == LEGACY_PUSH_FRACTION
    # Empty shell.
    empty = np.empty((0, 3))
    assert adaptive_push_fraction(empty, empty, empty, **kwargs) == LEGACY_PUSH_FRACTION


def test_host_and_engine_reconstruction_share_the_push_rule() -> None:
    """Host/engine parity: both reconstruction paths resolve the push identically.

    The serial (host) ``Reconstruction.reconstruct`` and the engine-side basin
    ``_basin_reconstruct_impl`` must apply an identical push, otherwise the two
    paths accept/reject different events. Each pushes twice (toward min1, then
    toward min2) and each must take its fraction from the shared helper, never
    from ``push_fraction`` directly -- change all four call sites or none.
    """
    import inspect

    import pykmc.enginemanager.lmpi.lammps_operations as ops
    from pykmc.reconstruction import Reconstruction

    for func in (Reconstruction.reconstruct, ops._basin_reconstruct_impl):
        source = inspect.getsource(func)
        assert source.count("push_towards(") == 2, func
        assert source.count("adaptive_push_fraction(") == 2, func
