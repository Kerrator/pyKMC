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
