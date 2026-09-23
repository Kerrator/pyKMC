"""Tests for minimum-image free-region selection."""

from __future__ import annotations

import numpy as np
import pytest

from pykmc.htst import HTSTGeometryError, HTSTRequestError, select_free_indices

_CELL = np.diag([10.0, 10.0, 10.0])


def test_boundary_crossing_sphere_under_pbc() -> None:
    """Atoms across the periodic boundary are inside the minimum-image sphere."""
    positions = np.array(
        [
            [0.5, 0.5, 0.5],  # centre
            [1.5, 0.5, 0.5],  # 1.0 away
            [9.8, 0.5, 0.5],  # 0.7 away across x
            [0.5, 9.0, 0.5],  # 1.5 away across y
            [0.5, 0.5, 8.9],  # 1.6 away across z
            [5.5, 0.5, 0.5],  # 5.0 away, outside
            [0.5, 3.5, 0.5],  # 3.0 away, outside
        ]
    )
    free = select_free_indices(positions, 0, 2.0, _CELL, (True, True, True))
    assert free.tolist() == [0, 1, 2, 3, 4]
    assert free.dtype.kind == "i"


def test_non_periodic_axis_uses_raw_distance() -> None:
    """With pbc False on z, an atom near the far z face is not wrapped."""
    positions = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 9.8], [9.8, 0.5, 0.5]])
    free = select_free_indices(positions, 0, 2.0, _CELL, (True, True, False))
    assert free.tolist() == [0, 2]
    free_all = select_free_indices(positions, 0, 2.0, _CELL, (True, True, True))
    assert free_all.tolist() == [0, 1, 2]


def test_triclinic_cell_raises_geometry_error() -> None:
    """A sheared cell is rejected before any distance is computed."""
    cell = np.array([[10.0, 0.0, 0.0], [3.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    positions = np.zeros((2, 3))
    with pytest.raises(HTSTGeometryError):
        select_free_indices(positions, 0, 2.0, cell, (True, True, True))


def test_centre_always_included_and_tiny_radius_gives_only_centre() -> None:
    """The centre sits at distance zero, so a tiny radius selects exactly it."""
    positions = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.3], [4.0, 4.0, 4.0]])
    assert select_free_indices(positions, 2, 1.0e-9, _CELL, (True,) * 3).tolist() == [2]
    assert select_free_indices(positions, 1, 0.0, _CELL, (True,) * 3).tolist() == [1]


def test_cutoff_is_inclusive_and_result_sorted() -> None:
    """An atom exactly at the radius is selected; indices come back sorted."""
    positions = np.array(
        [[5.0, 5.0, 5.0], [7.0, 5.0, 5.0], [5.0, 3.0, 5.0], [8.0, 5.0, 5.0]]
    )
    free = select_free_indices(positions, 2, 2.0, _CELL, (False, False, False))
    assert free.tolist() == [0, 2]
    free = select_free_indices(positions, 0, 2.0, _CELL, (False, False, False))
    assert free.tolist() == [0, 1, 2]


def test_large_sphere_selects_every_atom_once() -> None:
    """A radius beyond half the cell still returns each index at most once."""
    rng = np.random.default_rng(3)
    positions = rng.uniform(0.0, 10.0, size=(40, 3))
    free = select_free_indices(positions, 5, 50.0, _CELL, (True, True, True))
    assert free.tolist() == list(range(40))


@pytest.mark.parametrize("center", [-1, 3, True, 1.0])
def test_bad_center_index_raises_request_error(center: object) -> None:
    """Out-of-range or non-int centre indices raise HTSTRequestError."""
    positions = np.zeros((3, 3))
    with pytest.raises(HTSTRequestError):
        select_free_indices(positions, center, 1.0, _CELL, (True,) * 3)  # type: ignore[arg-type]


@pytest.mark.parametrize("radius", [-0.1, float("nan"), float("inf")])
def test_bad_radius_raises_value_error(radius: float) -> None:
    """Negative or non-finite radii raise ValueError."""
    with pytest.raises(ValueError):
        select_free_indices(np.zeros((2, 3)), 0, radius, _CELL, (True,) * 3)


def test_periodic_axis_with_zero_length_raises() -> None:
    """A periodic axis needs a positive cell length."""
    cell = np.diag([10.0, 0.0, 10.0])
    with pytest.raises(HTSTRequestError):
        select_free_indices(np.zeros((2, 3)), 0, 1.0, cell, (True, True, True))


def test_string_pbc_is_rejected_not_coerced_to_true() -> None:
    """String flags must not silently wrap a non-periodic axis."""
    positions = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 9.8], [9.8, 0.5, 0.5]])
    with pytest.raises(HTSTRequestError, match="pbc"):
        select_free_indices(positions, 0, 2.0, _CELL, ("True", "True", "False"))
    with pytest.raises(HTSTRequestError, match="pbc"):
        select_free_indices(positions, 0, 2.0, _CELL, (1.0, 1.0, 0.0))
    with pytest.raises(HTSTRequestError, match="pbc"):
        select_free_indices(positions, 0, 2.0, _CELL, (1, 2, 0))
    # Python bools, NumPy bools and 0/1 integers are all honoured axis by axis.
    for pbc in ((True, True, False), tuple(np.array([True, True, False])), (1, 1, 0)):
        assert select_free_indices(positions, 0, 2.0, _CELL, pbc).tolist() == [0, 2]


def test_non_finite_positions_raise_instead_of_shrinking_the_region() -> None:
    """A NaN row anywhere is a payload error, never a silently smaller free set."""
    positions = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [1.0, np.nan, 0.5]])
    with pytest.raises(HTSTRequestError, match="non-finite"):
        select_free_indices(positions, 0, 2.0, _CELL, (True,) * 3)
    with pytest.raises(HTSTRequestError, match="non-finite"):
        select_free_indices(positions, 2, 2.0, _CELL, (True,) * 3)
    positions[2, 1] = np.inf
    with pytest.raises(HTSTRequestError, match="non-finite"):
        select_free_indices(positions, 0, 2.0, _CELL, (True,) * 3)


def test_numpy_integer_and_float_radii_are_accepted() -> None:
    """np.int64 / np.float32 radii behave like the Python numbers they hold."""
    positions = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [5.5, 0.5, 0.5]])
    for radius in (np.int64(2), np.float32(2.0), 2, 2.0):
        assert select_free_indices(
            positions, 0, radius, _CELL, (True,) * 3
        ).tolist() == [
            0,
            1,
        ]
    with pytest.raises(ValueError):
        select_free_indices(positions, 0, True, _CELL, (True,) * 3)  # type: ignore[arg-type]
