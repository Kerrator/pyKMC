"""Tests for PBC-aware helpers in pykmc.utils.geometry."""

import numpy as np
import pytest

from pykmc.utils.geometry import minimum_image_distance, per_atom_displacement

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
