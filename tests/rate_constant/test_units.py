"""Frequency unit conversions of pykmc.rate_constant.units."""

from __future__ import annotations

import math

import pytest

from pykmc.rate_constant import hz_to_per_ps, thz_to_hz
from pykmc.rate_constant.units import HZ_PER_THZ

GRID = (0.0, 1.0, 3.7, 5e12, 1e13, 2.5e14, 1e-3)


@pytest.mark.parametrize("value", GRID)
def test_hz_to_per_ps_is_times_1e_minus_12(value: float) -> None:
    """Assert hz_to_per_ps(x) is exactly the product x * 1e-12."""
    assert hz_to_per_ps(value) == value * 1e-12


@pytest.mark.parametrize("value", GRID)
def test_thz_to_hz_is_times_1e12(value: float) -> None:
    """Assert thz_to_hz(x) is exactly the product x * 1e12."""
    assert thz_to_hz(value) == value * 1e12


def test_one_thz_is_one_per_ps() -> None:
    """Check the 1 THz == 1 ps^-1 identity through both helpers."""
    assert HZ_PER_THZ == 1e12
    assert hz_to_per_ps(thz_to_hz(1.0)) == 1.0
    assert hz_to_per_ps(1e13) == 10.0


def test_round_trip_is_close() -> None:
    """Check that THz -> Hz -> ps^-1 returns the input to double precision."""
    for thz in (1.0, 12.5, 99.9):
        assert math.isclose(hz_to_per_ps(thz_to_hz(thz)), thz, rel_tol=1e-15)
