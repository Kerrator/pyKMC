"""Tests for ``HTSTSettings`` validation, immutability and pickling."""

from __future__ import annotations

import dataclasses
import pickle

import numpy as np
import pytest

from pykmc.htst import HTSTSettings


def test_defaults_match_contract() -> None:
    """The frozen defaults are the contract values."""
    s = HTSTSettings()
    assert s.free_radius == 6.0
    assert s.fd_step == 0.01
    assert s.zone_radius is None
    assert s.premin is False
    assert s.nu0_min_hz == 1.0e12
    assert s.nu0_max_hz == 1.0e14
    assert s.zero_mode_tol == 1.0e-6


@pytest.mark.parametrize(
    "kwargs",
    [
        {"free_radius": 0.0},
        {"free_radius": -1.0},
        {"free_radius": float("nan")},
        {"free_radius": float("inf")},
        {"fd_step": 0.0},
        {"fd_step": -0.01},
        {"zone_radius": 0.0},
        {"zone_radius": float("inf")},
        {"premin": 1},
        {"nu0_min_hz": 0.0},
        {"nu0_max_hz": float("inf")},
        {"nu0_min_hz": 1.0e13, "nu0_max_hz": 1.0e13},
        {"nu0_min_hz": 2.0e13, "nu0_max_hz": 1.0e13},
        {"zero_mode_tol": -1.0e-6},
        {"zero_mode_tol": float("nan")},
        {"free_radius": "6.0"},
        {"free_radius": True},
        {"free_radius": np.bool_(True)},
        {"zero_mode_tol": np.bool_(False)},
        {"nu0_max_hz": np.float32("inf")},
    ],
)
def test_invalid_settings_raise_value_error(kwargs: dict[str, object]) -> None:
    """Non-finite, non-positive or inverted-window settings are rejected."""
    with pytest.raises(ValueError):
        HTSTSettings(**kwargs)


def test_zero_tolerance_is_allowed() -> None:
    """A zero tolerance is a valid strict classification threshold."""
    assert HTSTSettings(zero_mode_tol=0.0).zero_mode_tol == 0.0


def test_settings_are_frozen() -> None:
    """Assignment after construction raises."""
    s = HTSTSettings()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.fd_step = 0.02  # type: ignore[misc]


def test_settings_pickle_round_trip_keeps_equality() -> None:
    """Pickling preserves value equality and hashing."""
    s = HTSTSettings(free_radius=5.5, zone_radius=12.0, premin=True)
    back = pickle.loads(pickle.dumps(s))
    assert back == s
    assert hash(back) == hash(s)
    assert back is not s


@pytest.mark.parametrize(
    "value", [np.float32(6.0), np.float64(6.0), np.int64(6), np.int32(6), 6]
)
def test_numpy_real_scalars_are_accepted_and_stored_as_float(value: object) -> None:
    """NumPy reals are valid settings, like masses and radii elsewhere in the package."""
    s = HTSTSettings(
        free_radius=value,  # type: ignore[arg-type]
        fd_step=value,  # type: ignore[arg-type]
        zone_radius=value,  # type: ignore[arg-type]
        zero_mode_tol=value,  # type: ignore[arg-type]
    )
    for name in ("free_radius", "fd_step", "zone_radius", "zero_mode_tol"):
        stored = getattr(s, name)
        assert type(stored) is float
        assert stored == 6.0
    assert s == HTSTSettings(
        free_radius=6.0, fd_step=6.0, zone_radius=6.0, zero_mode_tol=6.0
    )
