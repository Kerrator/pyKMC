"""Tests for ``HTSTSettings`` validation, immutability and pickling."""

from __future__ import annotations

import dataclasses
import pickle
from typing import Any

import numpy as np
import pytest

from pykmc.htst import HTSTSettings
from pykmc.htst.settings import DEFAULT_INTERACTION_RANGE


def test_defaults_match_contract() -> None:
    """The frozen defaults are the contract values."""
    s = HTSTSettings()
    assert s.free_radius == 8.0
    assert s.fd_step == 0.01
    assert s.zone_radius is None
    assert s.resolved_zone_radius is None
    assert s.interaction_range == DEFAULT_INTERACTION_RANGE == 13.0
    assert s.premin is False
    assert s.nu0_min_hz == 1.0e12
    assert s.nu0_max_hz == 1.0e14
    assert s.zero_mode_tol == 1.0e-6
    assert s.free_region_center == "saddle"


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
        {"free_region_center": "min2"},
        {"free_region_center": "Saddle"},
        {"free_region_center": ""},
        {"free_region_center": None},
        {"free_region_center": 1},
        {"zone_radius": "Auto"},
        {"zone_radius": "none"},
        {"zone_radius": "19.0"},
        {"zone_radius": ""},
        {"interaction_range": 0.0},
        {"interaction_range": -13.0},
        {"interaction_range": float("nan")},
        {"interaction_range": float("inf")},
        {"interaction_range": True},
        {"interaction_range": "13.0"},
        {"interaction_range": None},
    ],
)
def test_invalid_settings_raise_value_error(kwargs: dict[str, object]) -> None:
    """Non-finite, non-positive or inverted-window settings are rejected."""
    with pytest.raises(ValueError):
        HTSTSettings(**kwargs)


@pytest.mark.parametrize("center", ["saddle", "min1"])
def test_free_region_center_accepts_the_two_centrings(center: str) -> None:
    """Both centrings validate, pickle and keep value equality."""
    s = HTSTSettings(free_region_center=center)
    assert s.free_region_center == center
    assert pickle.loads(pickle.dumps(s)) == s
    assert s != HTSTSettings(
        free_region_center="min1" if center == "saddle" else "saddle"
    )


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


@pytest.mark.parametrize(
    ("zone_radius", "resolved"),
    [(None, None), ("auto", 6.0 + DEFAULT_INTERACTION_RANGE), (12.0, 12.0)],
)
def test_resolved_zone_radius_for_every_spelling(
    zone_radius: Any, resolved: float | None
) -> None:
    """``None`` stays the full system, ``auto`` resolves, a number is itself."""
    s = HTSTSettings(free_radius=6.0, zone_radius=zone_radius)
    assert s.zone_radius == zone_radius
    assert s.resolved_zone_radius == resolved


def test_auto_zone_resolves_against_the_declared_interaction_range() -> None:
    """``auto`` is ``free_radius + interaction_range``; the range sizes nothing else."""
    s = HTSTSettings(free_radius=4.0, zone_radius="auto", interaction_range=5.5)
    assert s.interaction_range == 5.5
    assert s.resolved_zone_radius == 9.5
    assert (
        HTSTSettings(free_radius=4.0, interaction_range=5.5).resolved_zone_radius
        is None
    )
    assert (
        HTSTSettings(
            free_radius=4.0, zone_radius=7.0, interaction_range=5.5
        ).resolved_zone_radius
        == 7.0
    )


@pytest.mark.parametrize("value", [np.float64(13.0), np.int64(13), 13])
def test_interaction_range_is_stored_as_float(value: Any) -> None:
    """The range accepts the same real scalars as the other numeric settings."""
    s = HTSTSettings(interaction_range=value)
    assert type(s.interaction_range) is float and s.interaction_range == 13.0


def test_auto_settings_pickle_round_trip_keeps_equality() -> None:
    """An ``auto`` zone pickles like any other setting."""
    s = HTSTSettings(zone_radius="auto", interaction_range=11.0)
    back = pickle.loads(pickle.dumps(s))
    assert back == s and hash(back) == hash(s)
    assert back.resolved_zone_radius == 19.0


def test_calculation_fields_resolve_the_zone_and_drop_the_range() -> None:
    """``auto`` and the equal explicit radius describe one calculation.

    Without ``auto`` the fields are exactly those of the settings before
    ``interaction_range`` existed, so persisted digests keep their value.
    """
    auto = HTSTSettings(free_radius=6.0, zone_radius="auto")
    explicit = HTSTSettings(free_radius=6.0, zone_radius=19.0, interaction_range=4.0)
    assert auto != explicit
    assert auto.calculation_fields() == explicit.calculation_fields()
    assert auto.calculation_fields()["zone_radius"] == 19.0
    full = HTSTSettings(free_radius=6.0)
    assert full.calculation_fields() == {
        "free_radius": 6.0,
        "fd_step": 0.01,
        "zone_radius": None,
        "premin": False,
        "nu0_min_hz": 1.0e12,
        "nu0_max_hz": 1.0e14,
        "zero_mode_tol": 1.0e-6,
        "free_region_center": "saddle",
        "force_tol": 0.005,
    }
    assert full.calculation_fields() != auto.calculation_fields()


def test_settings_pickled_before_interaction_range_read_the_default() -> None:
    """A pickle without the new field (an older table's archive) still resolves.

    The instance state then lacks ``interaction_range`` and the class default
    applies; equality, hashing and the calculation fields are unaffected.
    """
    s = HTSTSettings(free_radius=5.5, zone_radius=12.0)
    old = HTSTSettings.__new__(HTSTSettings)
    state = dict(s.__dict__)
    del state["interaction_range"]
    object.__setattr__(old, "__dict__", state)
    back = pickle.loads(pickle.dumps(old))
    assert "interaction_range" not in back.__dict__
    assert back.interaction_range == DEFAULT_INTERACTION_RANGE
    assert back == s and hash(back) == hash(s)
    assert back.calculation_fields() == s.calculation_fields()
    assert back.resolved_zone_radius == 12.0
