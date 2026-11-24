"""Tests for the [RateConstant] keys of style = htst."""

import pytest
from pydantic import ValidationError

from pykmc.config import RateConstantConfig


def test_htst_defaults() -> None:
    config = RateConstantConfig(style="htst")

    assert config.rmov == 8.0
    assert config.rbuff is None
    assert config.hessian_dir is None
    assert config.fd_step == 0.01


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "full", "av"])
def test_rmov_is_a_positive_radius(value: str) -> None:
    assert RateConstantConfig(style="htst", rmov="6.5").rmov == 6.5
    with pytest.raises(ValidationError, match="rmov"):
        RateConstantConfig(style="htst", rmov=value)


def test_k0_in_hz_is_refused_under_htst_only() -> None:
    assert RateConstantConfig(style="constant", k0=1e13).k0 == 1e13
    with pytest.raises(ValidationError, match="k0 is in ps\\^-1"):
        RateConstantConfig(style="htst", k0=1e13)


@pytest.mark.parametrize("key", ["rbuff", "fd_step"])
@pytest.mark.parametrize("value", ["0", "nan", "inf"])
def test_rbuff_and_fd_step_are_positive_and_finite(key: str, value: str) -> None:
    with pytest.raises(ValidationError, match=key):
        RateConstantConfig(style="htst", **{key: value})


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan")])
def test_k0_must_be_positive_under_htst(value: float) -> None:
    with pytest.raises(ValidationError, match="must be positive"):
        RateConstantConfig(style="htst", k0=value)


@pytest.mark.parametrize("key", ["premin", "nu0_zone_radius", "nu0_min_THz"])
def test_removed_keys_are_refused_by_name(key: str) -> None:
    with pytest.raises(ValidationError, match=f"{key} is no longer supported"):
        RateConstantConfig(style="htst", **{key: "1"})
    with pytest.raises(ValidationError, match="style = rpa is no longer supported"):
        RateConstantConfig(style="rpa")


@pytest.mark.parametrize(
    ("old", "new"), [("free_radius", "rmov"), ("shell_width", "rbuff")]
)
def test_earlier_names_of_the_radii_are_refused(old: str, new: str) -> None:
    with pytest.raises(ValidationError, match=f"{old} is now {new}"):
        RateConstantConfig(style="htst", **{old: "6"})
