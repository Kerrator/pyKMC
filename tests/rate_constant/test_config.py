"""RateConstantConfig fields and validators."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from pykmc.config import Config, RateConstantConfig

ROOT = Path(__file__).resolve().parents[2]
INPUT_IN = ROOT / "tests" / "data" / "input.in"


def test_defaults() -> None:
    """Assert the documented default of every field."""
    cfg = RateConstantConfig(style="constant")
    assert cfg.k0 == 1.0
    assert cfg.T == 300.0
    assert cfg.free_radius == 6.0
    assert cfg.fd_step == 0.01
    assert cfg.zone_radius is None
    assert cfg.nu0_min_THz == 1.0
    assert cfg.nu0_max_THz == 100.0
    assert cfg.premin is False
    assert cfg.free_region_center == "saddle"


def test_style_is_required_and_restricted() -> None:
    """Assert style has no default and only accepts the three registered names."""
    with pytest.raises(ValidationError):
        RateConstantConfig()  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        RateConstantConfig(style="bogus")  # type: ignore[arg-type]
    for style in ("constant", "htst", "rpa"):
        assert RateConstantConfig(style=style).style == style


def test_k0_max_is_a_class_var_not_a_field() -> None:
    """Assert the k0 threshold is not a pydantic field (docs are generated from fields)."""
    assert RateConstantConfig.K0_MAX_PS_INV == 1e4
    assert "K0_MAX_PS_INV" not in RateConstantConfig.model_fields
    assert set(RateConstantConfig.model_fields) == {
        "style",
        "k0",
        "T",
        "free_radius",
        "free_region_center",
        "fd_step",
        "zone_radius",
        "nu0_min_THz",
        "nu0_max_THz",
        "premin",
    }


@pytest.mark.parametrize("center", ["saddle", "min1"])
def test_free_region_center_accepts_the_two_centrings(center: str) -> None:
    """Assert both documented centrings validate and are stored as given."""
    assert (
        RateConstantConfig(style="htst", free_region_center=center).free_region_center
        == center
    )
    # the INI path hands pydantic a plain string, exactly as here
    assert (
        RateConstantConfig.model_validate(
            {"style": "htst", "free_region_center": center}
        ).free_region_center
        == center
    )


@pytest.mark.parametrize("bad", ["min2", "SADDLE", "Saddle", "", "none", 0, None])
def test_free_region_center_rejects_anything_else(bad: object) -> None:
    """Assert other strings (and non-strings) are configuration errors."""
    with pytest.raises(ValidationError):
        RateConstantConfig(style="htst", free_region_center=bad)  # type: ignore[arg-type]


def test_free_region_center_description_states_the_min1_artifact() -> None:
    """Assert the field documents the measured min1 asymmetry and its purpose."""
    description = RateConstantConfig.model_fields["free_region_center"].description
    assert "20 percent" in description
    assert "comparison with older results" in description


def test_nu0_window_must_be_ordered() -> None:
    """Assert nu0_min_THz >= nu0_max_THz is rejected with a clear message."""
    with pytest.raises(ValidationError, match="nu0_min_THz must be < nu0_max_THz"):
        RateConstantConfig(style="htst", nu0_min_THz=100.0, nu0_max_THz=1.0)
    with pytest.raises(ValidationError, match="nu0_min_THz must be < nu0_max_THz"):
        RateConstantConfig(style="constant", nu0_min_THz=5.0, nu0_max_THz=5.0)
    cfg = RateConstantConfig(style="htst", nu0_min_THz=0.5, nu0_max_THz=200.0)
    assert (cfg.nu0_min_THz, cfg.nu0_max_THz) == (0.5, 200.0)


def test_window_descriptions_name_the_kernel() -> None:
    """Assert the nu0_*_THz descriptions say the window is applied by the HTST kernel."""
    for name in ("nu0_min_THz", "nu0_max_THz"):
        description = RateConstantConfig.model_fields[name].description
        assert "THz" in description
        assert "applied by the HTST kernel" in description


@pytest.mark.parametrize("style", ["htst", "rpa"])
def test_hz_scale_k0_rejected_for_vineyard_styles(style: str) -> None:
    """Assert k0=1e12 is rejected for htst/rpa and the message explains the units."""
    with pytest.raises(ValidationError) as excinfo:
        RateConstantConfig(style=style, k0=1e12)
    message = str(excinfo.value)
    assert "ps^-1" in message
    assert "1e12" in message
    assert "Hz" in message


@pytest.mark.parametrize("style", ["htst", "rpa"])
def test_k0_at_threshold_accepted_for_vineyard_styles(style: str) -> None:
    """Assert k0 = 1e4 ps^-1 passes the guard while 1e4 + 1 does not."""
    assert RateConstantConfig(style=style, k0=1e4).k0 == 1e4
    assert RateConstantConfig(style=style, k0=1.0).k0 == 1.0
    with pytest.raises(ValidationError):
        RateConstantConfig(style=style, k0=1e4 + 1.0)


def test_constant_accepts_hz_scale_k0() -> None:
    """Assert constant keeps accepting k0=1e12 (it only rescales absolute time)."""
    assert RateConstantConfig(style="constant", k0=1e12).k0 == 1e12


@pytest.mark.parametrize("field", ["k0", "T", "free_radius", "fd_step", "nu0_min_THz"])
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_non_positive_numeric_fields_rejected(field: str, value: float) -> None:
    """Assert zero and negative values are configuration errors."""
    with pytest.raises(ValidationError, match="greater than 0"):
        RateConstantConfig(style="constant", **{field: value})


def test_zone_radius_optional_and_positive() -> None:
    """Assert zone_radius accepts None (also the INI spelling) and positive values."""
    assert RateConstantConfig(style="htst", zone_radius=None).zone_radius is None
    assert RateConstantConfig(style="htst", zone_radius="None").zone_radius is None
    assert RateConstantConfig(style="htst", zone_radius=9.0).zone_radius == 9.0
    with pytest.raises(ValidationError, match="greater than 0"):
        RateConstantConfig(style="htst", zone_radius=0.0)


def test_premin_parses_ini_strings() -> None:
    """Assert the boolean premin flag accepts INI-style strings."""
    assert RateConstantConfig(style="htst", premin="True").premin is True
    assert RateConstantConfig(style="htst", premin="false").premin is False


def test_committed_input_file_loads() -> None:
    """Assert tests/data/input.in (style constant, k0 = 1e12) still loads."""
    config = Config.from_ini_file(str(INPUT_IN))
    rc = config.rateconstant
    assert rc.style == "constant"
    assert rc.k0 == 1e12
    assert rc.T == 300.0
    assert rc.zone_radius is None


def test_htst_ini_section_round_trip(tmp_path: Path) -> None:
    """Assert an htst [RateConstant] section parses through from_ini_file."""
    text = INPUT_IN.read_text()
    start = text.index("[RateConstant]")
    end = text.index("[PSR]")
    section = (
        "[RateConstant]\n"
        "style = htst\n"
        "k0 = 1.0\n"
        "T = 500.0\n"
        "free_radius = 5.5\n"
        "fd_step = 0.02\n"
        "zone_radius = 12.0\n"
        "nu0_min_THz = 0.5\n"
        "nu0_max_THz = 50.0\n"
        "premin = True\n\n"
    )
    ini = tmp_path / "input_htst.in"
    ini.write_text(text[:start] + section + text[end:])
    rc = Config.from_ini_file(str(ini)).rateconstant
    assert rc.style == "htst"
    assert (rc.k0, rc.T, rc.free_radius, rc.fd_step) == (1.0, 500.0, 5.5, 0.02)
    assert (rc.zone_radius, rc.nu0_min_THz, rc.nu0_max_THz) == (12.0, 0.5, 50.0)
    assert rc.premin is True


@pytest.mark.parametrize(
    "field",
    ["k0", "T", "free_radius", "fd_step", "zone_radius", "nu0_min_THz", "nu0_max_THz"],
)
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_numeric_fields_reject_non_finite_values(field: str, value: float) -> None:
    """Every bounded numeric field rejects inf and nan, not only values below 0."""
    import pydantic

    from pykmc.config import RateConstantConfig

    with pytest.raises(pydantic.ValidationError):
        RateConstantConfig(style="constant", **{field: value})


@pytest.mark.parametrize("zone_radius", [5.0, 6.0])
def test_zone_radius_must_exceed_free_radius(zone_radius: float) -> None:
    """A crop zone that does not enclose the free region is a configuration error."""
    with pytest.raises(ValidationError, match="zone_radius must be > free_radius"):
        RateConstantConfig(style="htst", free_radius=6.0, zone_radius=zone_radius)


def test_zone_radius_none_or_larger_is_accepted() -> None:
    """``None`` (full system) and a strictly larger zone both validate."""
    assert RateConstantConfig(style="htst", free_radius=6.0).zone_radius is None
    assert (
        RateConstantConfig(style="htst", free_radius=6.0, zone_radius=10.0).zone_radius
        == 10.0
    )
