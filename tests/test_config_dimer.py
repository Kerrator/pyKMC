"""Tests for `[EventSearch] style = dimer` and the `[Dimer]` section."""

import inspect
from pathlib import Path

import pytest

from pykmc.config import Config, DimerConfig
from pykmc.dimer import dimer_search

INPUT = Path(__file__).parent / "data" / "input.in"


def _write_input(
    tmp_path: Path,
    style: str = "dimer",
    dimer: str | None = "start_radius = 3.0",
    partn: bool = True,
) -> str:
    text = INPUT.read_text().replace("style = partn", f"style = {style}")
    if not partn:
        text = text.replace("[pARTn] \n", "")
        assert "[pARTn]" not in text
    if dimer is not None:
        text += f"\n[Dimer]\n{dimer}\n"
    path = tmp_path / "input.in"
    path.write_text(text)
    return str(path)


def test_dimer_with_dimer_and_partn_sections_parses(tmp_path: Path) -> None:
    config = Config.from_ini_file(_write_input(tmp_path))

    assert config.eventsearch.style == "dimer"
    assert config.dimer.start_radius == 3.0
    assert config.partn is not None


def test_dimer_defaults_follow_the_design_record(tmp_path: Path) -> None:
    dimer = Config.from_ini_file(_write_input(tmp_path)).dimer

    # Decision 12: the local start is the default.
    assert dimer.start == "local"
    # Decision 15: the largest force component, below 1e-3 eV/Å.
    assert dimer.convergence == "max_component"
    assert dimer.force_tolerance == 1e-3
    # Decision 14: SEAKMC's separation and step sizes.
    assert dimer.separation == 0.005
    assert dimer.trial_step == 0.015
    assert dimer.max_step == 0.05
    # DAV is on unless the input turns it off.
    assert dimer.dav is True


def test_dimer_algorithm_options_match_the_core_keywords_and_defaults() -> None:
    """The worker consumes the engine option and forwards the algorithm options."""
    keywords = inspect.signature(dimer_search).parameters
    section = DimerConfig(start_radius=3.0).model_dump()

    assert section.pop("reuse_neighbor_list") is True
    assert section.pop("eam_force_crop") is False
    for name, value in section.items():
        assert name in keywords
        if name != "start_radius":
            assert keywords[name].default == value


def test_dimer_reuses_neighbor_lists_by_default_and_crop_is_opt_in(
    tmp_path: Path,
) -> None:
    settings = Config.from_ini_file(_write_input(tmp_path)).dimer.model_dump()
    assert settings["reuse_neighbor_list"] is True
    assert settings["eam_force_crop"] is False


@pytest.mark.parametrize(
    ("option", "value"), [("reuse_neighbor_list", False), ("eam_force_crop", True)]
)
def test_dimer_worker_optimizations_follow_the_input(
    tmp_path: Path, option: str, value: bool
) -> None:
    path = _write_input(tmp_path, dimer=f"start_radius = 3.0\n{option} = {value}")

    assert Config.from_ini_file(path).dimer.model_dump()[option] is value


@pytest.mark.parametrize("option", ["reuse_neighbor_list", "eam_force_crop"])
def test_dimer_rejects_unsupported_worker_optimizations_before_dispatch(
    tmp_path: Path,
    option: str,
) -> None:
    path = Path(_write_input(tmp_path))
    hybrid = path.read_text().replace("eam/alloy", "hybrid/overlay eam/alloy zbl 1 2")
    path.write_text(hybrid)
    assert Config.from_ini_file(str(path)).dimer.model_dump()[option] is False

    path.write_text(hybrid + f"{option} = True\n")
    with pytest.raises(ValueError, match=f"{option} requires pair_style"):
        Config.from_ini_file(str(path))


def test_dimer_without_dimer_section_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dimer"):
        Config.from_ini_file(_write_input(tmp_path, dimer=None))


def test_dimer_without_partn_section_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="partn"):
        Config.from_ini_file(_write_input(tmp_path, partn=False))


def test_local_start_needs_a_radius(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="start_radius"):
        Config.from_ini_file(_write_input(tmp_path, dimer="start = local"))


def test_region_start_needs_no_radius(tmp_path: Path) -> None:
    config = Config.from_ini_file(_write_input(tmp_path, dimer="start = region"))

    assert config.dimer.start == "region"
    assert config.dimer.start_radius is None


@pytest.mark.parametrize(
    "dimer",
    [
        "start_radius = 3.0\ntrial_step = 0.05",
        "start_radius = 0",
        "start_radius = 3.0\nforce_tolerance = 0",
        "start_radius = 3.0\nconvergence = rms",
        "start = sphere\nstart_radius = 3.0",
    ],
)
def test_malformed_dimer_section_fails(tmp_path: Path, dimer: str) -> None:
    with pytest.raises(ValueError):
        Config.from_ini_file(_write_input(tmp_path, dimer=dimer))


def test_partn_input_needs_no_change() -> None:
    config = Config.from_ini_file(str(INPUT))

    assert config.eventsearch.style == "partn"
    assert config.dimer is None


def test_partn_does_not_need_a_dimer_section_and_ignores_one(tmp_path: Path) -> None:
    config = Config.from_ini_file(_write_input(tmp_path, style="partn"))

    assert config.eventsearch.style == "partn"
