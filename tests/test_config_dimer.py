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


def test_dimer_keys_are_the_core_keywords_with_its_defaults() -> None:
    """The section goes to the dimer core as keywords, unchanged."""
    keywords = inspect.signature(dimer_search).parameters
    section = DimerConfig(start_radius=3.0).model_dump()

    for name, value in section.items():
        assert name in keywords
        if name != "start_radius":
            assert keywords[name].default == value


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
