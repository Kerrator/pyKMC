"""pyKMC loads and runs without pypARTn, pARTn's Python module, unless pARTn is asked for.

Only `[EventSearch] style = partn` and `refine_style = partn` (the default) run
pARTn. A run that asks for it where pypARTn does not import stops once the input
is read, before any engine starts, and the message names pypARTn. No LAMMPS:
the engine module is imported with an empty stand-in for LAMMPS's module.
"""

import subprocess
import sys
import types
from pathlib import Path

import pytest

import pykmc.run as run
from pykmc.config import Config, check_pypartn

ROOT = Path(__file__).resolve().parents[1]
INPUT = Path(__file__).parent / "data" / "input.in"

# pypARTn blocked from import; LAMMPS's module replaced by an empty one.
_NO_PYPARTN = (
    "import sys, types; sys.modules['pypARTn'] = None; "
    "lmp = types.ModuleType('lammps'); lmp.lammps = object; "
    "sys.modules['lammps'] = lmp; "
)


def _input(tmp_path: Path, style: str, refine_style: str | None) -> str:
    """tests/data/input.in with the search style and, if given, refine_style."""
    text = INPUT.read_text().replace("style = partn", f"style = {style}")
    if refine_style is not None:
        text = text.replace(
            "emin_event = 0.0", f"emin_event = 0.0\nrefine_style = {refine_style}"
        )
    if style == "dimer":
        text += "\n[Dimer]\nstart_radius = 3.0\n"
    path = tmp_path / "input.in"
    path.write_text(text)
    return str(path)


@pytest.fixture
def no_pypartn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import pypARTn`` fail, as where it is not installed."""
    monkeypatch.setitem(sys.modules, "pypARTn", None)


def test_pykmc_and_its_lammps_engine_import_without_pypartn() -> None:
    """The engine, the active-volume setup and the ARTn extension import."""
    code = _NO_PYPARTN + (
        "import pykmc, pykmc.engine.lammps, pykmc.activevolume.active_volume, "
        "pykmc.engine.artn_search; "
        "from pykmc.engine.lammps import LammpsEngine; "
        "from pykmc.activevolume.active_volume import define_AV, partn_search_AV; "
        "from pykmc.engine.artn_search import ArtnSearchExtension; "
        "assert sys.modules['pypARTn'] is None"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


@pytest.mark.parametrize(
    "style, refine_style, needs",
    [
        ("partn", None, "style = partn and refine_style = partn (the default)"),
        ("partn", "artn", "style = partn runs"),
        ("artn", None, "refine_style = partn (the default)"),
        ("dimer", "partn", "refine_style = partn runs"),
    ],
)
def test_a_config_that_needs_partn_fails_naming_pypartn(
    tmp_path: Path, no_pypartn: None, style: str, refine_style: str, needs: str
) -> None:
    """A config with style = partn or refine_style = partn is refused without pypARTn."""
    config = Config.from_ini_file(_input(tmp_path, style, refine_style))
    with pytest.raises(ValueError, match="pypARTn") as err:
        check_pypartn(config)
    assert needs in str(err.value)


@pytest.mark.parametrize("style", ["artn", "dimer"])
def test_refine_style_artn_needs_no_pypartn(
    tmp_path: Path, no_pypartn: None, style: str
) -> None:
    """With refine_style = artn, style = artn or dimer passes the check."""
    check_pypartn(Config.from_ini_file(_input(tmp_path, style, "artn")))


def test_with_pypartn_installed_partn_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With pypARTn importable, a pARTn run passes the check."""
    monkeypatch.setitem(sys.modules, "pypARTn", types.ModuleType("pypARTn"))
    check_pypartn(Config.from_ini_file(_input(tmp_path, "partn", None)))


def test_run_stops_before_any_engine_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_pypartn: None
) -> None:
    """run.main refuses a pARTn input before it builds the engine factory."""
    built = []
    monkeypatch.setattr(run, "EngineManagerFactory", lambda **kw: built.append(kw))
    monkeypatch.setattr(sys, "argv", ["pykmc", "-in", _input(tmp_path, "artn", None)])
    with pytest.raises(ValueError, match="pypARTn"):
        run.main()
    assert built == []
