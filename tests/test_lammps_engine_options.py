"""Tests for the ``[Lammps]`` keys that choose the LAMMPS library and its command-line arguments.

``library`` is the ``name`` argument of ``lammps(...)``: ``library = mtpkk`` opens
``liblammps_mtpkk.so``. ``cmdargs`` is appended to the arguments pyKMC passes itself
(``-screen none -log <file>``), for example ``-k on g 1 -sf kk -pk kokkos newton on neigh half``
for a KOKKOS build on one GPU. With both unset the engine opens LAMMPS as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase.cell import Cell

from pykmc.config import Config, LammpsConfig

_INPUT = Path(__file__).resolve().parent / "data" / "input.in"
_KOKKOS = "-k on g 1 -sf kk -pk kokkos newton on neigh half"


def _ini_with_lammps_lines(tmp_path: Path, *lines: str) -> Path:
    """Copy ``tests/data/input.in`` with extra lines after ``pair_coeff`` in ``[Lammps]``."""
    out = []
    for line in _INPUT.read_text().splitlines():
        out.append(line)
        if line.lstrip().startswith("pair_coeff"):
            out.extend(lines)
    path = tmp_path / "input.in"
    path.write_text("\n".join(out) + "\n")
    return path


def test_keys_are_unset_by_default() -> None:
    config = Config.from_ini_file(str(_INPUT))
    assert config.lammps.library is None
    assert config.lammps.cmdargs is None


def test_keys_parse(tmp_path: Path) -> None:
    path = _ini_with_lammps_lines(tmp_path, "library = mtpkk", f"cmdargs = {_KOKKOS}")
    config = Config.from_ini_file(str(path))
    assert config.lammps.library == "mtpkk"
    assert config.lammps.cmdargs == _KOKKOS.split()


def test_search_cmdargs_parses(tmp_path: Path) -> None:
    assert Config.from_ini_file(str(_INPUT)).lammps.search_cmdargs is None
    path = _ini_with_lammps_lines(tmp_path, f"search_cmdargs = {_KOKKOS}")
    assert Config.from_ini_file(str(path)).lammps.search_cmdargs == _KOKKOS.split()


def test_cmdargs_keeps_a_quoted_argument_whole() -> None:
    cfg = LammpsConfig(
        pair_style="lj/cut 6.0", pair_coeff="* * 1 1", cmdargs='-var name "a b"'
    )
    assert cfg.cmdargs == ["-var", "name", "a b"]


@dataclass
class _OldConfig:
    """A config object written before the keys existed (as the engine test stubs are)."""

    pair_style: str = "lj/cut 6.0"
    pair_coeff: list[str] = field(default_factory=lambda: ["* * 1 1"])
    min_style: str = "cg"
    minimize: str = "1e-6 1e-8 10 10"
    frz_min: str = "1e-6 1e-8 10 10"
    verbosity: int = 0


class _FakeLammps:
    """Stand-in for the ``lammps`` object: records commands and reports whether KOKKOS is on."""

    def __init__(self, kokkos: bool = False) -> None:
        self.kokkos = kokkos
        self.commands: list[str] = []

    def command(self, cmd: str) -> None:
        self.commands.append(cmd)

    def create_atoms(self, *args) -> None:
        pass

    def extract_setting(self, name: str) -> int:
        assert name == "kokkos_active"
        return int(self.kokkos)


def _started_engine(monkeypatch, cfg, engine_id=0, kokkos=False, **engine_kwargs):
    """Start a ``LammpsEngine`` with ``lammps(...)`` replaced; return it and the keyword arguments it passed."""
    pytest.importorskip("lammps")
    import pykmc.engine.lammps as engine_module

    seen = {}

    def fake_lammps(*args, **kwargs):
        assert not args
        seen.update(kwargs)
        return _FakeLammps(kokkos)

    monkeypatch.setattr(engine_module, "lammps", fake_lammps)
    engine = engine_module.LammpsEngine(
        config=cfg, engine_id=engine_id, **engine_kwargs
    )
    engine.start()
    return engine, seen


def _start_and_record(monkeypatch, cfg, engine_id=0) -> dict:
    return _started_engine(monkeypatch, cfg, engine_id)[1]


@pytest.mark.parametrize(
    "cfg",
    [
        _OldConfig(),
        LammpsConfig(pair_style="lj/cut 6.0", pair_coeff="* * 1 1", verbosity=0),
    ],
)
def test_start_with_keys_unset_opens_lammps_as_before(monkeypatch, cfg) -> None:
    seen = _start_and_record(monkeypatch, cfg)
    assert seen == {"comm": None, "cmdargs": ["-screen", "none", "-log", "none"]}


def test_start_passes_library_and_cmdargs(monkeypatch) -> None:
    cfg = LammpsConfig(
        pair_style="mtp pot.mtp chunksize 32768",
        pair_coeff="* *",
        verbosity=1,
        library="mtpkk",
        cmdargs=_KOKKOS,
    )
    seen = _start_and_record(monkeypatch, cfg, engine_id="dimer.3")
    assert seen == {
        "name": "mtpkk",
        "comm": None,
        "cmdargs": ["-screen", "none", "-log", "lammps.log.dimer.3", *_KOKKOS.split()],
    }


@pytest.mark.parametrize("search_instance", [False, True])
def test_search_cmdargs_go_to_a_search_instance_only(
    monkeypatch, search_instance
) -> None:
    cfg = LammpsConfig(
        pair_style="mtp pot.mtp chunksize 32768",
        pair_coeff="* *",
        verbosity=0,
        cmdargs="-echo none",
        search_cmdargs=_KOKKOS,
    )
    _, seen = _started_engine(monkeypatch, cfg, search_instance=search_instance)
    expected = ["-screen", "none", "-log", "none", "-echo", "none"]
    assert seen["cmdargs"] == expected + (_KOKKOS.split() if search_instance else [])


@pytest.mark.parametrize("kokkos", [False, True])
def test_no_labelmap_under_kokkos(monkeypatch, kokkos) -> None:
    """LAMMPS refuses ``labelmap`` when KOKKOS is on, so the engine leaves it out there."""
    engine, _ = _started_engine(monkeypatch, _OldConfig(), kokkos=kokkos)
    cell = np.diag([10.0, 10.0, 10.0])
    x = np.array([[1.0, 1.0, 1.0], [4.0, 4.0, 4.0]])
    engine.initialize_system(
        types=["Zr", "Zr"], positions=x, cell=Cell(cell), pbc=[True] * 3
    )
    assert any(c.startswith("labelmap") for c in engine.lmp.commands) != kokkos

    pytest.importorskip("pypARTn")
    from pykmc.activevolume import active_volume as av

    engine.lmp.commands.clear()
    av.initialize_system(
        engine, SimpleNamespace(types=["Zr", "Zr"], cell=cell, positions=x)
    )
    assert any(c.startswith("labelmap") for c in engine.lmp.commands) != kokkos
