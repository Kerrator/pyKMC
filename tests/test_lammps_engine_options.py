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

    def close(self) -> None:
        pass


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


def test_search_pair_style_parses(tmp_path: Path) -> None:
    assert Config.from_ini_file(str(_INPUT)).lammps.search_pair_style is None
    path = _ini_with_lammps_lines(
        tmp_path, "search_pair_style = mtp/small pot.mtp chunksize 32768"
    )
    config = Config.from_ini_file(str(path))
    assert config.lammps.search_pair_style == "mtp/small pot.mtp chunksize 32768"


@pytest.mark.parametrize("search_instance", [False, True])
def test_search_pair_style_goes_to_a_search_instance_only(
    monkeypatch, search_instance
) -> None:
    cfg = LammpsConfig(
        pair_style="mtp pot.mtp chunksize 32768",
        pair_coeff="* *",
        verbosity=0,
        search_pair_style="mtp/small pot.mtp chunksize 32768",
    )
    engine, _ = _started_engine(monkeypatch, cfg, search_instance=search_instance)
    engine.initialize_potential()
    style = cfg.search_pair_style if search_instance else cfg.pair_style
    assert engine.lmp.commands == [f"pair_style {style}", "pair_coeff * *"]


def test_a_search_instance_keeps_pair_style_without_the_key(monkeypatch) -> None:
    engine, _ = _started_engine(monkeypatch, _OldConfig(), search_instance=True)
    engine.initialize_potential()
    assert engine.lmp.commands[0] == "pair_style lj/cut 6.0"


def test_neighbor_reuse_is_decided_by_the_search_instances_pair_style() -> None:
    from pykmc.config import search_pair_style, supports_neighbor_reuse

    plain = LammpsConfig(pair_style="lj/cut 6.0", pair_coeff="* * 1 1")
    assert search_pair_style(plain) == "lj/cut 6.0"
    assert search_pair_style(_OldConfig()) == "lj/cut 6.0"
    mixed = LammpsConfig(
        pair_style="lj/cut 6.0",
        pair_coeff="* * 1 1",
        search_pair_style="mtp/small pot.mtp chunksize 64",
    )
    assert supports_neighbor_reuse(search_pair_style(mixed))


# ---------------------------------------------------------------------------
# [Lammps] group_cmdargs and group_pair_style: the group instance (lammps.log.0), which does the
# full-cell minimizations (the first one and event execution) and never runs pARTn.
# ---------------------------------------------------------------------------


def test_group_keys_parse(tmp_path: Path) -> None:
    config = Config.from_ini_file(str(_INPUT))
    assert config.lammps.group_cmdargs is None
    assert config.lammps.group_pair_style is None
    path = _ini_with_lammps_lines(
        tmp_path,
        f"group_cmdargs = {_KOKKOS}",
        "group_pair_style = mtp/small pot.mtp chunksize 32768",
    )
    config = Config.from_ini_file(str(path))
    assert config.lammps.group_cmdargs == _KOKKOS.split()
    assert config.lammps.group_pair_style == "mtp/small pot.mtp chunksize 32768"


@pytest.mark.parametrize(
    "engine_kwargs, extra",
    [
        ({}, []),
        ({"group_instance": True}, ["-sf", "kk"]),
        ({"search_instance": True}, ["-sf", "omp"]),
    ],
)
def test_group_cmdargs_go_to_the_group_instance_only(
    monkeypatch, engine_kwargs, extra
) -> None:
    cfg = LammpsConfig(
        pair_style="mtp pot.mtp chunksize 32768",
        pair_coeff="* *",
        verbosity=0,
        cmdargs="-echo none",
        search_cmdargs="-sf omp",
        group_cmdargs="-sf kk",
    )
    _, seen = _started_engine(monkeypatch, cfg, **engine_kwargs)
    assert (
        seen["cmdargs"] == ["-screen", "none", "-log", "none", "-echo", "none"] + extra
    )


@pytest.mark.parametrize(
    "engine_kwargs, style",
    [
        ({}, "mtp pot.mtp chunksize 32768"),
        ({"group_instance": True}, "mtp/small pot.mtp chunksize 32768"),
        ({"search_instance": True}, "mtp pot.mtp chunksize 64"),
    ],
)
def test_group_pair_style_goes_to_the_group_instance_only(
    monkeypatch, engine_kwargs, style
) -> None:
    cfg = LammpsConfig(
        pair_style="mtp pot.mtp chunksize 32768",
        pair_coeff="* *",
        verbosity=0,
        search_pair_style="mtp pot.mtp chunksize 64",
        group_pair_style="mtp/small pot.mtp chunksize 32768",
    )
    engine, _ = _started_engine(monkeypatch, cfg, **engine_kwargs)
    engine.initialize_potential()
    assert engine.lmp.commands == [f"pair_style {style}", "pair_coeff * *"]


def test_the_group_instance_opens_as_before_without_the_keys(monkeypatch) -> None:
    engine, seen = _started_engine(monkeypatch, _OldConfig(), group_instance=True)
    assert seen == {"comm": None, "cmdargs": ["-screen", "none", "-log", "none"]}
    engine.initialize_potential()
    assert engine.lmp.commands[0] == "pair_style lj/cut 6.0"


@pytest.mark.parametrize("mode, group", [("group", True), ("local", False)])
def test_factory_marks_the_group_engine(mode, group) -> None:
    pytest.importorskip("lammps")
    from pykmc.factory import EngineManagerFactory

    factory = EngineManagerFactory.__new__(EngineManagerFactory)
    factory._engine_style = "lammps"
    factory._engine_config = _OldConfig()
    factory._engine_extensions = []
    engine = factory._make_engine(None, mode, 2)
    assert engine.group_instance is group
    assert engine.engine_id == (0 if group else 3)


@pytest.mark.parametrize("min_style, accepted", [("cg", True), ("fire", False)])
def test_group_instance_under_kokkos_needs_min_style_cg(
    monkeypatch, min_style, accepted
) -> None:
    """KOKKOS has `cg` only (cg/kk); LAMMPS would stop at the first minimization, naming `cg/kk`."""
    cfg = _OldConfig(min_style=min_style)
    if accepted:
        _started_engine(monkeypatch, cfg, kokkos=True, group_instance=True)
    else:
        with pytest.raises(ValueError, match="min_style"):
            _started_engine(monkeypatch, cfg, kokkos=True, group_instance=True)
    # Without KOKKOS, and on the other instances, min_style is not checked.
    _started_engine(monkeypatch, cfg, kokkos=False, group_instance=True)
    _started_engine(monkeypatch, cfg, kokkos=True)
