"""``r_push_dist_thr`` under ``style = partn`` refinement.

``partn_refine`` sends ``push_mode = r_push_mode`` to pARTn, so the radius it
sends with it, ``r_push_dist_thr``, must follow ``r_push_mode`` too. Keyed on
the search's ``push_mode`` instead, ``r_push_mode = rad`` with
``push_mode = list`` left pARTn's radius at its default of 0, so only the
central atom was pushed; ``r_push_mode = list`` with ``push_mode = rad`` sent
a radius pARTn does not use.

The tests drive ``partn_refine`` with a recording ``lmp`` and a stub pARTn
whose refinement fails at once, and read the ``set`` calls it receives.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("lammps")

from pykmc.engine import lammps as lammps_engine  # noqa: E402
from pykmc.engine.lammps import LammpsEngine  # noqa: E402

PUSH_DIST_THR = 9.0
R_PUSH_DIST_THR = 2.5


class _Recorder:
    """Stand-in for the ``lammps`` object: record every command string."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    def command(self, cmd: str) -> None:
        self.commands.append(cmd)


class _StubArtn:
    """pARTn stand-in that records its ``set`` calls and fails at once."""

    lib = SimpleNamespace(_name="stub-artn")
    calls: list[tuple[str, object]] = []

    def __init__(self, engine: str) -> None:
        pass

    def reset_input(self) -> None:
        pass

    def set(self, key: str, value: object) -> None:
        _StubArtn.calls.append((key, value))

    def get_error(self) -> tuple[int, str]:
        return (1, "stub refinement")


class _Params:
    """pARTn parameters: only the push keys matter to these tests."""

    r_nperp = None
    r_nperp_limitation = None
    r_max_attempts = 1
    push_dist_thr = PUSH_DIST_THR
    r_push_dist_thr = R_PUSH_DIST_THR

    def __init__(self, push_mode: str, r_push_mode: str) -> None:
        self.push_mode = push_mode
        self.r_push_mode = r_push_mode

    def __getattr__(self, name: str) -> int:
        return 1


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> LammpsEngine:
    """Engine with a recording ``lmp`` and the stub pARTn."""
    monkeypatch.setattr(lammps_engine, "pypARTn", SimpleNamespace(artn=_StubArtn))
    monkeypatch.setattr(_StubArtn, "calls", [])
    eng = LammpsEngine(config=None)
    eng.lmp = _Recorder()
    return eng


def _refine_sets(engine: LammpsEngine, push_mode: str, r_push_mode: str) -> dict:
    config = SimpleNamespace(
        frozen_atoms=None,
        control=SimpleNamespace(active_volume=False),
        partn=_Params(push_mode, r_push_mode),
    )
    engine.partn_refine(config, 0, saddle_idx=[0])
    keys = [key for key, _ in _StubArtn.calls]
    assert keys.count("push_dist_thr") <= 1
    return dict(_StubArtn.calls)


@pytest.mark.parametrize("push_mode", ["list", "rad"])
def test_rad_refinement_sends_r_push_dist_thr(
    engine: LammpsEngine, push_mode: str
) -> None:
    """``r_push_mode = rad`` sends ``r_push_dist_thr``, whatever ``push_mode``."""
    sets = _refine_sets(engine, push_mode, r_push_mode="rad")
    assert sets["push_mode"] == "rad"
    assert sets.get("push_dist_thr") == R_PUSH_DIST_THR


@pytest.mark.parametrize("push_mode", ["list", "rad"])
def test_list_refinement_sends_no_push_dist_thr(
    engine: LammpsEngine, push_mode: str
) -> None:
    """``r_push_mode = list`` sends no radius, whatever ``push_mode``."""
    sets = _refine_sets(engine, push_mode, r_push_mode="list")
    assert sets["push_mode"] == "list"
    assert "push_dist_thr" not in sets
