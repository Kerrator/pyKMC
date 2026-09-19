"""Public engine ordering/cleanup protocol; no LAMMPS or pARTn instance.

Native commands and the pARTn object are explicit recording substitutes.
Actual public search/refine methods, their implementation bodies, make_AV,
frozen helper decisions and exception/restoration dispatch execute unchanged.
This tests command ordering only; the unchanged native LJ5 gate must prove
that native coordinates/returned tau obey the fixed-buffer contract.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import pykmc.engine.lammps as engine_module
from pykmc.activevolume.active_volume import make_AV
from pykmc.config import PartnConfig
from pykmc.engine.lammps import FullSystem, LammpsEngine


POSITIONS = np.array([[2.0, 2.0, 2.0], [3.0, 2.0, 2.0], [2.0, 3.0, 2.0]])
CELL = np.diag([8.0, 8.0, 8.0])
TYPES = np.array(["Cu"] * 3)


class MinimizeFailure(RuntimeError):
    pass


class Recorder:
    """LAMMPS fix replacement retains insertion order until explicit unfix."""

    def __init__(self, outcome):
        self.outcome = outcome
        self.commands = []
        self.fixes = {}
        self.groups = {"all"}
        self.minimizations = []
        # Adapter v2: installed fix-external Python registry and local arrays.
        self.callback = {}
        self.velocity = np.ones((3, 3))
        self.force = np.zeros((3, 3))
        self.numpy = SimpleNamespace(extract_atom=self.extract_local_atom)

    def get_natoms(self):
        return len(POSITIONS)

    def extract_local_atom(self, name, *args, **kwargs):
        assert name in ("v", "f")
        return self.velocity if name == "v" else self.force

    def set_fix_external_callback(self, fix_id, callback, caller=None):
        assert fix_id in self.fixes
        self.callback[fix_id] = {"function": callback, "caller": caller}

    def command(self, command):
        self.commands.append(command)
        words = command.split()
        if words[0] == "fix":
            self.fixes[words[1]] = tuple(words[2:])
        elif words[0] == "unfix":
            assert words[1] in self.fixes, f"unfix without owned fix: {command}"
            del self.fixes[words[1]]
        elif words[0] == "group":
            if words[2] == "delete":
                self.groups.remove(words[1])
            else:
                self.groups.add(words[1])
        elif words[0] == "clear":
            self.fixes.clear()
            self.groups = {"all"}
        elif words[0] == "minimize":
            self.minimizations.append(list(self.fixes.items()))
            # The real guard runs; only the native callback dispatch is recorded.
            # Native/local order is deliberately different from source row order.
            for fix_id, record in list(self.callback.items()):
                if fix_id in self.fixes:
                    record["function"](
                        record["caller"],
                        1,
                        3,
                        np.array([3, 1, 2]),
                        POSITIONS[[2, 0, 1]].copy(),
                        np.full((3, 3), 19.0),
                    )
            if self.outcome == "raise":
                raise MinimizeFailure("injected pARTn minimization failure")

    def has_id(self, kind, name):
        return name in (self.fixes if kind == "fix" else self.groups)

    def available_ids(self, kind):
        return tuple(self.fixes if kind == "fix" else self.groups)


class FakeARTn:
    def __init__(self, recorder):
        self.recorder = recorder
        self.lib = SimpleNamespace(_name="recorded-partn-not-a-library")

    def reset_input(self):
        pass

    def set(self, *args):
        pass

    def get_error(self):
        if self.recorder.outcome == "err":
            return (1, "declared no convergence")
        if self.recorder.outcome == "retry" and len(self.recorder.minimizations) == 1:
            return (1, "declared first-attempt failure")
        return (0, "")

    def extract(self, name):
        if name in ("tau_min1", "tau_min2", "tau_sad"):
            return POSITIONS.copy()
        return {
            "delr_min1": 0.0,
            "delr_min2": 0.01,
            "delr_sad": 0.0,
            "etot_sad": -0.8,
            "etot_min1": -1.0,
            "etot_min2": -1.0,
        }[name]


class Harness(LammpsEngine):
    name = "r05_partn_buffer_order_protocol_v2"

    def __init__(self, recorder):
        # Deliberately bypass native allocation; all code under review is
        # inherited. Full native replay is an explicit substituted boundary.
        self.lmp = recorder
        self.comm = None
        self.engine_id = 23
        self._is_orthorhombic = True
        self._cleared_since_init = False
        # Adapter v2: public validation now needs actual source cell/PBC/types.
        self.full_system = FullSystem(
            types=tuple(TYPES),
            species=("Cu",),
            masses=(63.546,),
            cell=CELL.copy(),
            pbc=(True, True, True),
        )
        self.restore_calls = []

    def ensure_full_system(self, positions=None):
        self.restore_calls.append(np.array(positions, copy=True))
        self.command("clear")
        self._cleared_since_init = False
        return True

    def set_positions(self, positions):
        np.testing.assert_array_equal(positions, POSITIONS)

    def minimize_freeze_core(self, core_idx):
        # Non-AV control only; unrelated native core preparation is substituted.
        np.testing.assert_array_equal(core_idx, np.arange(3))


def setup(monkeypatch, outcome, active_volume=True):
    recorder = Recorder(outcome)
    engine = Harness(recorder)
    cfg = SimpleNamespace(
        control=SimpleNamespace(active_volume=active_volume),
        # The unchanged identity crop has only row 0 mobile and rows 1/2 fixed.
        activevolume=SimpleNamespace(rmov=0.5, ract=5.0),
        frozen_atoms=None,
        eventsearch=SimpleNamespace(delr_thr=0.1),
        atomicenvironment=SimpleNamespace(rcut=4.0),
        partn=PartnConfig(r_max_attempts=2 if outcome == "retry" else 1),
    )

    def prepare(
        engine,
        _cfg,
        center,
        positions,
        cell,
        types,
        *,
        constraints=None,
        user_constraints=None,
    ):
        assert center == 0
        np.testing.assert_array_equal(positions, POSITIONS)
        np.testing.assert_array_equal(cell, CELL)
        np.testing.assert_array_equal(types, TYPES)
        engine.command("clear")
        # Actual make_AV installs the existing pre-pARTn buffer constraint.
        make_AV(engine, np.arange(3), np.array([1, 2]))
        return np.arange(3), np.array([1])

    def prepare_refine(
        engine,
        cfg,
        center,
        positions,
        cell,
        types,
        idx,
        saddle,
        *,
        constraints=None,
        user_constraints=None,
    ):
        mapping, central = prepare(
            engine,
            cfg,
            center,
            positions,
            cell,
            types,
            constraints=constraints,
            user_constraints=user_constraints,
        )
        np.testing.assert_array_equal(idx, np.arange(3))
        np.testing.assert_array_equal(saddle, POSITIONS)
        return -1.0, mapping, central

    monkeypatch.setattr(engine_module, "partn_search_AV", prepare)
    monkeypatch.setattr(engine_module, "partn_refine_AV", prepare_refine)
    monkeypatch.setattr(
        engine_module, "pypARTn", SimpleNamespace(artn=lambda **_: FakeARTn(recorder))
    )
    return engine, cfg, recorder


def invoke(engine, cfg, operation):
    kwargs = dict(
        config=cfg,
        central_atom_idx=0,
        positions=POSITIONS.copy(),
        cell=CELL.copy(),
        types=TYPES.copy(),
    )
    if operation == "refine":
        kwargs.update(saddle_idx=np.arange(3), saddle_positions=POSITIONS.copy())
    return getattr(engine, f"partn_{operation}")(**kwargs)


def assert_buffer_brackets_every_artn(recorder):
    assert recorder.minimizations, "no actual implementation minimization boundary"
    for attempt, fixes in enumerate(recorder.minimizations):
        artn = [i for i, (_, spec) in enumerate(fixes) if spec[1] == "artn"]
        assert len(artn) == 1, (attempt, fixes)
        buffer = [
            i
            for i, (_, spec) in enumerate(fixes)
            if spec[:2] == ("buffer", "setforce")
            and tuple(float(v) for v in spec[2:]) == (0.0, 0.0, 0.0)
        ]
        assert any(i < artn[0] for i in buffer), (
            "buffer physical forces must be zero before pARTn observes them",
            attempt,
            fixes,
        )
        assert any(i > artn[0] for i in buffer), (
            "pARTn-generated buffer forces need a subsequent setforce constraint",
            attempt,
            fixes,
        )


@pytest.mark.parametrize("operation", ["search", "refine"])
@pytest.mark.parametrize("outcome", ["ok", "err", "raise"])
def test_public_av_artn_has_pre_and_post_buffer_constraints_and_restores(
    monkeypatch, operation, outcome
):
    engine, cfg, recorder = setup(monkeypatch, outcome)
    descriptor = engine.full_system
    if outcome == "raise":
        with pytest.raises(
            MinimizeFailure, match="injected pARTn minimization failure"
        ):
            invoke(engine, cfg, operation)
    else:
        result = invoke(engine, cfg, operation)
        assert result.is_ok() == (outcome == "ok")
    # The public success/Err/exception path must all request source replay.
    assert len(engine.restore_calls) == 1
    np.testing.assert_array_equal(engine.restore_calls[0], POSITIONS)
    assert engine.full_system is descriptor
    assert not engine._cleared_since_init
    assert recorder.fixes == {} and recorder.groups == {"all"}
    assert recorder.callback == {}, "owned Python callbacks must be cleaned too"
    assert_buffer_brackets_every_artn(recorder)


def test_refinement_reinstalls_post_constraint_after_artn_on_every_retry(monkeypatch):
    engine, cfg, recorder = setup(monkeypatch, "retry")
    assert invoke(engine, cfg, "refine").is_ok()
    assert len(recorder.minimizations) == 2
    assert_buffer_brackets_every_artn(recorder)
    assert len(engine.restore_calls) == 1
    assert recorder.fixes == {} and recorder.groups == {"all"}
    assert recorder.callback == {}, "owned Python callbacks must be cleaned too"


@pytest.mark.parametrize("operation", ["search", "refine"])
def test_non_av_operations_do_not_introduce_buffer_resources(monkeypatch, operation):
    engine, cfg, recorder = setup(monkeypatch, "ok", active_volume=False)
    assert invoke(engine, cfg, operation).is_ok()
    assert len(recorder.minimizations) == 1
    assert all(spec[0] != "buffer" for _, spec in recorder.minimizations[0])
    assert recorder.fixes == {} and recorder.groups == {"all"}
    assert recorder.callback == {}, "owned Python callbacks must be cleaned too"
    assert engine.restore_calls == []
