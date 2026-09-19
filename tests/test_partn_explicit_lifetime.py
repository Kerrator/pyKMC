"""Independent explicit ARTn-owner lifetime oracle; no native launch.

Recorder and public-call seams derive from frozen buffer protocol v3. The real
public search/refine implementation bodies run. Extract returns independent
copies exactly as installed pypARTn.get_data does; destroy poisons only internal
native-like backing. Strong references deliberately prevent garbage collection.
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
    name = "r04_partn_explicit_lifetime_protocol_v1"

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

    def get_positions(self):
        # The full-system failure transaction snapshots this unchanged source.
        return POSITIONS.copy()

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


class DestructionFailure(RuntimeError):
    pass


class OwnedARTn(FakeARTn):
    def __init__(self, recorder, destroy_fails=False):
        super().__init__(recorder)
        self._alive = True
        self.destroy_calls = 0
        self.history = []
        self.destroy_problem = DestructionFailure("explicit destroy failed sentinel")
        self.destroy_fails = destroy_fails
        self.tau = {}
        for name, displacement in (
            ("tau_min1", 0.0),
            ("tau_sad", 0.03),
            ("tau_min2", 0.06),
        ):
            self.tau[name] = POSITIONS.copy()
            self.tau[name][0, 0] += displacement
        self.expected = {name: value.copy() for name, value in self.tau.items()}

    def get_error(self):
        assert self._alive, "error extraction must precede explicit destruction"
        self.history.append("get_error")
        return super().get_error()

    def extract(self, name):
        assert self._alive, "data extraction must precede explicit destruction"
        self.history.append(f"extract:{name}")
        if name in self.tau:
            # Real pypARTn.get_data makes a hard copy before freeing C data.
            return self.tau[name].copy()
        return super().extract(name)

    def destroy(self):
        self.destroy_calls += 1
        self.history.append("destroy")
        if self.destroy_fails:
            raise self.destroy_problem
        assert self._alive, "this operation must explicitly destroy its owner once"
        for value in self.tau.values():
            value.fill(np.nan)
        self._alive = False


def owned_setup(monkeypatch, outcome, active=True, destroy_fails=False):
    engine, cfg, recorder = setup(monkeypatch, outcome, active_volume=active)
    owned = []

    def factory(**kwargs):
        assert kwargs == {"engine": "lmp"}
        obj = OwnedARTn(recorder, destroy_fails=destroy_fails)
        owned.append(obj)  # Strong ownership excludes __del__/GC as a solution.
        return obj

    monkeypatch.setattr(engine_module, "pypARTn", SimpleNamespace(artn=factory))
    return engine, cfg, recorder, owned


@pytest.mark.parametrize("operation", ["search", "refine"])
def test_retained_exception_does_not_retain_live_artn_run_state(monkeypatch, operation):
    engine, cfg, recorder, owned = owned_setup(monkeypatch, "raise")
    retained = []
    try:
        invoke(engine, cfg, operation)
    except MinimizeFailure as exc:
        retained.append(exc)
    assert len(retained) == 1 and retained[0].__traceback__ is not None
    assert recorder.minimizations and len(owned) == 1
    assert owned[0].destroy_calls == 1, (
        "cleanup must not depend on traceback collection"
    )
    assert not owned[0]._alive
    assert owned[0].history[-1] == "destroy"
    assert all(np.isnan(value).all() for value in owned[0].tau.values())
    assert len(engine.restore_calls) == 1
    assert recorder.fixes == {} and recorder.callback == {}


@pytest.mark.parametrize("operation", ["search", "refine"])
@pytest.mark.parametrize("active", [True, False], ids=["av", "full-system"])
def test_success_extracts_outputs_before_destroy_and_keeps_independent_copies(
    monkeypatch, operation, active
):
    engine, cfg, recorder, owned = owned_setup(monkeypatch, "ok", active=active)
    result = invoke(engine, cfg, operation)
    assert result.is_ok()
    assert recorder.minimizations and len(owned) == 1
    obj = owned[0]
    assert obj.destroy_calls == 1 and not obj._alive
    assert obj.history[-1] == "destroy"
    assert all(np.isnan(value).all() for value in obj.tau.values())
    output = result.ok_value()
    fields = {"saddle_positions": "tau_sad"}
    if operation == "search":
        fields.update(min1_positions="tau_min1", min2_positions="tau_min2")
        assert output.dE_forward == pytest.approx(0.2)
        assert output.dE_backward == pytest.approx(0.2)
    else:
        assert output.E_saddle == pytest.approx(0.2 if active else -0.8)
    for field, native_field in fields.items():
        assert f"extract:{native_field}" in obj.history
        np.testing.assert_array_equal(
            getattr(output, field), obj.expected[native_field]
        )
        assert not np.shares_memory(getattr(output, field), obj.tau[native_field])


@pytest.mark.parametrize("operation", ["search", "refine"])
def test_err_return_also_destroys_owned_artn_state(monkeypatch, operation):
    engine, cfg, recorder, owned = owned_setup(monkeypatch, "err")
    result = invoke(engine, cfg, operation)
    assert not result.is_ok()
    assert recorder.minimizations and len(owned) == 1
    assert owned[0].destroy_calls == 1 and not owned[0]._alive
    assert "get_error" in owned[0].history[:-1]
    assert owned[0].history[-1] == "destroy"


@pytest.mark.parametrize("operation", ["search", "refine"])
def test_destroy_failure_preserves_retained_initiating_exception(
    monkeypatch, operation
):
    engine, cfg, recorder, owned = owned_setup(monkeypatch, "raise", destroy_fails=True)
    retained = []
    try:
        invoke(engine, cfg, operation)
    except MinimizeFailure as exc:
        retained.append(exc)
    assert len(retained) == 1 and retained[0].__traceback__ is not None
    assert recorder.minimizations and len(owned) == 1
    assert owned[0].destroy_calls == 1
    assert "injected pARTn minimization failure" in str(retained[0])
    if hasattr(BaseException, "add_note"):
        assert "explicit destroy failed sentinel" in " ".join(retained[0].__notes__)


@pytest.mark.parametrize("operation", ["search", "refine"])
def test_destroy_failure_after_success_rejects_instead_of_returning_ok(
    monkeypatch, operation
):
    engine, cfg, recorder, owned = owned_setup(monkeypatch, "ok", destroy_fails=True)
    with pytest.raises(DestructionFailure) as caught:
        invoke(engine, cfg, operation)
    assert recorder.minimizations and len(owned) == 1
    assert owned[0].destroy_calls == 1
    assert caught.value is owned[0].destroy_problem
