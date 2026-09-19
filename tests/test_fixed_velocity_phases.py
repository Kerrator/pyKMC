"""Independent rank-scripted guard protocol; no MPI/native constructor runs.

The actual engine context manager receives a communicator whose peer outcome
is attached to an observable native phase, not a production call count. Each
local/remote case represents one rank's view of a two-rank operation. These
tests require phase ordering, not true MPI progress or native error recovery.
V2 retains callback errors through bounded native completion without soft stop.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from pykmc.engine.lammps import LammpsEngine


class BoundaryFailure(RuntimeError):
    pass


class ScriptedComm:
    def __init__(self, native, remote_phase=None):
        self.native = native
        self.remote_phase = remote_phase
        self.calls = []

    def Get_size(self):
        return 2

    def Get_rank(self):
        return 0

    def allgather(self, value):
        phase = self.native.phase
        self.calls.append((phase, value))
        remote = None
        if isinstance(value, (bool, np.bool_)):
            remote = False
        elif phase == self.remote_phase:
            remote = f"BoundaryFailure('remote {phase} failure')"
        return [value, remote]


class Endpoint:
    def __init__(self):
        self.phase = "entry"
        self.fixes = {"user_fix"}
        self.user_callback = {"function": object(), "caller": object()}
        self.callback = {"user_fix": self.user_callback}
        self.numpy = SimpleNamespace(extract_atom=self.extract_atom)
        self.velocity = np.ones((3, 3))
        self.failure_phase = None
        self.failure = BoundaryFailure("local boundary failure")
        self.escaped = []
        self.stop_phases = []
        self.extract_calls = 0

    def get_natoms(self):
        return 3

    def has_id(self, kind, name):
        assert kind == "fix"
        return name in self.fixes

    def command(self, command):
        words = command.split()
        if words[0] == "fix":
            assert words[1] not in self.fixes
            assert words[2:] == ["all", "external", "pf/callback", "1", "1"]
            self.phase = "allocation"
            self.fixes.add(words[1])
        elif words[0] == "unfix":
            assert words[1] != "user_fix"
            self.phase = "cleanup"
            if self.failure_phase == self.phase:
                raise self.failure
            self.fixes.remove(words[1])
        else:
            raise AssertionError(command)

    def set_fix_external_callback(self, name, callback, caller=None):
        self.phase = "registration"
        self.callback[name] = {"function": callback, "caller": caller}
        if self.failure_phase == self.phase:
            raise self.failure

    def extract_atom(self, name, nelem, dim):
        self.extract_calls += 1
        assert name == "v" and dim == 3
        if self.failure_phase == "callback":
            raise self.failure
        return self.velocity[:nelem]

    def force_timeout(self):
        self.stop_phases.append(self.phase)

    def fire(self, empty=False):
        self.phase = "callback"
        names = set(self.callback) - {"user_fix"}
        assert len(names) == 1
        record = self.callback[names.pop()]
        nlocal = 0 if empty else 2
        tags = None if empty else np.array([3, 1], dtype=np.int64)
        positions = None if empty else np.zeros((2, 3))
        added = None if empty else np.full((2, 3), 9.0)
        try:
            record["function"](record["caller"], 5, nlocal, tags, positions, added)
        except BaseException as exc:
            # Record escape as ctypes would; it must not become the test's
            # expected outer context-manager exception.
            self.escaped.append(exc)
        if added is not None:
            np.testing.assert_array_equal(added, np.zeros((2, 3)))


def make_case(phase, location):
    native = Endpoint()
    if location == "local":
        native.failure_phase = phase
    comm = ScriptedComm(native, phase if location == "remote" else None)
    engine = LammpsEngine.__new__(LammpsEngine)
    engine.lmp = native
    engine.comm = comm
    engine.engine_id = 0
    engine.full_system = None
    engine._cleared_since_init = False
    return engine, native, comm


def assert_user_intact(native):
    assert "user_fix" in native.fixes
    assert native.callback["user_fix"] is native.user_callback
    assert not native.escaped, "Python exception escaped a ctypes callback"


@pytest.mark.parametrize("location", ["local", "remote"])
def test_registration_failure_is_agreed_before_any_body_entry(location):
    engine, native, comm = make_case("registration", location)
    body_entered = False
    with pytest.raises(Exception) as caught:
        with engine._fixed_velocity_guard((0,)):
            body_entered = True
            native.phase = "body"
    assert not body_entered, "peer registration failure must stop entry to minimize"
    assert any(phase == "registration" for phase, _ in comm.calls)
    if location == "local":
        assert caught.value is native.failure
    else:
        assert "remote" in str(caught.value) and "registration" in str(caught.value)
    assert native.fixes == {"user_fix"}
    assert set(native.callback) == {"user_fix"}
    assert_user_intact(native)


@pytest.mark.parametrize("location", ["local", "remote"])
def test_cleanup_failure_rejects_and_marks_pending_on_every_rank(location):
    engine, native, comm = make_case("cleanup", location)
    body_finished = False
    with pytest.raises(Exception) as caught:
        with engine._fixed_velocity_guard((0,)):
            native.phase = "body"
            body_finished = True
    assert body_finished
    assert any(phase == "cleanup" for phase, _ in comm.calls)
    assert engine._cleared_since_init, "a peer's failed cleanup also requires recovery"
    if location == "local":
        assert caught.value is native.failure
        owned = native.fixes - {"user_fix"}
        assert len(owned) == 1
        assert owned <= set(native.callback), "live native callback must stay retained"
    else:
        assert "remote" in str(caught.value) and "cleanup" in str(caught.value)
        assert native.fixes == {"user_fix"}
        assert set(native.callback) == {"user_fix"}
    assert_user_intact(native)


@pytest.mark.parametrize(
    "location,empty", [("local", False), ("remote", False), ("remote", True)]
)
def test_callback_error_retained_until_bounded_return_without_timer_mutation(
    location, empty
):
    engine, native, comm = make_case("callback", location)
    body_finished = False
    with pytest.raises(Exception) as caught:
        with engine._fixed_velocity_guard((0,)):
            native.fire(empty=empty)
            assert not native.escaped
            assert native.stop_phases == [], "callback must not expire the native timer"
            callback_calls = sum(phase == "callback" for phase, _ in comm.calls)
            assert callback_calls >= 1
            # An already recorded failure must not bypass the next agreement;
            # simulate another native callback before finite native completion.
            native.fire(empty=empty)
            assert sum(phase == "callback" for phase, _ in comm.calls) > callback_calls
            assert not native.escaped
            assert native.stop_phases == [], "retained errors must not mutate the timer"
            native.phase = "body"
            body_finished = True
    assert body_finished, "callback failures must be retained until native returns"
    assert native.stop_phases == [], "neither callback nor cleanup may expire the timer"
    assert "callback" in str(caught.value).lower()
    if empty:
        assert native.extract_calls == 0
    assert native.fixes == {"user_fix"}
    assert set(native.callback) == {"user_fix"}
    assert_user_intact(native)


@pytest.mark.parametrize("location", ["local", "remote"])
def test_returned_body_failure_is_agreed_before_successful_exit(location):
    engine, native, comm = make_case("body", location)
    with pytest.raises(Exception) as caught:
        with engine._fixed_velocity_guard((0,)):
            native.phase = "body"
            if location == "local":
                raise native.failure
    assert any(phase == "body" for phase, _ in comm.calls)
    if location == "local":
        assert caught.value is native.failure
    else:
        assert "remote" in str(caught.value) and "body" in str(caught.value)
    assert native.fixes == {"user_fix"}
    assert set(native.callback) == {"user_fix"}
    assert_user_intact(native)
