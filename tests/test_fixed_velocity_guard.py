"""Independent pure protocol for a temporary post-ARTn velocity guard.

The real proposed LammpsEngine._fixed_velocity_guard context manager runs
against this recording native API. No LAMMPS constructor or minimizer runs.
This establishes callback/mapping/resource behavior, not native physics.
"""

from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from pykmc.engine.lammps import LammpsEngine


class BoundaryFailure(RuntimeError):
    pass


class RecordingEndpoint:
    """Only the installed native APIs relevant to fix external are exposed."""

    def __init__(self):
        self.commands = [
            "fix 10 all artn dmax 0.1",
            "fix f_buffer_post buffer setforce 0.0 0.0 0.0",
        ]
        self.fixes = {"10", "f_buffer_post", "user_fix"}
        self.groups = {"all", "buffer", "user_group"}
        self.user_callback = {"function": object(), "caller": object()}
        self.callback = {"user_fix": self.user_callback}
        self.numpy = SimpleNamespace(extract_atom=self.extract_atom)
        self.velocity = np.zeros((5, 3))
        self.force = np.zeros((5, 3))
        self.extract_failure = None
        self.registration_failure = None
        self.unfix_failures = {}
        self.escaped_callback_errors = []

    def get_natoms(self):
        return 5

    def has_id(self, kind, name):
        return name in self.available_ids(kind)

    def available_ids(self, kind):
        if kind == "fix":
            return sorted(self.fixes)
        if kind == "group":
            return sorted(self.groups)
        return []

    def command(self, command):
        fields = command.split()
        self.commands.append(command)
        if fields[0] == "fix":
            name = fields[1]
            assert name not in self.fixes, "must not replace a preexisting fix"
            assert fields[2:] == ["all", "external", "pf/callback", "1", "1"]
            self.fixes.add(name)
        elif fields[0] == "unfix":
            name = fields[1]
            assert name not in {"10", "f_buffer_post", "user_fix"}
            if name in self.unfix_failures:
                raise self.unfix_failures[name]
            self.fixes.remove(name)
        else:
            raise AssertionError(f"unexpected velocity-guard command: {command}")

    def set_fix_external_callback(self, fix_id, callback, caller=None):
        assert fix_id in self.fixes
        # Installed wrapper retains this entry before its C registration call.
        self.callback[fix_id] = {"function": callback, "caller": caller}
        if self.registration_failure is not None:
            raise self.registration_failure

    def extract_atom(self, name, *args, **kwargs):
        assert name in {"v", "f"}
        if name == "v" and self.extract_failure is not None:
            raise self.extract_failure
        return self.velocity if name == "v" else self.force

    def fire(self, fix_id, tags, positions, added_force):
        record = self.callback[fix_id]
        try:
            record["function"](
                record["caller"], 7, len(tags), tags, positions, added_force
            )
        except BaseException as exc:
            # ctypes cannot propagate this exception through the native call.
            # Retain it for an explicit assertion, rather than failing early
            # and accidentally letting a broken guard pass the raises check.
            self.escaped_callback_errors.append(exc)


def make_engine():
    endpoint = RecordingEndpoint()
    engine = LammpsEngine.__new__(LammpsEngine)
    engine.lmp = endpoint
    engine.comm = None
    engine.engine_id = 0
    engine._cleared_since_init = False
    engine.full_system = None
    return engine, endpoint


def new_fix(endpoint, before):
    created = endpoint.fixes - before
    assert len(created) == 1
    name = created.pop()
    assert name in endpoint.callback
    assert endpoint.commands.index(f"fix {name} all external pf/callback 1 1") > 1
    return name


def assert_existing_intact(endpoint):
    assert {"10", "f_buffer_post", "user_fix"} <= endpoint.fixes
    assert endpoint.groups == {"all", "buffer", "user_group"}
    assert endpoint.callback["user_fix"] is endpoint.user_callback
    assert not endpoint.escaped_callback_errors


def exception_contains(actual, expected):
    seen = set()
    while actual is not None and id(actual) not in seen:
        if actual is expected:
            return True
        seen.add(id(actual))
        actual = actual.__cause__ or actual.__context__
    return False


def test_callback_uses_current_native_tags_and_never_moves_coordinates_or_free_rows():
    engine, endpoint = make_engine()
    before = endpoint.fixes.copy()
    # These are rows of the current native/cropped source, not global IDs.
    # Required native tags are 1 and 4, irrespective of current local order.
    with engine._fixed_velocity_guard((0, 3)):
        fix_id = new_fix(endpoint, before)
        for tag_values in ([4, 2, 1], [1, 4, 3], [2, 3, 5], []):
            tags = np.array(tag_values, dtype=np.int64)
            nlocal = len(tags)
            # Two extra native-storage rows represent ghosts/unused capacity.
            endpoint.velocity = (
                np.arange((nlocal + 2) * 3, dtype=float).reshape(-1, 3) + 0.25
            )
            initial_velocity = endpoint.velocity.copy()
            positions = np.arange(nlocal * 3, dtype=float).reshape(-1, 3) + 8.5
            initial_positions = positions.copy()
            initial_tags = tags.copy()
            added_force = np.full((nlocal, 3), 91.25)
            endpoint.fire(fix_id, tags, positions, added_force)
            expected = initial_velocity.copy()
            for row, native_id in enumerate(tag_values):
                if native_id in (1, 4):
                    expected[row] = 0.0
            np.testing.assert_array_equal(endpoint.velocity, expected)
            np.testing.assert_array_equal(positions, initial_positions)
            np.testing.assert_array_equal(tags, initial_tags)
            np.testing.assert_array_equal(added_force, np.zeros_like(added_force))
            assert not endpoint.escaped_callback_errors
    assert endpoint.fixes == before
    assert set(endpoint.callback) == {"user_fix"}
    assert_existing_intact(endpoint)


def test_nested_guards_have_distinct_owned_resources_and_cleanup_only_their_own():
    engine, endpoint = make_engine()
    before = endpoint.fixes.copy()
    with engine._fixed_velocity_guard((0,)):
        outer_id = new_fix(endpoint, before)
        outer_callback = endpoint.callback[outer_id]
        with engine._fixed_velocity_guard((3,)):
            inner_id = new_fix(endpoint, before | {outer_id})
            assert inner_id != outer_id
        assert endpoint.fixes == before | {outer_id}
        assert endpoint.callback[outer_id] is outer_callback
        assert inner_id not in endpoint.callback
    assert endpoint.fixes == before
    assert set(endpoint.callback) == {"user_fix"}
    assert_existing_intact(endpoint)


def test_callback_error_is_retained_until_body_returns_and_then_rejected():
    engine, endpoint = make_engine()
    before = endpoint.fixes.copy()
    problem = BoundaryFailure("injected live velocity extraction failure")
    endpoint.extract_failure = problem
    body_finished = False
    with pytest.raises(Exception) as caught:
        with engine._fixed_velocity_guard((0,)):
            fix_id = new_fix(endpoint, before)
            added_force = np.full((1, 3), 99.0)
            endpoint.fire(fix_id, np.array([1]), np.ones((1, 3)), added_force)
            np.testing.assert_array_equal(added_force, np.zeros((1, 3)))
            assert not endpoint.escaped_callback_errors
            body_finished = True
    assert body_finished
    assert exception_contains(caught.value, problem)
    assert endpoint.fixes == before
    assert set(endpoint.callback) == {"user_fix"}
    assert_existing_intact(endpoint)


def test_body_failure_remains_the_initiating_exception_and_cleans_owned_resources():
    engine, endpoint = make_engine()
    before = endpoint.fixes.copy()
    problem = BoundaryFailure("injected minimizer failure")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(BoundaryFailure) as caught:
            with engine._fixed_velocity_guard((0,)):
                new_fix(endpoint, before)
                raise problem
    assert caught.value is problem
    assert endpoint.fixes == before
    assert set(endpoint.callback) == {"user_fix"}
    assert_existing_intact(endpoint)


@pytest.mark.parametrize("body_fails", [False, True], ids=["success", "failure"])
def test_unfix_failure_is_explicit_and_retains_callable_while_native_fix_is_live(
    body_fails,
):
    engine, endpoint = make_engine()
    before = endpoint.fixes.copy()
    cleanup = BoundaryFailure("injected unfix failure: native fix remains live")
    primary = BoundaryFailure("injected minimizer failure before failed cleanup")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(Exception) as caught:
            with engine._fixed_velocity_guard((0,)):
                fix_id = new_fix(endpoint, before)
                callback_record = endpoint.callback[fix_id]
                endpoint.unfix_failures[fix_id] = cleanup
                if body_fails:
                    raise primary
    if body_fails:
        assert caught.value is primary
        assert "unfix" in " ".join(getattr(primary, "__notes__", ())).lower()
    else:
        assert exception_contains(caught.value, cleanup)
    assert fix_id in endpoint.fixes
    assert endpoint.callback[fix_id] is callback_record
    # The surviving native fix still has an actual usable Python callable.
    endpoint.velocity = np.ones((2, 3))
    added_force = np.full((1, 3), 71.0)
    endpoint.fire(fix_id, np.array([1]), np.zeros((1, 3)), added_force)
    np.testing.assert_array_equal(endpoint.velocity[0], np.zeros(3))
    np.testing.assert_array_equal(endpoint.velocity[1], np.ones(3))
    np.testing.assert_array_equal(added_force, np.zeros((1, 3)))
    assert_existing_intact(endpoint)


def test_registration_failure_cleans_both_partial_native_and_python_resources():
    engine, endpoint = make_engine()
    before = endpoint.fixes.copy()
    problem = BoundaryFailure("injected external callback registration failure")
    endpoint.registration_failure = problem
    body_entered = False
    with pytest.raises(Exception) as caught:
        with engine._fixed_velocity_guard((0,)):
            body_entered = True
    assert not body_entered
    assert exception_contains(caught.value, problem)
    assert endpoint.fixes == before
    assert set(endpoint.callback) == {"user_fix"}
    assert_existing_intact(endpoint)
