"""Protocol and lifecycle tests for the manager stack that need no MPI launch.

Sessions and the world communicator are replaced by small fakes; the Worker
tests use ``MPI.COMM_SELF`` as the active communicator so ``_dispatch`` runs its
real barrier/gather collectives on a single rank.
"""

from __future__ import annotations

import functools
import inspect
import logging
import threading
import types
from typing import Any

import pytest
from mpi4py import MPI

import pykmc.engine.base as engine_base
import pykmc.manager.worker as worker_module
from pykmc.engine import Engine, EngineExtension
from pykmc.manager import Manager, Session, Worker
from pykmc.manager.manager import RESERVED_OPS
from pykmc.manager.worker import (
    DispatchResult,
    DispatchStatus,
    build_registry,
    is_static_callable,
)

# ----------------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------------


class _FakeSession:
    """Session stand-in: records every message and answers ``call`` locally."""

    def __init__(
        self, session_id: int = 1, gate: threading.Event | None = None
    ) -> None:
        self.session_id = session_id
        self.sent: list[tuple[str, dict]] = []
        self.gate = gate  # when set, call() blocks until the gate is set

    def use_local(self) -> None:
        self.sent.append(("use_local", {}))

    def use_group(self) -> None:
        self.sent.append(("use_group", {}))

    def use_global(self) -> None:
        self.sent.append(("use_global", {}))

    def shutdown(self) -> None:
        self.sent.append(("shutdown", {}))

    def call(self, op_name: str, **kwargs: Any) -> Any:
        self.sent.append((op_name, kwargs))
        if self.gate is not None:
            self.gate.wait()
        if op_name == "fail":
            raise RuntimeError("fake failure")
        return kwargs.get("x")


class _FakeWorldComm:
    """Minimal world communicator: rank 0, records sends, replays scripted recvs."""

    def __init__(self, replies: list[dict] | None = None) -> None:
        self.sent: list[tuple[dict, int, int]] = []
        self.replies = list(replies or [])

    def Get_rank(self) -> int:  # noqa: N802 - mpi4py naming
        return 0

    def Set_errhandler(self, _eh: Any) -> None:  # noqa: N802 - mpi4py naming
        pass

    def send(self, msg: dict, dest: int, tag: int) -> None:
        self.sent.append((msg, dest, tag))

    def recv(self, source: int, tag: int) -> dict:
        return self.replies.pop(0)


def _make_manager(n: int = 1, gate: threading.Event | None = None) -> Manager:
    sessions = [_FakeSession(i + 1, gate) for i in range(n)]
    return Manager(
        local_sessions=sessions,
        global_session=_FakeSession(0),
        group_session=_FakeSession(-1),
    )


# ----------------------------------------------------------------------------
# Reserved control names
# ----------------------------------------------------------------------------


class TestReservedNames:
    """Reserved control names are rejected on every submit path."""

    def test_reserved_set(self) -> None:
        """Reserved set."""
        assert RESERVED_OPS == {"use_local", "use_group", "use_global", "shutdown"}

    def test_reserved_set_is_derived_from_worker_builtins(self) -> None:
        """Reserved set is derived from worker builtins."""
        # One definition: the Manager imports the Worker's set, and the Worker
        # builds its dispatch table from the same tuple, so a control builtin
        # added to the Worker later is reserved automatically.
        assert RESERVED_OPS is worker_module.RESERVED_OPS
        w = Worker(
            local_obj=None,
            local_comm=MPI.COMM_SELF,
            worker_id=0,
            world_comm=_FakeWorldComm(),
        )
        assert RESERVED_OPS == set(w._builtins_op) - {"list_ops"}
        assert list(w._builtins_op) == list(worker_module.BUILTIN_OPS)
        assert all(callable(fn) for fn in w._builtins_op.values())

    @pytest.mark.parametrize("op", sorted(RESERVED_OPS))
    def test_submit_paths_reject_before_sending(self, op: str) -> None:
        """Submit paths reject before sending."""
        mgr = _make_manager()
        mgr.start()
        try:
            with pytest.raises(ValueError, match="control builtin"):
                mgr.submit(op)
            with pytest.raises(ValueError, match="control builtin"):
                mgr.broadcast(op)
            with pytest.raises(ValueError, match="control builtin"):
                mgr.submit_group(op)
            with pytest.raises(ValueError, match="control builtin"):
                mgr.submit_global(op)
            # Wrappers strip the prefix and apply the same rejection.
            with pytest.raises(ValueError, match="control builtin"):
                getattr(mgr, f"group_{op}")()
            with pytest.raises(ValueError, match="control builtin"):
                getattr(mgr, f"global_{op}")()
            if op != "shutdown":  # mgr.shutdown is the real method
                with pytest.raises(ValueError, match="control builtin"):
                    getattr(mgr, op)()
            # Nothing was enqueued or sent, and the mode did not change.
            assert mgr._local_queue.unfinished_tasks == 0
            assert mgr.mode == "local"
            for s in mgr.local_sessions + [mgr.group_session, mgr.global_session]:
                assert s.sent == []
        finally:
            mgr.shutdown()

    def test_list_ops_stays_reachable(self) -> None:
        """List ops stays reachable."""
        mgr = _make_manager()
        mgr.start()
        try:
            mgr.list_ops()  # goes through submit("list_ops")
            assert mgr.local_sessions[0].sent == [("list_ops", {"mode": "local"})]
            mgr.group_list_ops(mode="group")
            assert mgr.group_session.sent[-1] == ("list_ops", {"mode": "group"})
            mgr.global_list_ops(mode="global")
            assert mgr.global_session.sent[-1] == ("list_ops", {"mode": "global"})
        finally:
            mgr.shutdown()

    def test_lifecycle_engine_ops_are_ordinary(self) -> None:
        """Lifecycle engine ops are ordinary."""
        mgr = _make_manager()
        mgr.start()
        try:
            for op in ("start", "close", "initialize_parameters"):
                mgr.broadcast(op)
            assert [n for n, _ in mgr.local_sessions[0].sent] == [
                "start",
                "close",
                "initialize_parameters",
            ]
        finally:
            mgr.shutdown()

    def test_underscore_names_are_not_forwarded(self) -> None:
        """Underscore names are not forwarded."""
        mgr = _make_manager()
        with pytest.raises(AttributeError):
            _ = mgr.__deepcopy__  # dunder probe (copy/pickle) must not submit
        with pytest.raises(AttributeError):
            _ = mgr._not_an_op


# ----------------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------------


class TestLifecycle:
    """Shutdown idempotency and post-shutdown/not-started errors."""

    def test_not_started_error_is_distinct(self) -> None:
        """Not started error is distinct."""
        mgr = _make_manager()
        with pytest.raises(RuntimeError, match="not been started"):
            mgr.submit("echo", x=1)

    def test_shutdown_is_idempotent(self) -> None:
        """Shutdown is idempotent."""
        mgr = _make_manager()
        mgr.start()
        mgr.shutdown()
        mgr.shutdown()
        mgr.shutdown()
        shutdowns = [m for m, _ in mgr.local_sessions[0].sent if m == "shutdown"]
        assert shutdowns == ["shutdown"]
        assert mgr._local_threads == []
        assert mgr._local_queue.unfinished_tasks == 0

    def test_shutdown_without_start(self) -> None:
        """Shutdown without start."""
        mgr = _make_manager()
        mgr.shutdown()
        assert mgr.local_sessions[0].sent == [("shutdown", {})]
        with pytest.raises(RuntimeError, match="shut down"):
            mgr.start()

    def test_post_shutdown_calls_raise(self) -> None:
        """Post shutdown calls raise."""
        mgr = _make_manager()
        mgr.start()
        mgr.shutdown()
        for fn in (
            lambda: mgr.start(),
            lambda: mgr.submit("echo", x=1),
            lambda: mgr.broadcast("echo", x=1),
            lambda: mgr.submit_group("echo", x=1),
            lambda: mgr.submit_global("echo", x=1),
            lambda: mgr.echo(x=1),
            lambda: mgr.group_echo(x=1),
            lambda: mgr.global_echo(x=1),
            lambda: mgr.list_ops(),
        ):
            with pytest.raises(RuntimeError, match="shut down") as exc:
                fn()
            assert "not been started" not in str(exc.value)
        # Nothing reached the sessions after shutdown.
        assert mgr.local_sessions[0].sent == [("shutdown", {})]
        assert mgr.group_session.sent == []
        assert mgr.global_session.sent == []

    def test_shutdown_resumes_after_a_failed_session_send(self) -> None:
        """Shutdown resumes after a failed session send."""

        class _FlakySession(_FakeSession):
            failures_left = 1

            def shutdown(self) -> None:
                if self.failures_left:
                    self.failures_left -= 1
                    raise OSError("MPI send failed")
                super().shutdown()

        sessions = [_FakeSession(1), _FlakySession(2), _FakeSession(3)]
        mgr = Manager(local_sessions=sessions)
        mgr.start()
        with pytest.raises(OSError, match="MPI send failed"):
            mgr.shutdown()
        # Session 1 was notified, sessions 2 and 3 were not: the manager is not
        # closed, no work is accepted, and a retry is not a silent no-op.
        assert mgr._closed is False
        assert [s.sent for s in sessions] == [[("shutdown", {})], [], []]
        with pytest.raises(RuntimeError, match="shut down"):
            mgr.submit("echo", x=1)
        with pytest.raises(RuntimeError, match="shut down"):
            mgr.start()
        mgr.shutdown()
        assert mgr._closed is True
        assert [s.sent for s in sessions] == [[("shutdown", {})]] * 3
        mgr.shutdown()  # now idempotent: nothing is re-sent
        assert [s.sent for s in sessions] == [[("shutdown", {})]] * 3

    def test_shutdown_from_group_mode_switches_back_first(self) -> None:
        """Shutdown from group mode switches back first."""
        mgr = _make_manager()
        mgr.start()
        mgr.submit_group("echo", x=1)
        assert mgr.mode == "group"
        mgr.shutdown()
        assert mgr.mode == "local"
        assert mgr.group_session.sent == [("echo", {"x": 1}), ("use_local", {})]

    def test_pending_jobs_run_before_shutdown_and_errors_reach_futures(self) -> None:
        """Pending jobs run before shutdown and errors reach futures."""
        mgr = _make_manager()
        mgr.start()
        ok = mgr.submit("echo", x=7)
        bad = mgr.submit("fail")
        mgr.shutdown()
        assert ok.result(timeout=5) == 7
        with pytest.raises(RuntimeError, match="fake failure"):
            bad.result(timeout=5)


# ----------------------------------------------------------------------------
# Cancellation accounting
# ----------------------------------------------------------------------------


class TestCancellation:
    """Future cancellation and worker-thread survival."""

    def test_cancel_before_dispatch_skips_job_and_balances_queue(self) -> None:
        """Cancel before dispatch skips job and balances queue."""
        gate = threading.Event()
        mgr = _make_manager(n=1, gate=gate)
        mgr.start()
        try:
            blocker = mgr.submit("echo", x="a")  # occupies the single thread
            victim = mgr.submit("echo", x="b")
            assert victim.cancel() is True
            assert victim.cancelled()
            gate.set()
            assert blocker.result(timeout=5) == "a"
            mgr._local_queue.join()  # would hang if task_done were unbalanced
            assert mgr._local_queue.unfinished_tasks == 0
            # The skipped job was never sent, and the thread is still alive.
            sent_ops = [(n, kw) for n, kw in mgr.local_sessions[0].sent]
            assert ("echo", {"x": "b"}) not in sent_ops
            assert all(t.is_alive() for t in mgr._local_threads)
            after = mgr.submit("echo", x="c")
            assert after.result(timeout=5) == "c"
        finally:
            gate.set()
            mgr.shutdown()
        assert mgr._local_queue.unfinished_tasks == 0

    def test_cancel_after_dispatch_is_refused(self) -> None:
        """Cancel after dispatch is refused."""
        gate = threading.Event()
        mgr = _make_manager(n=1, gate=gate)
        mgr.start()
        try:
            fut = mgr.submit("echo", x=1)
            # Wait until the thread has taken the job (it is blocked on the gate).
            for _ in range(200):
                if fut.running():
                    break
                threading.Event().wait(0.01)
            assert fut.running()
            assert fut.cancel() is False
            gate.set()
            assert fut.result(timeout=5) == 1
        finally:
            gate.set()
            mgr.shutdown()

    def test_worker_thread_survives_unexpected_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Worker thread survives unexpected error."""

        class _Broken(_FakeSession):
            def call(self, op_name: str, **kwargs: Any) -> Any:
                if op_name == "base_exc":
                    raise SystemExit(3)  # not an Exception: reaches the outer guard
                return super().call(op_name, **kwargs)

        mgr = Manager(local_sessions=[_Broken()])
        mgr.start()
        try:
            fut = mgr.submit("base_exc")
            # SystemExit is a BaseException: the inner boundary catches only
            # Exception, and the outer guard must not let the thread die either.
            with pytest.raises(SystemExit):
                fut.result(timeout=5)
            assert any("base_exc" in rec.getMessage() for rec in caplog.records)
            assert all(t.is_alive() for t in mgr._local_threads)
            assert mgr.submit("echo", x=2).result(timeout=5) == 2
        finally:
            mgr.shutdown()


# ----------------------------------------------------------------------------
# Session status handling
# ----------------------------------------------------------------------------


class TestSessionStatus:
    """Session status decoding and fire-and-forget control messages."""

    def _session(self, replies: list[dict]) -> tuple[Session, _FakeWorldComm]:
        comm = _FakeWorldComm(replies)
        return Session(engine_master_rank=1, world_comm=comm), comm

    def test_empty_error_string_is_a_failure(self) -> None:
        """Empty error string is a failure."""
        session, _ = self._session(
            [{"type": "status", "value": {"has_result": False, "error": ""}}]
        )
        with pytest.raises(RuntimeError):
            session.call("op")

    def test_none_error_is_success(self) -> None:
        """None error is success."""
        session, comm = self._session(
            [
                {"type": "status", "value": {"has_result": True, "error": None}},
                {"type": "result", "value": 42},
            ]
        )
        assert session.call("op", a=1) == 42
        sent_msg = comm.sent[0][0]
        assert sent_msg == {"type": "op", "reply": True, "value": {"a": 1}}

    def test_void_status_returns_none(self) -> None:
        """Void status returns none."""
        session, _ = self._session(
            [{"type": "status", "value": {"has_result": False, "error": None}}]
        )
        assert session.call("op") is None

    def test_control_messages_are_fire_and_forget(self) -> None:
        """Control messages are fire and forget."""
        session, comm = self._session([])
        session.use_local()
        session.use_group()
        session.use_global()
        session.shutdown()
        assert [m for m, _, _ in comm.sent] == [
            {"type": "use_local", "reply": False},
            {"type": "use_group", "reply": False},
            {"type": "use_global", "reply": False},
            {"type": "shutdown", "reply": False},
        ]


# ----------------------------------------------------------------------------
# Worker reply contract on a single rank (COMM_SELF)
# ----------------------------------------------------------------------------


class _Ops:
    """Operations covering every reply path, run on COMM_SELF."""

    def __init__(self, comm: MPI.Comm) -> None:
        self.comm = comm

    def void(self) -> None:
        return None

    def unpicklable(self) -> Any:
        return lambda: 0

    def echo(self, x: Any) -> Any:
        return x

    def fail(self) -> None:
        raise ValueError("bad input")

    def fail_silently(self) -> None:
        raise ValueError()  # str(e) == ""


def _worker(comm_world: _FakeWorldComm) -> Worker:
    return Worker(
        local_obj=_Ops(MPI.COMM_SELF),
        local_comm=MPI.COMM_SELF,
        worker_id=0,
        world_comm=comm_world,
    )


def _statuses(comm: _FakeWorldComm) -> list[dict]:
    return [m for m, _, tag in comm.sent if tag == Worker._TAG_STATUS]


def _results(comm: _FakeWorldComm) -> list[dict]:
    return [m for m, _, tag in comm.sent if tag == Worker._TAG_RESULT]


class TestWorkerReplies:
    """Worker reply contract: exactly one terminal message per request."""

    def _roundtrip(self, msg: dict) -> _FakeWorldComm:
        comm = _FakeWorldComm()
        w = _worker(comm)
        r = w._dispatch(msg)
        w._send_result(r, msg.get("type"), reply=msg.get("reply", True))
        return comm

    def test_none_result_gets_exactly_one_void_status(self) -> None:
        """None result gets exactly one void status."""
        comm = self._roundtrip({"type": "void", "reply": True})
        assert _statuses(comm) == [
            {"type": "status", "value": {"has_result": False, "error": None}}
        ]
        assert _results(comm) == []

    def test_value_result_gets_status_then_result(self) -> None:
        """Value result gets status then result."""
        comm = self._roundtrip({"type": "echo", "value": {"x": 5}, "reply": True})
        assert _statuses(comm) == [
            {"type": "status", "value": {"has_result": True, "error": None}}
        ]
        assert _results(comm) == [{"type": "result", "value": 5}]

    def test_unpicklable_result_is_an_error_status_naming_the_op(self) -> None:
        """Unpicklable result is an error status naming the op."""
        comm = self._roundtrip({"type": "unpicklable", "reply": True})
        statuses = _statuses(comm)
        assert len(statuses) == 1 and _results(comm) == []
        err = statuses[0]["value"]["error"]
        assert statuses[0]["value"]["has_result"] is False
        assert "unpicklable" in err and "could not be pickled" in err
        assert "PicklingError" in err or "AttributeError" in err

    def test_error_status_carries_type_and_message(self) -> None:
        """Error status carries type and message."""
        comm = self._roundtrip({"type": "fail", "reply": True})
        err = _statuses(comm)[0]["value"]["error"]
        assert "ValueError: bad input" in err and "rank 0" in err
        assert _results(comm) == []

    def test_empty_exception_message_still_reported(self) -> None:
        """Empty exception message still reported."""
        comm = self._roundtrip({"type": "fail_silently", "reply": True})
        err = _statuses(comm)[0]["value"]["error"]
        assert err is not None and "ValueError" in err

    def test_unknown_operation_replies_error(self) -> None:
        """Unknown operation replies error."""
        comm = self._roundtrip({"type": "nope", "reply": True})
        err = _statuses(comm)[0]["value"]["error"]
        assert "Unknown operation 'nope'" in err

    def test_silent_builtin_with_reply_gets_void_status(self) -> None:
        """Silent builtin with reply gets void status."""
        # A control builtin reached with reply=True (not through Manager) must
        # still release the caller.
        comm = self._roundtrip({"type": "use_local", "reply": True})
        assert _statuses(comm) == [
            {"type": "status", "value": {"has_result": False, "error": None}}
        ]

    def test_fire_and_forget_sends_nothing(self) -> None:
        """Fire and forget sends nothing."""
        comm = self._roundtrip({"type": "use_local", "reply": False})
        assert comm.sent == []

    def test_builtin_exception_is_an_error_not_a_crash(self) -> None:
        """Builtin exception is an error not a crash."""
        comm = self._roundtrip(
            {"type": "list_ops", "value": {"mode": "bogus"}, "reply": True}
        )
        err = _statuses(comm)[0]["value"]["error"]
        assert "ValueError" in err and "Unknown mode" in err

    def test_fire_and_forget_error_is_logged_not_dropped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Fire and forget error is logged not dropped."""
        comm = _FakeWorldComm()
        w = _worker(comm)
        r = DispatchResult(DispatchStatus.ERROR, error="TypeError: boom")
        with caplog.at_level(logging.ERROR, logger="pykmc.manager.worker"):
            w._send_result(r, "use_group", reply=False)
        assert comm.sent == []
        assert any("use_group" in rec.getMessage() for rec in caplog.records)

    def test_list_ops_replies_with_value(self) -> None:
        """List ops replies with value."""
        comm = self._roundtrip({"type": "list_ops", "reply": True})
        assert _statuses(comm)[0]["value"] == {"has_result": True, "error": None}
        assert set(_results(comm)[0]["value"]) == {
            "void",
            "unpicklable",
            "echo",
            "fail",
            "fail_silently",
        }


class _Closable:
    """Object whose close() records the call and optionally raises."""

    def __init__(self, name: str, error: Exception | None = None) -> None:
        self.name = name
        self.error = error
        self.closed = False

    def close(self) -> None:
        self.closed = True
        if self.error is not None:
            raise self.error


class _NotClosable:
    """Object without close(): skipped by shutdown."""

    def op(self) -> None:
        pass


def _three_mode_worker(
    local: object, global_: object, group: object, comm: _FakeWorldComm
) -> Worker:
    """One object per mode (a registry list may not repeat a method name)."""
    return Worker(
        local_obj=local,
        local_comm=MPI.COMM_SELF,
        worker_id=4,
        global_obj=global_,
        global_comm=MPI.COMM_SELF,
        group_obj=group,
        group_comm=MPI.COMM_SELF,
        world_comm=comm,
    )


class TestWorkerShutdown:
    """Worker.shutdown closes every object even when a close() raises."""

    def test_every_object_is_closed_and_failures_are_aggregated(self) -> None:
        """Every object is closed and failures are aggregated."""
        first = _Closable("first", OSError("socket gone"))  # local: raises
        second = _Closable("second")  # global: must still be closed
        third = _Closable("third", ValueError())  # group: empty message
        w = _three_mode_worker(first, second, third, _FakeWorldComm())
        with pytest.raises(
            RuntimeError, match="2 close\\(\\) call\\(s\\) failed"
        ) as exc:
            w.shutdown()
        msg = str(exc.value)
        assert "Worker 4" in msg
        assert "OSError: socket gone" in msg and "ValueError" in msg
        assert first.closed and second.closed and third.closed
        assert w._is_alive is False

    def test_objects_without_close_are_skipped(self) -> None:
        """Objects without close are skipped."""
        w = Worker(
            local_obj=_NotClosable(),
            local_comm=MPI.COMM_SELF,
            worker_id=0,
            world_comm=_FakeWorldComm(),
        )
        assert w.shutdown() is None
        assert w._is_alive is False

    def test_dispatch_reports_every_failed_close(self) -> None:
        """Dispatch reports every failed close."""
        comm = _FakeWorldComm()
        bad_a = _Closable("a", OSError("a failed"))
        good = _Closable("b")
        bad_c = _Closable("c", RuntimeError("c failed"))
        w = _three_mode_worker(bad_a, good, bad_c, comm)
        r = w._dispatch({"type": "shutdown", "reply": True})
        assert r.status is DispatchStatus.ERROR
        assert "a failed" in r.error and "c failed" in r.error
        assert good.closed and bad_a.closed and bad_c.closed
        w._send_result(r, "shutdown", reply=True)
        assert _statuses(comm)[0]["value"]["has_result"] is False
        assert "c failed" in _statuses(comm)[0]["value"]["error"]

    def test_clean_shutdown_closes_all_and_returns_silent(self) -> None:
        """Clean shutdown closes all and returns silent."""
        objs = [_Closable("a"), _Closable("b"), _Closable("c")]
        w = _three_mode_worker(*objs, _FakeWorldComm())
        r = w._dispatch({"type": "shutdown", "reply": False})
        assert r.status is DispatchStatus.SILENT
        assert all(o.closed for o in objs)


# ----------------------------------------------------------------------------
# Registry discovery never evaluates properties
# ----------------------------------------------------------------------------


class _Callable:
    """Callable object without ``__get__``: an operation when stored anywhere."""

    def __call__(self) -> int:
        return 1


class _LazyDescriptor:
    """``__get__``-only descriptor: ``inspect.isroutine`` calls it a routine."""

    evaluations = 0

    def __get__(self, obj: Any, owner: type | None = None) -> Any:
        if obj is None:
            return self
        type(self).evaluations += 1
        return 1.5  # not callable: must never be stored as an operation


class _Native:
    """Plain object with every attribute flavour discovery has to handle."""

    evaluations = 0
    lmp = None  # unstarted-engine shape: `cached` would dereference it

    def __init__(self) -> None:
        self.value = 3  # plain instance attribute
        self.dyn = types.MethodType(lambda self, y: y * 2, self)  # added method
        self.callable_obj = _Callable()  # callable stored on the instance

    @property
    def prop(self) -> int:
        type(self).evaluations += 1
        raise RuntimeError("property evaluated during discovery")

    @functools.cached_property
    def cached(self) -> Any:
        type(self).evaluations += 1
        return self.lmp.version()  # AttributeError before start()

    @functools.partialmethod
    def partial_method(self, x: int) -> int:
        return x

    lazy = _LazyDescriptor()

    def method(self, x: int) -> int:
        return x

    @classmethod
    def cm(cls) -> str:
        return "cm"

    @staticmethod
    def sm() -> str:
        return "sm"

    def _private(self) -> None:
        pass


class TestBuildRegistry:
    """build_registry inspects statically and never evaluates properties."""

    def test_properties_are_neither_evaluated_nor_registered(self) -> None:
        """Properties are neither evaluated nor registered."""
        _Native.evaluations = 0
        reg = build_registry(_Native())
        assert _Native.evaluations == 0
        assert "prop" not in reg
        assert "value" not in reg
        assert "_private" not in reg

    def test_cached_property_and_lazy_descriptors_are_never_evaluated(self) -> None:
        """Cached property and lazy descriptors are never evaluated."""
        # functools.cached_property and any __get__-only descriptor satisfy
        # inspect.isroutine; discovery must still leave them alone (the
        # cached_property here would crash on an unstarted object, and the lazy
        # descriptor returns a float that must not become an "operation").
        _Native.evaluations = 0
        _LazyDescriptor.evaluations = 0
        obj = _Native()
        reg = build_registry(obj)  # must not raise
        assert _Native.evaluations == 0
        assert _LazyDescriptor.evaluations == 0
        assert "cached" not in reg and "lazy" not in reg
        assert "partial_method" not in reg
        assert "cached" not in obj.__dict__  # never computed, never cached
        assert all(callable(fn) for fn in reg.values())
        w = Worker(  # construction on the unstarted object succeeds
            local_obj=obj,
            local_comm=MPI.COMM_SELF,
            worker_id=0,
            world_comm=_FakeWorldComm(),
        )
        assert _Native.evaluations == 0 and _LazyDescriptor.evaluations == 0
        assert "cached" not in w.list_ops() and "lazy" not in w.list_ops()

    def test_bound_class_static_and_dynamic_methods_are_registered(self) -> None:
        """Bound class static and dynamic methods are registered."""
        obj = _Native()
        reg = build_registry(obj)
        assert reg["method"](x=4) == 4
        assert reg["method"].__self__ is obj
        assert reg["cm"]() == "cm"
        assert reg["sm"]() == "sm"
        assert reg["dyn"](y=5) == 10
        assert reg["callable_obj"]() == 1  # callable instance attribute

    def test_duplicate_names_across_objects_raise(self) -> None:
        """Duplicate names across objects raise."""
        with pytest.raises(ValueError, match="multiple objects"):
            build_registry([_Native(), _Native()])

    def test_none_and_empty(self) -> None:
        """None and empty."""
        assert build_registry(None) == {}
        assert build_registry([]) == {}


class _FakeEngine(Engine):
    """Concrete Engine with a side-effecting property, no external dependency."""

    name = "fake_for_protocol_tests"
    evaluations = 0

    def __init__(self) -> None:
        super().__init__()
        self.comm = None

    @property
    def rank(self) -> int:
        type(self).evaluations += 1
        raise RuntimeError("engine property evaluated during discovery")

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def initialize_parameters(self) -> None:
        pass

    def initialize_system(
        self, types: Any, positions: Any, cell: Any, pbc: Any
    ) -> None:
        pass

    def initialize_potential(self) -> None:
        pass

    def get_positions(self) -> Any:
        return None

    def set_positions(self, positions: Any) -> None:
        pass

    def get_total_energy(self, positions: Any = None, recompute: bool = True) -> Any:
        return 0.0

    def get_potential_energy(
        self, positions: Any = None, recompute: bool = True
    ) -> Any:
        return 0.0

    def minimize(self, positions: Any = None) -> None:
        pass

    def minimize_with_results(self, positions: Any = None) -> Any:
        return None


class _SideEffectExtension(EngineExtension):
    """Extension whose property records every evaluation."""

    evaluations = 0

    def __init__(self, engine: Engine) -> None:
        super().__init__(engine)
        self.param = 2.0

    @property
    def tolerance(self) -> float:
        type(self).evaluations += 1
        return self.param * 0.01  # would also crash before param exists

    @functools.cached_property
    def cached_tolerance(self) -> float:
        # The S5 shape: a side-effecting lazy attribute (e.g. one that builds a
        # scratch engine) that must not run during discovery on any rank.
        type(self).evaluations += 1
        return self.param * 0.02

    lazy = _LazyDescriptor()

    def ext_op(self, x: int) -> int:
        return x + 1

    @classmethod
    def ext_classmethod(cls) -> str:
        return cls.__name__


class TestEngineDiscovery:
    """Engine.register/__dir__/__getattr__ never evaluate properties."""

    def setup_method(self) -> None:
        """Reset the evaluation counters before each test."""
        _FakeEngine.evaluations = 0
        _SideEffectExtension.evaluations = 0
        _LazyDescriptor.evaluations = 0

    def test_register_dir_and_registry_never_evaluate_properties(self) -> None:
        """Register dir and registry never evaluate properties."""
        engine = _FakeEngine()
        ext = _SideEffectExtension(engine)  # register() runs here
        names = dir(engine)
        reg = build_registry(engine)
        _ = inspect.getmembers(type(engine))  # class-level inspection is safe too
        assert _FakeEngine.evaluations == 0
        assert _SideEffectExtension.evaluations == 0
        assert _LazyDescriptor.evaluations == 0
        assert "ext_op" in names and "ext_classmethod" in names
        assert "tolerance" not in names
        assert "cached_tolerance" not in names and "lazy" not in names
        assert "rank" not in reg and "tolerance" not in reg
        assert "cached_tolerance" not in reg and "lazy" not in reg
        assert reg["ext_op"](x=1) == 2
        assert reg["ext_classmethod"]() == "_SideEffectExtension"
        assert reg["ext_op"].__self__ is ext
        # Lifecycle/framework methods stay ordinary registry entries.
        assert {"start", "close", "create", "register"} <= set(reg)

    def test_getattr_delegates_only_class_level_callables(self) -> None:
        """Getattr delegates only class level callables."""
        engine = _FakeEngine()
        ext = _SideEffectExtension(engine)
        assert engine.ext_op(x=2) == 3
        assert not hasattr(engine, "tolerance")
        assert not hasattr(engine, "cached_tolerance")  # hasattr must not evaluate
        assert not hasattr(engine, "lazy")
        assert not hasattr(engine, "param")
        ext.instance_callable = lambda: "not delegated"
        assert not hasattr(engine, "instance_callable")
        assert _SideEffectExtension.evaluations == 0
        assert _LazyDescriptor.evaluations == 0
        assert "cached_tolerance" not in ext.__dict__
        # The descriptors are still usable on the extension itself.
        assert ext.tolerance == pytest.approx(0.02)
        assert ext.cached_tolerance == pytest.approx(0.04)
        assert ext.lazy == 1.5
        assert _SideEffectExtension.evaluations == 2

    def test_extension_conflict_checks_preserved(self) -> None:
        """Extension conflict checks preserved."""
        engine = _FakeEngine()
        _SideEffectExtension(engine)

        class _Conflicting(EngineExtension):
            def ext_op(self) -> None:
                pass

        with pytest.raises(ValueError, match="conflicting methods"):
            _Conflicting(engine)

        class _Shadowing(EngineExtension):
            def minimize(self) -> None:
                pass

        with pytest.raises(ValueError, match="shadows native"):
            _Shadowing(engine)
        assert _SideEffectExtension.evaluations == 0

    def test_worker_registry_on_engine_before_start(self) -> None:
        """Worker registry on engine before start."""
        engine = _FakeEngine()
        _SideEffectExtension(engine)
        w = Worker(
            local_obj=engine,
            local_comm=MPI.COMM_SELF,
            worker_id=0,
            world_comm=_FakeWorldComm(),
        )
        assert _FakeEngine.evaluations == 0
        assert _SideEffectExtension.evaluations == 0
        assert _LazyDescriptor.evaluations == 0
        assert "ext_op" in w.list_ops() and "rank" not in w.list_ops()
        assert "cached_tolerance" not in w.list_ops()
        assert "lazy" not in w.list_ops()


class _Probe:
    """Class carrying one attribute of every flavour for the static rule."""

    def function(self) -> None:
        pass

    @classmethod
    def cm(cls) -> None:
        pass

    @staticmethod
    def sm() -> None:
        pass

    @property
    def prop(self) -> int:
        return 1

    @functools.cached_property
    def cached(self) -> int:
        return 1

    @functools.partialmethod
    def partial_method(self, x: int) -> int:
        return x

    @functools.singledispatchmethod
    def dispatched(self, x: Any) -> Any:
        return x

    lazy = _LazyDescriptor()
    callable_obj = _Callable()
    builtin = len
    number = 3


class TestStaticCallableRule:
    """The one static rule shared by build_registry and Engine discovery."""

    def test_engine_and_worker_share_one_helper(self) -> None:
        """Engine and worker share one helper."""
        assert engine_base.is_static_callable is is_static_callable
        assert engine_base._is_class_operation(_Probe, "function") is True
        assert engine_base._is_class_operation(_Probe, "cached") is False
        assert engine_base._is_class_operation(_Probe, "missing") is False

    @pytest.mark.parametrize(
        "name",
        ["function", "cm", "sm", "callable_obj", "builtin"],
    )
    def test_routines_are_operations(self, name: str) -> None:
        """Routines are operations."""
        raw = inspect.getattr_static(_Probe, name)
        assert is_static_callable(raw) is True
        assert engine_base._is_class_operation(_Probe, name) is True

    @pytest.mark.parametrize(
        "name",
        ["prop", "cached", "partial_method", "dispatched", "lazy", "number"],
    )
    def test_value_descriptors_are_not_operations(self, name: str) -> None:
        """Value descriptors are not operations."""
        raw = inspect.getattr_static(_Probe, name)
        assert is_static_callable(raw) is False
        assert engine_base._is_class_operation(_Probe, name) is False

    def test_isroutine_would_have_accepted_the_lazy_descriptors(self) -> None:
        """Isroutine would have accepted the lazy descriptors."""
        # Documents why an allow-list is used instead of inspect.isroutine.
        for name in ("cached", "partial_method", "dispatched", "lazy"):
            assert inspect.isroutine(inspect.getattr_static(_Probe, name))

    def test_c_level_descriptors_and_bound_methods_qualify(self) -> None:
        """C level descriptors and bound methods qualify."""
        assert is_static_callable(str.upper)  # method descriptor
        assert is_static_callable(dict.__dict__["fromkeys"])  # classmethod descr
        assert is_static_callable(object.__init__)  # wrapper descriptor
        assert is_static_callable(object().__str__)  # method-wrapper
        assert is_static_callable([].append)  # builtin bound method
        assert is_static_callable(_Probe().function)  # bound method


class TestLammpsEngineRegistryBeforeStart:
    """A LammpsEngine registry is built before start without touching lmp."""

    def test_registry_built_before_start_without_touching_lmp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Registry built before start without touching lmp."""
        pytest.importorskip("lammps")
        from dataclasses import dataclass

        from pykmc.engine.lammps import LammpsEngine

        @dataclass
        class _Cfg:
            pair_style: str = "lj/cut 6.0"
            pair_coeff: str = "* * 0.52 2.274"
            min_style: str = "cg"
            minimize: str = "1e-6 1e-8 1000 10000"
            frz_min: str = "1e-4 1e-6 100 1000"
            verbosity: int = 0

        evaluated: list[str] = []

        def _rank(self: LammpsEngine) -> int:
            evaluated.append("rank")
            return 0

        monkeypatch.setattr(LammpsEngine, "rank", property(_rank))
        engine = Engine.create("lammps", config=_Cfg(), comm=None)
        _SideEffectExtension(engine)
        assert engine.lmp is None
        reg = build_registry(engine)
        names = dir(engine)
        assert evaluated == []
        assert _SideEffectExtension.evaluations == 0
        assert engine.lmp is None  # never started
        assert "rank" not in reg and "tolerance" not in reg
        assert "tolerance" not in names
        assert {"minimize", "command", "start", "close", "ext_op"} <= set(reg)
