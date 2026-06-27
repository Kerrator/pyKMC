"""Tests for the opt-in engine-op wall guard (recycle pool-hang follow-up).

Covers the pure deadline-poll helper that backs rank 0's bounded wait for an engine
reply. The MPI wiring (irecv polling + MPI.COMM_WORLD.Abort on timeout) is exercised
on the cluster; here we pin the timing/contract logic deterministically.
"""
import pytest

from pykmc.enginemanager.lmpi.sessions.mpi_api_sessions import (
    EngineOpTimeout,
    _await_reply,
)


def test_await_reply_returns_value_once_poll_reports_done():
    """Returns the delivered value as soon as poll() reports done, without waiting."""
    state = {"n": 0}

    def poll():
        state["n"] += 1
        done = state["n"] >= 3
        return (done, "REPLY" if done else None)

    clock = {"t": 0.0}
    result = _await_reply(
        poll, timeout_s=100.0, monotonic=lambda: clock["t"],
        sleep=lambda _s: clock.__setitem__("t", clock["t"] + 0.1),
    )
    assert result == "REPLY"


def test_await_reply_raises_engine_op_timeout_past_the_deadline():
    """If poll() never reports done, it raises EngineOpTimeout after the deadline."""
    clock = {"t": 0.0}
    with pytest.raises(EngineOpTimeout):
        _await_reply(
            lambda: (False, None), timeout_s=2.0, monotonic=lambda: clock["t"],
            sleep=lambda _s: clock.__setitem__("t", clock["t"] + 1.0),
        )
