"""Pure-python tests for the session reply parsing used by the basin operations.

``MpiApiSession._receive_result_or_error`` is exercised with a fake messenger, so no
MPI launch is needed. These tests pin the contract: a ``result`` message returns its
value, an ``error`` message raises a descriptive RuntimeError, and anything else
raises immediately (a session must never silently swallow an engine reply).
"""

import pytest

from pykmc.enginemanager.lmpi.sessions.mpi_api_sessions import MpiApiSession


class _FakeMessenger:
    """Minimal messenger returning a queued sequence of recv() messages."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.sent = []

    def recv(self, source=None, tag=None):
        return self._replies.pop(0)

    def send(self, msg, dest=None, tag=None):
        self.sent.append(msg)


def _make_session(replies):
    """Build a session without running __init__ (no MPI), wiring only what we test."""
    session = MpiApiSession.__new__(MpiApiSession)
    session.messenger = _FakeMessenger(replies)
    session.engine_master_rank = 1
    session._is_busy = False
    return session


class TestReceiveResultOrError:

    def test_result_message_returns_value(self):
        session = _make_session([{"type": "result", "value": {"ok": True, "data": 42}}])
        assert session._receive_result_or_error() == {"ok": True, "data": 42}

    def test_error_message_raises_with_details(self):
        session = _make_session([
            {"type": "error", "value": {"operation": "basin_reconstruct",
                                        "error_type": "ValueError",
                                        "message": "bad geometry"}}
        ])
        with pytest.raises(RuntimeError, match="basin_reconstruct.*ValueError.*bad geometry"):
            session._receive_result_or_error()

    def test_unexpected_message_raises(self):
        session = _make_session([{"type": "status", "value": {"alive": True}}])
        with pytest.raises(RuntimeError, match="Unexpected message type"):
            session._receive_result_or_error()

    def test_basin_methods_send_and_clear_busy(self):
        """basin_reconstruct/basin_explore send the right message type and reset _is_busy.

        send_message() performs a status handshake before the reply, so the fake
        messenger queues a status message first, then the result.
        """
        for method_name in ("basin_reconstruct", "basin_explore"):
            session = _make_session([
                {"type": "status", "value": {"alive": True, "busy": True}},
                {"type": "result", "value": []},
            ])
            result = getattr(session, method_name)(some_kwarg=1)
            assert result == []
            assert session._is_busy is False
            assert session.messenger.sent[0]["type"] == method_name
            assert session.messenger.sent[0]["value"] == {"some_kwarg": 1}
