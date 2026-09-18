from .session import Session
from .worker import RESERVED_OPS
from concurrent.futures import Future
from dataclasses import dataclass, field
import logging
import queue
import threading
from typing import Any, Literal

logger = logging.getLogger(__name__)

# RESERVED_OPS (defined next to Worker's builtins so it cannot drift from them)
# holds the control builtins that must never travel through a reply-expecting
# Session.call: `shutdown` stops the worker loop and the `use_*` switches desync
# the worker from Manager.mode, and all of them return SILENT so the caller
# would hang. `list_ops` is deliberately absent: it is read-only and returns a
# value, so it stays reachable through Manager.list_ops and the group_/global_
# wrappers. `start`, `close` and `initialize_*` are ordinary engine operations.


def _reject_reserved(op_name: str) -> None:
    """Raise ValueError if ``op_name`` is a Worker control builtin.

    Parameters
    ----------
    op_name : str
        Operation name about to be submitted.

    Raises
    ------
    ValueError
        If the name is one of ``RESERVED_OPS``.

    """
    if op_name in RESERVED_OPS:
        raise ValueError(
            f"'{op_name}' is a Worker control builtin and cannot be submitted. Mode "
            "switching is handled automatically by submit/submit_group/"
            "submit_global; use Manager.shutdown() to stop the pool."
        )


@dataclass
class Job:
    """Unit of work dispatched to a thread worker.

    Parameters
    ----------
    op_name : str
        Operation name in the Worker registry.
    kwargs : dict
        Keyword arguments forwarded to the operation.
    future : Future
        Resolved with the result once the job completes.

    """

    op_name: str
    kwargs: dict = field(default_factory=dict)
    future: Future = field(default_factory=Future)


class Manager:
    def __init__(
        self,
        local_sessions: list[Session],
        global_session: Session | None = None,
        group_session: Session | None = None,
    ) -> None:
        self.local_sessions = local_sessions
        self.global_session = global_session
        self.group_session = group_session

        self._local_queue: queue.Queue[Job] = queue.Queue()
        self._local_threads: list[threading.Thread] = []
        self._closed = False
        # Sessions still owed the shutdown control message (None until the
        # first shutdown() call); lets a retry resume after a failed send.
        self._shutdown_pending: list[Session] | None = None

        self.mode: Literal["local", "group", "global"] = "local"

    def _ensure_open(self) -> None:
        """Raise RuntimeError if shutdown() has been called (even partially)."""
        if self._closed or self._shutdown_pending is not None:
            raise RuntimeError(
                "Manager has been shut down; its workers have left their message "
                "loop and cannot accept further operations."
            )

    def start(self) -> None:
        """Start one thread per local session. Must be called before submit().

        Raises
        ------
        RuntimeError
            If start() has already been called, or if the manager was shut down.

        """
        self._ensure_open()
        if self._local_threads:
            raise RuntimeError("Manager is already started.")
        for session in self.local_sessions:
            t = threading.Thread(
                target=self._worker_loop, args=(session, self._local_queue), daemon=True
            )
            t.start()
            self._local_threads.append(t)

    def shutdown(self) -> None:
        """Stop all thread workers and shut down sessions.

        Idempotent: once every session has been notified a second call is a
        no-op. Re-sending the shutdown command would leave unconsumed messages
        addressed to workers that have already left their loop. Jobs still
        queued are dispatched before the threads stop; afterwards every submit
        path raises ``RuntimeError``.

        The manager is marked closed only after the last session has been
        notified. If a session's ``shutdown()`` send raises, the exception
        propagates, the sessions not yet notified (the failing one included)
        stay pending, and a later ``shutdown()`` call resumes with them instead
        of silently returning.
        """
        if self._closed:
            return
        if self.mode != "local":
            self._use_local()
        for _ in self._local_threads:
            self._local_queue.put(None)
        for t in self._local_threads:
            t.join()
        self._local_threads.clear()
        if self._shutdown_pending is None:
            self._shutdown_pending = list(self.local_sessions)
        while self._shutdown_pending:
            self._shutdown_pending[0].shutdown()  # a raise keeps it pending
            self._shutdown_pending.pop(0)
        self._closed = True

    def list_ops(self, mode: str = "local") -> list[str]:
        """Return the list of available operations for the given mode."""
        return self.submit("list_ops", mode=mode).result()

    def broadcast(self, op_name: str, **kwargs) -> None:
        """Send the same op to all local sessions sequentially.

        Useful for initialisation steps that every worker must run.
        Switches to local mode automatically if needed.

        Raises
        ------
        ValueError
            If ``op_name`` is a reserved Worker control builtin.
        RuntimeError
            If the manager was shut down.

        """
        _reject_reserved(op_name)
        self._ensure_open()
        if self.mode != "local":
            self._use_local()
        self._local_queue.join()
        for session in self.local_sessions:
            session.call(op_name, **kwargs)

    # ------------------------------------------------------------------
    # Mode transitions
    # ------------------------------------------------------------------

    def _use_local(self) -> None:
        """Switch all workers back to local mode.

        Only one message is sent (to the global or group session master rank).
        All workers in that communicator receive it via the bcast in Worker._loop,
        so a single send is enough to switch the entire collective back to local.
        """
        if self.mode == "global":
            self.global_session.use_local()
        elif self.mode == "group":
            self.group_session.use_local()
        self.mode = "local"

    def _use_global(self) -> None:
        """Drain local queue then switch all workers to global mode."""
        if self.mode != "local":
            self._use_local()
        self._local_queue.join()
        for session in self.local_sessions:
            session.use_global()
        self.mode = "global"

    def _use_group(self) -> None:
        """Drain local queue then switch workers to group mode.

        Workers without a group_comm silently stay in local mode and become
        idle for the duration of the group operation.
        """
        if self.mode != "local":
            self._use_local()
        self._local_queue.join()
        for session in self.local_sessions:
            session.use_group()
        self.mode = "group"

    # ------------------------------------------------------------------
    # Job submission
    # ------------------------------------------------------------------

    def _worker_loop(self, session: Session, job_queue: queue.Queue) -> None:
        """Pull jobs from the queue and execute via the session.

        Every dequeued item, sentinel included, is balanced by ``task_done`` so
        ``queue.join()`` stays meaningful. A job whose Future was cancelled
        before dispatch is skipped. The thread never dies on an exception: a
        failure is attached to the job's Future, and anything unexpected (even
        a ``BaseException``) is logged and the loop continues.
        """
        while True:
            job = job_queue.get()
            if job is None:  # sentinel — stop
                job_queue.task_done()
                break
            try:
                if not job.future.set_running_or_notify_cancel():
                    continue  # cancelled before dispatch — skip
                try:
                    result = session.call(job.op_name, **job.kwargs)
                except Exception as e:
                    job.future.set_exception(e)
                else:
                    job.future.set_result(result)
            except BaseException as e:
                # threading swallows SystemExit and friends silently; log and keep
                # the pool thread alive, the caller learns through its Future.
                logger.exception(
                    "Worker thread for session %s: unexpected error handling '%s'",
                    getattr(session, "session_id", "?"),
                    job.op_name,
                )
                if not job.future.done():
                    job.future.set_exception(e)
            finally:
                job_queue.task_done()

    def submit(self, op_name: str, **kwargs) -> Future:
        """Submit a job to the local worker pool (async).

        If the manager is currently in group or global mode, switches back to
        local mode first. Similarly, submit_group() and submit_global() switch
        to their respective modes automatically before dispatching.

        Returns
        -------
        Future
            Resolved when the job completes.

        Raises
        ------
        ValueError
            If ``op_name`` is a reserved Worker control builtin.
        RuntimeError
            If the manager was shut down, or start() has not been called.

        """
        _reject_reserved(op_name)
        self._ensure_open()
        if not self._local_threads:
            raise RuntimeError("Manager has not been started. Call start() first.")
        if self.mode != "local":
            self._use_local()
        job = Job(op_name=op_name, kwargs=kwargs)
        self._local_queue.put(job)
        return job.future

    def submit_group(self, op_name: str, **kwargs) -> Any:
        """Submit a job to the group worker and block until it completes.

        The group spans only the subset of workers configured at factory time.
        All other workers are idle for the duration of the call.

        Returns
        -------
        Any
            Result of the operation, or None for void operations.

        Raises
        ------
        ValueError
            If ``op_name`` is a reserved Worker control builtin.
        RuntimeError
            If no group session was configured, or the manager was shut down.

        """
        _reject_reserved(op_name)
        self._ensure_open()
        if self.group_session is None:
            raise RuntimeError("No group session configured.")
        if self.mode != "group":
            self._use_group()
        return self.group_session.call(op_name, **kwargs)

    def submit_global(self, op_name: str, **kwargs) -> Any:
        """Submit a job to the global worker and block until it completes.

        All MPI ranks work collectively, so the call is synchronous.

        Returns
        -------
        Any
            Result of the operation, or None for void operations.

        Raises
        ------
        ValueError
            If ``op_name`` is a reserved Worker control builtin.
        RuntimeError
            If no global session was configured, or the manager was shut down.

        """
        _reject_reserved(op_name)
        self._ensure_open()
        if self.global_session is None:
            raise RuntimeError("No global session configured.")
        if self.mode != "global":
            self._use_global()
        return self.global_session.call(op_name, **kwargs)

    def __getattr__(self, name: str):
        """Auto-generate submit wrappers from attribute access.

        mgr.minimize(positions=pos)        → submit("minimize", positions=pos)        → Future
        mgr.group_minimize(positions=pos)  → submit_group("minimize", positions=pos)  → result
        mgr.global_minimize(positions=pos) → submit_global("minimize", positions=pos) → result

        Underscore-prefixed names (including dunders probed by copy/pickle)
        raise AttributeError instead of being forwarded. The wrappers apply the
        same reserved-name rejection as the submit methods: calling
        ``mgr.use_local()`` or ``mgr.group_shutdown()`` raises ValueError.
        """
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        if name.startswith("global_"):
            op = name[len("global_") :]
            return lambda **kw: self.submit_global(op, **kw)
        if name.startswith("group_"):
            op = name[len("group_") :]
            return lambda **kw: self.submit_group(op, **kw)
        return lambda **kw: self.submit(name, **kw)
