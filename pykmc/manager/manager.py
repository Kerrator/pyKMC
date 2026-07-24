from .session import Session
from concurrent.futures import Future
from dataclasses import dataclass, field
import queue
import threading
from typing import Any, Literal


# Worker builtins that must never be reached through Session.call: shutdown stops
# the worker loop and the use_* switches desync it from Manager.mode (in group or
# global mode only one worker is rank 0, so the others never reply at all).
# `list_ops` is deliberately absent: it is read-only, returns a value, and is the
# supported way to introspect a mode's registry.
_BLOCKED_OPS = frozenset({"use_local", "use_group", "use_global", "shutdown"})


def _reject_blocked(op_name: str) -> None:
    """Raise if an operation must not be routed through a Session."""
    if op_name in _BLOCKED_OPS:
        raise ValueError(
            f"'{op_name}' is a Worker builtin and cannot be submitted. Mode "
            f"switching is handled automatically by submit/submit_group/"
            f"submit_global; use Manager.shutdown() to stop the pool."
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
    args : tuple
        Positional arguments forwarded to the operation, ahead of the kwargs.
    future : Future
        Resolved with the result once the job completes.
    """

    op_name: str
    kwargs: dict = field(default_factory=dict)
    args: tuple = ()
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

        self.mode: Literal["local", "group", "global"] = "local"

    def start(self) -> None:
        """Start one thread per local session. Must be called before submit().

        Raises
        ------
        RuntimeError
            If start() has already been called.
        """
        if self._closed:
            raise RuntimeError(
                "Manager has been shut down; its workers have left their message "
                "loop and cannot be restarted."
            )
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

        Idempotent: a second call is a no-op. Re-sending the shutdown command
        would leave unconsumed messages addressed to workers that have already
        left their loop, which the next Manager on the same communicator would
        read as its first command.
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
        self._closed = True
        for session in self.local_sessions:
            session.shutdown()

    def list_ops(self, mode: str = "local") -> list[str]:
        """Return the list of available operations for the given mode."""
        return self.submit("list_ops", mode=mode).result()

    def broadcast(self, op_name: str, *args: Any, **kwargs) -> None:
        """Send the same op to all local sessions sequentially.

        Useful for initialisation steps that every worker must run.
        Switches to local mode automatically if needed.
        """
        _reject_blocked(op_name)
        if self.mode != "local":
            self._use_local()
        self._local_queue.join()
        for session in self.local_sessions:
            session.call(op_name, *args, **kwargs)

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
        """Pull jobs from the queue and execute via the session."""
        while True:
            job = job_queue.get()
            if job is None:  # sentinel — stop
                job_queue.task_done()
                break
            if not job.future.set_running_or_notify_cancel():  # cancelled — skip
                job_queue.task_done()
                continue
            try:
                result = session.call(job.op_name, *job.args, **job.kwargs)
                job.future.set_result(result)
            except Exception as e:
                job.future.set_exception(e)
            finally:
                job_queue.task_done()

    def submit(self, op_name: str, *args: Any, **kwargs) -> Future:
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
        RuntimeError
            If start() has not been called.
        """
        _reject_blocked(op_name)
        if not self._local_threads:
            raise RuntimeError("Manager has not been started. Call start() first.")
        if self.mode != "local":
            self._use_local()
        job = Job(op_name=op_name, args=args, kwargs=kwargs)
        self._local_queue.put(job)
        return job.future

    def submit_group(self, op_name: str, *args: Any, **kwargs) -> Any:
        """Submit a job to the group worker and block until it completes.

        The group spans only the subset of workers configured at factory time.
        All other workers are idle for the duration of the call.

        Returns
        -------
        Any
            Result of the operation, or None for void operations.

        Raises
        ------
        RuntimeError
            If no group session was configured.
        """
        _reject_blocked(op_name)
        if self.group_session is None:
            raise RuntimeError("No group session configured.")
        if self.mode != "group":
            self._use_group()
        return self.group_session.call(op_name, *args, **kwargs)

    def submit_global(self, op_name: str, *args: Any, **kwargs) -> Any:
        """Submit a job to the global worker and block until it completes.

        All MPI ranks work collectively, so the call is synchronous.

        Returns
        -------
        Any
            Result of the operation, or None for void operations.

        Raises
        ------
        RuntimeError
            If no global session was configured.
        """
        _reject_blocked(op_name)
        if self.global_session is None:
            raise RuntimeError("No global session configured.")
        if self.mode != "global":
            self._use_global()
        return self.global_session.call(op_name, *args, **kwargs)

    def __getattr__(self, name: str):
        """Auto-generate submit wrappers from attribute access.

        mgr.minimize(positions=pos)        → submit("minimize", positions=pos)        → Future
        mgr.group_minimize(positions=pos)  → submit_group("minimize", positions=pos)  → result
        mgr.global_minimize(positions=pos) → submit_global("minimize", positions=pos) → result

        Underscore-prefixed names (including dunders probed by copy/pickle) and
        the Worker builtins (use_local/use_group/use_global/shutdown) raise
        AttributeError instead of being forwarded — mode switching is handled
        automatically by the submit methods. Manager's own attributes always win
        over registry ops, and the group_/global_ prefixes are stripped before
        lookup; reach a shadowed op, or an op whose real name starts with
        group_/global_, via submit(op_name, ...) explicitly — which applies the
        same builtin rejection.
        """
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        if name.startswith("global_"):
            op, submit = name[len("global_") :], self.submit_global
        elif name.startswith("group_"):
            op, submit = name[len("group_") :], self.submit_group
        else:
            op, submit = name, self.submit
        # Checked after prefix stripping: group_shutdown/global_use_local must be
        # rejected too, since reaching a builtin over the reply-expecting path
        # kills the worker loop (shutdown) or desyncs its mode (use_*).
        if op in _BLOCKED_OPS:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}': "
                f"'{op}' is a Worker builtin and is not callable through the "
                f"submit wrappers."
            )
        return lambda *a, **kw: submit(op, *a, **kw)
