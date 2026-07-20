from .session import Session
from concurrent.futures import Future
from dataclasses import dataclass, field
import queue
import threading
from typing import Any

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

    def __init__(self, local_sessions: list[Session], global_session: Session|None = None) -> None:

        self.local_sessions = local_sessions
        self.global_session = global_session

        self._local_queue: queue.Queue[Job] = queue.Queue()
        self._local_threads = []

        self.using_global = False

    def start(self) -> None:
        for session in self.local_sessions:
            t = threading.Thread(target=self._worker_loop, args=(session, self._local_queue), daemon=True)
            t.start()
            self._local_threads.append(t)

    def close(self) -> None:
        """Stop all thread workers and close sessions."""
        for _ in self._local_threads:
            self._local_queue.put(None)
        for t in self._local_threads:
            t.join()

        for session in self.local_sessions:
            session.close()

        if self.global_session is not None:
            self.global_session.close()

    def broadcast(self, op_name: str, **kwargs) -> None:
        """Send the same op to all local sessions sequentially.

        Useful when initializing and mode switching."""
        for session in self.local_sessions:
            session.call(op_name, **kwargs)

    def _use_local(self) -> None:
        """Switch all workers to local mode."""
        self.global_session.use_local()
        self.using_global = False

    def _use_global(self) -> None:
        """Wait for local queue to drain, then switch all workers to global."""
        self._local_queue.join()
        for session in self.local_sessions:
            session.use_global()
        self.using_global = True

    def _worker_loop(self, session: Session, job_queue: queue.Queue) -> None:
        """Pull jobs from the queue and execute via the session."""
        while True:
            job = job_queue.get()
            if job is None:  # sentinel — stop
                break
            try:
                result = session.call(job.op_name, **job.kwargs)
                job.future.set_result(result)
            except Exception as e:
                job.future.set_exception(e)
            finally:
                job_queue.task_done()

    def submit(self, op_name: str, **kwargs) -> Future:
        """Submit a job to the local worker pool.

        Parameters
        ----------
        op_name : str
            Operation name in the Worker registry.
        **kwargs
            Forwarded to the operation.

        Returns
        -------
        Future
            Resolved when the job completes.
        """
        if self.using_global:
            self._use_local()
        job = Job(op_name=op_name, kwargs=kwargs)
        self._local_queue.put(job)
        return job.future

    def submit_global(self, op_name: str, **kwargs) -> Any:
        """Submit a job to the global worker and block until it completes.

        Unlike local submissions, global operations are synchronous — all MPI
        ranks work collectively on a single job, so there is nothing to
        parallelise while waiting. The result (or exception) is returned
        directly.

        Parameters
        ----------
        op_name : str
            Operation name in the Worker registry.
        **kwargs
            Forwarded to the operation.

        Returns
        -------
        Any
            Result of the operation, or None for void operations.

        Raises
        ------
        RuntimeError
            If no global session was configured.
        """
        if self.global_session is None:
            raise RuntimeError("No global session configured.")
        if not self.using_global:
            self._use_global()
        return self.global_session.call(op_name, **kwargs)

    def __getattr__(self, name: str):
        """Auto-generate submit/submit_global wrappers.

        mgr.minimize(positions=pos)        → submit("minimize", positions=pos)  → Future
        mgr.global_minimize(positions=pos) → submit_global("minimize", positions=pos)  → result
        """
        if name.startswith("global_"):
            op = name[len("global_"):]
            return lambda **kw: self.submit_global(op, **kw)
        return lambda **kw: self.submit(name, **kw)
