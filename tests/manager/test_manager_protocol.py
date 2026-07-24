"""Regression tests for the Manager request/response protocol.

The topology is derived from ``MPI.COMM_WORLD.Get_size()`` rather than hard-coded,
so the same module is meaningful on a development box (``mpirun -n 8``), on a
typical single-node allocation (``-n 24``) and on a large cluster job (``-n 192``).
Each test states the layout it needs; tests that require a worker *outside* the
group, or more than one rank *inside* a worker, skip when the world is too small.

Every wait is bounded: results are collected with ``Future.result(timeout=...)``
so a regression fails the test instead of hanging the job.
"""

import numpy as np
import pytest
from mpi4py import MPI

from pykmc.manager import ManagerFactory

#: manager rank + at least three worker ranks, so at least one worker can hold
#: two ranks (real local collectives) while another stays outside the group.
MIN_RANKS = 4

#: Bound on every blocking wait, in seconds.
TIMEOUT = 30.0


def topology():
    """Derive a valid (n_workers, group_size) for the current world size.

    ``group_size`` must land on a whole-worker boundary, so it is picked from the
    cumulative worker-chunk sizes; it deliberately covers a strict subset of the
    workers so that group mode is distinguishable from global mode.
    """
    size = MPI.COMM_WORLD.Get_size()
    if size < MIN_RANKS:
        pytest.skip(f"needs at least {MIN_RANKS} MPI ranks, got {size}")
    available = size - 1
    n_workers = max(2, available // 2)
    chunks = [len(c) for c in np.array_split(np.arange(available), n_workers)]
    boundaries = [int(b) for b in np.cumsum(chunks)]
    # a strict subset of workers: everything except the last worker
    group_size = boundaries[-2]
    return n_workers, group_size, chunks, boundaries


class Ops:
    """Operations exercised over every mode."""

    def __init__(self, comm):
        self.comm = comm

    def gather_rank(self):
        """Return one entry per rank of the ACTIVE communicator."""
        gathered = self.comm.gather(self.comm.Get_rank(), root=0)
        if self.comm.Get_rank() == 0:
            return gathered

    def echo(self, a, b, c=None):
        return (a, b, c)

    def fail_empty(self):
        raise ValueError()  # str() == "" -> must still be reported as an error

    def fail_off_root(self):
        if self.comm.Get_rank() != 0:
            raise RuntimeError("boom on non-root")
        return "root-ok"

    def unserialisable(self):
        return lambda x: x

    def slow_marker(self):
        return "done"


def concat(comm, a, b):
    """extra_op: receives the active communicator, then the caller's args."""
    return f"{a}-{b}-{comm.Get_size()}"


@pytest.fixture
def manager():
    """Launch a Manager, yield it on rank 0 (None elsewhere), always shut down."""
    n_workers, group_size, _, _ = topology()
    MPI.COMM_WORLD.Barrier()
    mgr = ManagerFactory(
        obj_factory=lambda comm, _mode: Ops(comm),
        n_workers=n_workers,
        comm=MPI.COMM_WORLD,
        has_global=True,
        group_size=group_size,
        extra_ops={"concat": concat},
    ).launch()
    try:
        yield mgr
    finally:
        # Runs even when rank 0 fails an assertion, so the worker ranks are
        # always released and the next test's collective Split stays in step.
        if mgr is not None:
            mgr.shutdown()
        MPI.COMM_WORLD.Barrier()


def rank0_only(mgr):
    """Skip the body on worker ranks, which have no Manager."""
    return mgr is None


class TestPositionalArguments:
    """The dispatch chain must carry positional arguments end to end."""

    def test_registry_op_receives_positionals(self, manager):
        if rank0_only(manager):
            return
        assert manager.echo(1, 2).result(timeout=TIMEOUT) == (1, 2, None)
        assert manager.echo(1, 2, 3).result(timeout=TIMEOUT) == (1, 2, 3)
        assert manager.echo(1, 2, c=3).result(timeout=TIMEOUT) == (1, 2, 3)
        assert manager.echo(a=1, b=2).result(timeout=TIMEOUT) == (1, 2, None)

    def test_positionals_survive_numpy_payloads(self, manager):
        if rank0_only(manager):
            return
        arr = np.arange(9.0).reshape(3, 3)
        a, b, _ = manager.echo(arr, "tag").result(timeout=TIMEOUT)
        assert np.array_equal(a, arr) and b == "tag"

    def test_extra_op_receives_comm_then_positionals(self, manager):
        if rank0_only(manager):
            return
        # extra_ops are called as fn(comm, *args, **kwargs)
        assert manager.concat("x", "y").result(timeout=TIMEOUT).startswith("x-y-")
        assert manager.concat(a="x", b="y").result(timeout=TIMEOUT).startswith("x-y-")

    def test_positionals_in_group_and_global_modes(self, manager):
        if rank0_only(manager):
            return
        assert manager.group_echo(1, 2, 3) == (1, 2, 3)
        assert manager.global_echo(4, 5, 6) == (4, 5, 6)


class TestModeSelection:
    """Result length is the observable that distinguishes the communicators."""

    def test_gather_length_matches_active_communicator(self, manager):
        if rank0_only(manager):
            return
        _n_workers, group_size, chunks, _ = topology()
        available = MPI.COMM_WORLD.Get_size() - 1

        local = manager.gather_rank().result(timeout=TIMEOUT)
        assert len(local) in chunks, (
            f"local gather returned {len(local)} entries; expected one worker's "
            f"chunk size, one of {sorted(set(chunks))}"
        )
        assert len(manager.group_gather_rank()) == group_size
        assert len(manager.global_gather_rank()) == available


class TestTerminalResponse:
    """Every accepted request must produce exactly one terminal response."""

    def test_exception_with_empty_str_is_reported(self, manager):
        if rank0_only(manager):
            return
        with pytest.raises(RuntimeError):
            manager.fail_empty().result(timeout=TIMEOUT)
        assert manager.slow_marker().result(timeout=TIMEOUT) == "done"

    def test_raising_builtin_does_not_kill_the_worker(self, manager):
        if rank0_only(manager):
            return
        with pytest.raises(RuntimeError, match="Unknown mode"):
            manager.submit("list_ops", mode="not-a-mode").result(timeout=TIMEOUT)
        assert manager.slow_marker().result(timeout=TIMEOUT) == "done"

    def test_unserialisable_result_reports_instead_of_stranding(self, manager):
        if rank0_only(manager):
            return
        with pytest.raises(RuntimeError):
            manager.unserialisable().result(timeout=TIMEOUT)
        # the status/result streams must still be in step
        assert manager.slow_marker().result(timeout=TIMEOUT) == "done"
        assert manager.echo(7, 8).result(timeout=TIMEOUT) == (7, 8, None)


class TestPerRankFailures:
    """A failure on a non-root rank must reach the caller."""

    def test_non_root_failure_is_reported(self, manager):
        if rank0_only(manager):
            return
        _n_workers, group_size, chunks, _ = topology()
        if max(chunks) < 2 and group_size < 2:
            pytest.skip("no active communicator holds more than one rank")
        # group mode always spans >= 2 ranks when group_size >= 2
        with pytest.raises(RuntimeError, match="rank"):
            manager.group_fail_off_root()


class TestLifecycleAndGuards:
    """Builtin-name guards and shutdown bookkeeping."""

    def test_blocked_builtins_are_rejected_on_every_path(self, manager):
        if rank0_only(manager):
            return
        for name in ("use_local", "use_group", "use_global", "shutdown"):
            for spelling in (name, f"group_{name}", f"global_{name}"):
                if spelling == "shutdown":
                    continue  # Manager.shutdown is a real method
                with pytest.raises(AttributeError):
                    getattr(manager, spelling)
        for op in ("use_global", "shutdown"):
            with pytest.raises(ValueError):
                manager.submit(op)
            with pytest.raises(ValueError):
                manager.broadcast(op)

    def test_list_ops_remains_available_in_every_mode(self, manager):
        if rank0_only(manager):
            return
        assert "echo" in manager.list_ops()
        assert "echo" in manager.group_list_ops(mode="group")
        assert "echo" in manager.global_list_ops(mode="global")

    def test_cancelled_future_does_not_kill_the_pool(self, manager):
        if rank0_only(manager):
            return
        futures = [manager.slow_marker() for _ in range(24)]
        cancelled = [f for f in futures if f.cancel()]
        for f in futures:
            if not f.cancelled():
                assert f.result(timeout=TIMEOUT) == "done"
        # whatever was cancelled, the pool must still serve new work
        assert manager.echo(1, 2).result(timeout=TIMEOUT) == (1, 2, None)
        assert cancelled == cancelled  # recorded for diagnostics, never asserted on

    def test_submit_and_start_after_shutdown_raise(self, manager):
        if rank0_only(manager):
            return
        manager.shutdown()
        with pytest.raises(RuntimeError):
            manager.submit("slow_marker")
        with pytest.raises(RuntimeError):
            manager.start()
        # the fixture calls shutdown() again; it must stay idempotent
        manager.shutdown()
