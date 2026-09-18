"""Real-MPI protocol tests: run with ``mpirun -n 8``.

Eight ranks with ``n_workers=3`` give chunks of 3/2/2 ranks, so every worker
owns a local rank 1 and non-root failure paths are exercised on every worker.
Under plain ``pytest`` (one rank) every test skips.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pytest
from mpi4py import MPI

from pykmc.engine import EngineExtension
from pykmc.manager import ManagerFactory


def _require_8_ranks() -> None:
    if MPI.COMM_WORLD.Get_size() != 8:
        pytest.skip("Manager protocol tests must be run with mpirun -n 8.")


def _is_root() -> bool:
    return MPI.COMM_WORLD.Get_rank() == 0


class Operations:
    """Test operations covering every reply path of the worker protocol."""

    def __init__(self, comm: MPI.Comm) -> None:
        self.comm = comm

    def echo(self, x: Any) -> Any:
        """Return ``x`` on every rank."""
        return x

    def void(self) -> None:
        """Return None on every rank."""
        return None

    def unpicklable(self) -> Any:
        """Return a value that cannot be pickled."""
        return lambda: 0  # a lambda cannot be pickled

    def root_only(self, x: Any) -> Any:
        """Return ``x`` on rank 0 and None elsewhere."""
        # The future HTST extension op returns None off-root: a normal success.
        return x if self.comm.Get_rank() == 0 else None

    def fail_on_rank(self, rank: int) -> str:
        """Raise on the given rank of the active communicator only."""
        if self.comm.Get_rank() == rank:
            raise ValueError(f"boom on rank {rank}")
        return "fine"

    def fail_everywhere(self) -> None:
        """Raise on every rank."""
        raise ValueError("boom everywhere")

    def sleep(self, seconds: float) -> float:
        """Sleep so that a pool thread stays busy."""
        time.sleep(seconds)
        return seconds


class TestManagerProtocolMPI:
    """Reply, failure, cancellation and lifecycle paths over real MPI."""

    @pytest.fixture(autouse=True)
    def setup(self) -> Any:
        """Launch a 3-worker manager (chunks 3/2/2, group of 5) on every rank."""
        _require_8_ranks()
        MPI.COMM_WORLD.Barrier()
        self.factory = ManagerFactory(
            obj_factory=lambda comm, _: Operations(comm),
            n_workers=3,
            comm=MPI.COMM_WORLD,
            has_global=True,
            group_size=5,  # workers 0 and 1 (3 + 2 ranks)
        )
        self.manager = self.factory.launch()
        yield
        if self.manager is not None:
            self.manager.shutdown()
        MPI.COMM_WORLD.Barrier()

    def test_every_worker_owns_a_local_rank_1(self) -> None:
        """Every worker owns a local rank 1."""
        assert [len(c) for c in self.factory.chunks] == [3, 2, 2]

    def test_void_result_then_valid_request(self) -> None:
        """Void result then valid request."""
        if not _is_root():
            return
        m = self.manager
        assert m.void().result() is None
        assert m.echo(x=1).result() == 1
        assert m.group_void() is None
        assert m.group_echo(x=2) == 2
        assert m.global_void() is None
        assert m.global_echo(x=3) == 3

    def test_unpicklable_result_then_valid_request(self) -> None:
        """Unpicklable result then valid request."""
        if not _is_root():
            return
        m = self.manager
        with pytest.raises(RuntimeError, match="unpicklable.*could not be pickled"):
            m.unpicklable().result()
        assert m.echo(x=1).result() == 1
        with pytest.raises(RuntimeError, match="could not be pickled"):
            m.group_unpicklable()
        assert m.group_echo(x=2) == 2
        with pytest.raises(RuntimeError, match="could not be pickled"):
            m.global_unpicklable()
        assert m.global_echo(x=3) == 3

    def test_none_off_root_is_a_normal_success(self) -> None:
        """None off root is a normal success."""
        if not _is_root():
            return
        m = self.manager
        assert m.root_only(x=4).result() == 4
        assert m.group_root_only(x=5) == 5
        assert m.global_root_only(x=6) == 6

    def test_failure_on_local_rank_1_surfaces_and_worker_stays_usable(self) -> None:
        """Failure on local rank 1 surfaces and worker stays usable."""
        if not _is_root():
            return
        m = self.manager
        pattern = r"fail_on_rank.*rank\(s\) 1 .*rank 1: ValueError: boom on rank 1"
        for _ in range(3):  # hit every worker at least once, then reuse
            with pytest.raises(RuntimeError, match=pattern):
                m.fail_on_rank(rank=1).result()
            assert m.echo(x="again").result() == "again"
        with pytest.raises(RuntimeError, match=pattern):
            m.group_fail_on_rank(rank=1)
        assert m.group_echo(x=2) == 2
        with pytest.raises(RuntimeError, match=pattern):
            m.global_fail_on_rank(rank=1)
        assert m.global_echo(x=3) == 3
        # Failure on the last rank of the global communicator (rank 6).
        with pytest.raises(RuntimeError, match=r"rank\(s\) 6 .*boom on rank 6"):
            m.global_fail_on_rank(rank=6)
        assert m.global_echo(x=7) == 7

    def test_failure_on_all_ranks(self) -> None:
        """Failure on all ranks."""
        if not _is_root():
            return
        m = self.manager
        with pytest.raises(RuntimeError, match=r"rank\(s\) 0, 1.*boom everywhere"):
            m.fail_everywhere().result()
        assert m.echo(x=1).result() == 1
        with pytest.raises(RuntimeError, match=r"rank\(s\) 0, 1, 2, 3, 4"):
            m.group_fail_everywhere()
        with pytest.raises(RuntimeError, match=r"rank\(s\) 0, 1, 2, 3, 4, 5, 6"):
            m.global_fail_everywhere()
        assert m.global_echo(x=3) == 3

    def test_unknown_operation(self) -> None:
        """Unknown operation."""
        if not _is_root():
            return
        m = self.manager
        with pytest.raises(RuntimeError, match="Unknown operation 'nope'"):
            m.nope().result()
        with pytest.raises(RuntimeError, match="Unknown operation 'nope'"):
            m.group_nope()
        with pytest.raises(RuntimeError, match="Unknown operation 'nope'"):
            m.global_nope()
        assert m.echo(x=1).result() == 1

    def test_cancel_before_dispatch(self) -> None:
        """Cancel before dispatch."""
        if not _is_root():
            return
        m = self.manager
        blockers = [m.sleep(seconds=1.0) for _ in range(3)]  # one per thread
        victim = m.sleep(seconds=1.0)
        assert victim.cancel() is True
        assert victim.cancelled()
        assert [b.result() for b in blockers] == [1.0, 1.0, 1.0]
        m._local_queue.join()  # balanced accounting: does not hang
        assert m._local_queue.unfinished_tasks == 0
        assert m.echo(x="after").result() == "after"

    def test_reserved_names_rejected_before_sending(self) -> None:
        """Reserved names rejected before sending."""
        if not _is_root():
            return
        m = self.manager
        for op in ("use_local", "use_group", "use_global", "shutdown"):
            with pytest.raises(ValueError, match="control builtin"):
                m.submit(op)
            with pytest.raises(ValueError, match="control builtin"):
                m.broadcast(op)
            with pytest.raises(ValueError, match="control builtin"):
                m.submit_group(op)
            with pytest.raises(ValueError, match="control builtin"):
                m.submit_global(op)
        assert m.echo(x=1).result() == 1  # workers are untouched
        assert set(m.list_ops()) >= {"echo", "void", "sleep"}
        assert set(m.group_list_ops(mode="group")) >= {"echo", "void"}

    def test_empty_error_message_still_fails(self) -> None:
        """Empty error message still fails."""
        if not _is_root():
            return
        # Any exception text is prefixed by its type, so the caller sees it.
        with pytest.raises(RuntimeError, match="ValueError"):
            self.manager.fail_everywhere().result()

    def test_shutdown_is_idempotent_and_rejects_further_work(self) -> None:
        """Shutdown is idempotent and rejects further work."""
        if not _is_root():
            return
        m = self.manager
        assert m.echo(x=1).result() == 1
        m.shutdown()
        m.shutdown()  # no-op: no second shutdown message is sent
        for fn in (
            lambda: m.submit("echo", x=1),
            lambda: m.broadcast("echo", x=1),
            lambda: m.submit_group("echo", x=1),
            lambda: m.submit_global("echo", x=1),
            lambda: m.start(),
        ):
            with pytest.raises(RuntimeError, match="shut down"):
                fn()
        # The fixture's shutdown() is the third call and must also be a no-op.


# ----------------------------------------------------------------------------
# Extension discovery on a real LammpsEngine, before the engine is started
# ----------------------------------------------------------------------------

_PROBE_EVALUATIONS = 0  # per-process counter, gathered from every worker rank


class _ProbeExtension(EngineExtension):
    """Extension with a property that records (and would fail) if evaluated."""

    @property
    def probe(self) -> Any:
        global _PROBE_EVALUATIONS
        _PROBE_EVALUATIONS += 1
        return self.engine.lmp.version()  # lmp is None before start

    def probe_op(self, x: int) -> int:
        return x * 2


def gather_probe_evaluations(comm: MPI.Comm) -> list[int] | None:
    """Gather the per-rank property evaluation counters on rank 0 (extra_op)."""
    counts = comm.gather(_PROBE_EVALUATIONS, root=0)
    return counts if comm.Get_rank() == 0 else None


@dataclass
class _LammpsConfig:
    """Minimal LAMMPS config for an unstarted engine."""

    pair_style: str = "lj/cut 6.0"
    pair_coeff: str = "* * 0.52 2.274"
    min_style: str = "cg"
    minimize: str = "1e-6 1e-8 1000 10000"
    frz_min: str = "1e-4 1e-6 100 1000"
    verbosity: int = 0


class TestExtensionDiscoveryMPI:
    """Extension discovery on real LammpsEngines before start, over MPI."""

    @pytest.fixture(autouse=True)
    def setup(self) -> Any:
        """Launch 3 unstarted LammpsEngine workers with the probe extension."""
        _require_8_ranks()
        pytest.importorskip("lammps")
        from pykmc.factory import EngineManagerFactory

        MPI.COMM_WORLD.Barrier()
        self.manager = EngineManagerFactory(
            engine_style="lammps",
            engine_config=_LammpsConfig(),
            n_workers=3,
            comm=MPI.COMM_WORLD,
            engine_extensions=[_ProbeExtension],
            extra_ops={"gather_probe_evaluations": gather_probe_evaluations},
        ).launch()
        yield
        if self.manager is not None:
            self.manager.shutdown()
        MPI.COMM_WORLD.Barrier()

    def test_extension_property_never_evaluated_during_discovery(self) -> None:
        """Extension property never evaluated during discovery."""
        if not _is_root():
            return
        m = self.manager
        ops = set(m.list_ops())
        assert "probe_op" in ops
        assert "probe" not in ops
        assert "rank" not in ops  # LammpsEngine property, not an operation
        assert {"start", "close", "create", "register", "command"} <= ops
        # Every worker rank built its registry without touching the property.
        assert m.global_gather_probe_evaluations() == [0] * 7
        # The extension method is dispatchable before the engine is started.
        assert m.probe_op(x=21).result() == 42
        assert m.global_gather_probe_evaluations() == [0] * 7
