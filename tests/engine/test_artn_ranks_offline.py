"""Owned-row ARTn checks with in-process collectives and synthetic force fields."""

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import shlex
from threading import Barrier
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import Mock

import numpy as np
import pytest
from mpi4py import MPI

from pykmc import Config, System
from pykmc.artn import NumpyVectorSpace, artn_search
from pykmc.dimer import DimerResult
from pykmc.engine.artn_ranks import DistributedSpace, OwnedRows, RankForces
from pykmc.engine.dimer_forces import _crop_selection, _interaction_bound
from pykmc.event_table import ReferenceEventTable
from pykmc.eventsearch import EventSearch
from pykmc.utils.geometry import translate
from tests.eventsearch.test_dimer_strategy import _raw
from tests.test_dimer import NO_PBC, OPEN_CELL, curved_valley, valley_start


class ThreadComm:
    """Rendezvous by operation name; peers timing out is an immediate failure."""

    def __init__(self, rank: int, values: list, barrier: Barrier) -> None:
        self.rank, self.values, self.barrier = rank, values, barrier

    def Get_size(self) -> int:  # noqa: N802
        """Return the simulated communicator size."""
        return len(self.values)

    def Get_rank(self) -> int:  # noqa: N802
        """Return the simulated rank."""
        return self.rank

    def exchange(self, kind: str, value: Any, combine: Callable) -> Any:
        """Run one collective on every participating thread."""
        self.values[self.rank] = (kind, value)
        self.barrier.wait()
        assert all(k == kind for k, _ in self.values)
        result = combine([v for _, v in self.values])
        self.barrier.wait()
        return result

    def alltoall(self, values: list) -> list:
        """Deliver one payload per sender to its designated recipient."""
        return self.exchange(
            "alltoall", values, lambda chunks: [p[self.rank] for p in chunks]
        )

    def allgather(self, value: Any) -> list:
        """Gather scalar status values only; a global atom or force gather fails."""
        assert np.isscalar(value)
        return self.exchange("allgather", value, list)

    def allreduce(self, value: Any, op: Any) -> Any:
        """Reduce scalars using the requested operation, without entering MPI."""
        if op == MPI.SUM:
            return self.exchange("sum", value, sum)
        if op == MPI.MAX:
            return self.exchange("max", value, max)
        assert op == MPI.LAND
        return self.exchange("land", value, all)

    def scatter(self, data: Any, root: int) -> Any:
        """Scatter one entry of the leader's payload to each peer."""
        return self.exchange("scatter", data, lambda values: values[root][self.rank])

    def bcast(self, data: Any, root: int) -> Any:
        """Broadcast the leader's value to each peer."""
        return self.exchange("bcast", data, lambda values: values[root])

    def gather(self, data: Any, root: int) -> Any:
        """Assemble final results only on the leader."""
        return self.exchange(
            "gather", data, lambda values: values if self.rank == root else None
        )


def parallel(n: int, task: Callable) -> list:
    """Run fixed-size fake ranks and surface exceptions without hanging a test."""
    barrier = Barrier(n, timeout=5)
    shared = [None] * n

    def run(r: int) -> Any:
        try:
            return task(ThreadComm(r, shared, barrier))
        except BaseException:
            barrier.abort()
            raise

    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(run, range(n)))


def test_swapped_rows_empty_owners_and_seed_order() -> None:
    """Exercise swaps without count changes, empty ranks, reorders and tied offsets."""
    x = np.array([[0.0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 0, 0]])
    mask = np.array([True, True, False, True, True])
    original = [[4, 0], [1], [3], []]
    changed = [[3, 1], [], [4], [0]]

    def run(comm: ThreadComm) -> list:
        r = comm.rank
        vec = DistributedSpace(comm, np.array(original[r], int), mask, x)
        routes = OwnedRows(vec)
        for arrangement in (original, changed, [a[::-1] for a in changed], original):
            physical = np.array(arrangement[r], int)
            routes.update(physical)
            vals = np.column_stack([physical, physical * 3])
            home = routes.transfer(vals, to_home=True)
            np.testing.assert_array_equal(home[:, 0], vec.home)
            np.testing.assert_array_equal(routes.transfer(home, to_home=False), vals)
        plans = vec.draw_orders(
            x[vec.home], 0, OPEN_CELL, NO_PBC, mask[vec.home], "rad", 1.0
        )
        rng = np.random.default_rng(99)
        samples = [
            plans[i].draw(rng, len(vec.home), push=(i == 0)) for i in (0, 1, 0, 0)
        ]
        return vec.home, samples

    result = parallel(4, run)
    plans = NumpyVectorSpace().draw_orders(x, 0, OPEN_CELL, NO_PBC, mask, "rad", 1.0)
    rng = np.random.default_rng(99)
    for j, i in enumerate((0, 1, 0, 0)):
        full = plans[i].draw(rng, len(x), push=(i == 0))
        for ids, samples in result:
            np.testing.assert_array_equal(samples[j], full[ids])


@pytest.mark.parametrize("limit", [2, 9999])
def test_partitioned_core_search(limit: int) -> None:
    """Run the faithful core and compare partial and complete returns with serial."""
    x = valley_start(2)
    mask = np.ones(2, bool)
    options = dict(nevalf_max=limit, push_mode="list", push_step_size=0.3)
    serial = artn_search(x, OPEN_CELL, NO_PBC, mask, 0, 5, curved_valley, **options)

    def run(comm: ThreadComm) -> Any:
        ids = np.flatnonzero(np.arange(len(x)) == comm.rank)
        vec = DistributedSpace(comm, ids, mask, x)

        def query(local: np.ndarray) -> tuple[float, np.ndarray]:
            e, f = curved_valley(local)
            return comm.allreduce(e, MPI.SUM), f

        local = artn_search(
            x[ids], OPEN_CELL, NO_PBC, mask[ids], 0, 5, query, vectors=vec, **options
        )
        return vec.collect_result(local, OPEN_CELL, NO_PBC)

    combined, *peers = parallel(4, run)
    assert peers == [None] * 3
    assert combined.success == serial.success and combined.reason == serial.reason
    for k in (
        "e_saddle",
        "saddle_positions",
        "min1_positions",
        "min2_positions",
        "e_min1",
        "e_min2",
        "delr1",
        "delr2",
    ):
        actual, expected = getattr(combined, k), getattr(serial, k)
        if expected is None:
            assert actual is None
        else:
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_nonfinite_on_one_rank_and_partial_results() -> None:
    """A singleton NaN ends every rank and carried partial endpoint rows reassemble."""
    x = valley_start(3)
    mask = np.array([True, False, True])

    def run(comm: ThreadComm) -> Any:
        home = (
            np.flatnonzero(mask & (np.arange(3) % 2 == (0 if comm.rank == 0 else 1)))
            if comm.rank < 2
            else np.empty(0, int)
        )
        vec = DistributedSpace(comm, home, mask, x)

        def query(y: np.ndarray) -> tuple[float, np.ndarray]:
            f = np.zeros_like(y)
            if comm.rank == 0:
                f[0, 0] = np.nan
            return 0.0, f

        out = artn_search(
            x[home], OPEN_CELL, NO_PBC, mask[home], 0, 5, query, vectors=vec
        )
        assert out.reason == "non_finite_force" and out.force_calls_total == 1
        return vec.collect_result(
            out._replace(saddle_positions=x[home] + 2, min1_positions=x[home] - 1),
            OPEN_CELL,
            NO_PBC,
        )

    out, *peers = parallel(4, run)
    assert all(p is None for p in peers) and out.min2_positions is None
    np.testing.assert_array_equal(out.saddle_positions, x + mask[:, None] * 2)
    np.testing.assert_array_equal(out.min1_positions, x - mask[:, None])


@pytest.mark.parametrize("fault", ["none", "malformed", "command", "create", "cleanup"])
def test_collective_bound(monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    """All ranks use one INFO command/filename and receive one bound or fallback."""
    from pykmc.engine import dimer_forces

    original = dimer_forces.TemporaryDirectory
    lifetimes = []

    class Directory:
        def __init__(self) -> None:
            lifetimes.append(1)
            if fault == "create":
                raise OSError
            self.dir = original()
            self.name = self.dir.name

        def cleanup(self) -> None:
            self.dir.cleanup()
            if fault == "cleanup":
                raise OSError

    monkeypatch.setattr(dimer_forces, "TemporaryDirectory", Directory)

    def run(comm: ThreadComm) -> tuple:
        commands = []

        def command(line: str) -> None:
            commands.append(line)
            if comm.rank == 2 and fault == "command":
                raise OSError
            if comm.rank == 0:
                path = Path(shlex.split(line)[-1])
                path.write_text(
                    "bad" if fault == "malformed" else "Communication cutoff = 7\n"
                )

        return _interaction_bound(
            SimpleNamespace(comm=comm, command=command), True
        ), commands

    results = parallel(4, run)
    assert lifetimes == [1]
    for bound, commands in results:
        assert commands == results[0][1] and len(commands) == int(fault != "create")
        if fault == "none":
            assert bound == pytest.approx(6.50007)
        else:
            assert bound is None


def test_force_rows_frames_ghosts_and_irregular_jump() -> None:
    """Prescribe rotated local rows, force a cross-owner swap, and ignore ghosts."""
    x = np.zeros((2, 3))
    mask = np.ones(2, bool)
    q = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

    def run(comm: ThreadComm) -> None:
        r = comm.rank
        vec = DistributedSpace(comm, np.array([r]), mask, x)
        arrays = dict(
            id=np.array([r + 1, 99]),
            v=np.full((2, 3), 17.0),
            f=np.zeros((2, 3)),
            x=np.zeros((2, 3)),
            type=np.array([r + 1, 99]),
            mass=np.array([0.0, 5.0, 9.0]),
        )
        lmp = SimpleNamespace(
            numpy=SimpleNamespace(extract_atom=arrays.__getitem__),
            extract_global=lambda key: 1,
            get_thermo=lambda key: 4.0,
        )
        commands = []

        def command(text: str) -> None:
            commands.append(text)
            if text.startswith("run 1"):
                np.testing.assert_array_equal(
                    arrays["v"][0], np.array([0.1 * (r + 1), 0, 0]) @ q.T
                )
                np.testing.assert_array_equal(arrays["v"][1], [17.0] * 3)
                arrays["id"][0] = 2 - r
            elif text.startswith("displace_atoms"):
                np.testing.assert_array_equal(
                    arrays["x"][0], np.array([2.0 * (2 - r), 0, 0]) @ q.T
                )
            arrays["f"][0] = np.array([float(arrays["id"][0]), 3.0, 0.0]) @ q.T

        engine = SimpleNamespace(
            lmp=lmp,
            command=command,
            _positions_to_lammps=lambda a: a @ q.T,
            _positions_from_lammps=lambda a: a @ q,
        )
        query = RankForces(engine, np.arange(2), vec, OPEN_CELL, NO_PBC, True)
        np.testing.assert_array_equal(query.masses(), [5.0, 9.0][r])
        for scale in (0.1, 2.0):
            e, f = query(np.array([[scale * (r + 1), 0, 0]]))
            assert e == 4.0
            np.testing.assert_array_equal(f, [[r + 1, 3.0, 0]])
        assert commands == [
            "run 1 pre no post no",
            "displace_atoms all move 0 0 0 units box",
            "run 0 post no",
        ]

    parallel(2, run)


def test_periodic_crop_selection_at_ninety_percent() -> None:
    """Keep two shells across a periodic face and reject a crop over 90 percent."""
    cell = np.diag([16.0, 16.0, 16.0])
    pbc = np.ones(3, dtype=bool)
    start = np.zeros((11, 3))
    start[:, 0] = [0, 15, 1, 2, 3, 3.5, 12.5, 14, 15.5, 8, 1.5]
    movable = np.arange(len(start)) == 1

    # The movable atom is 1 Å away across the face. The reach is 1.5 Å
    # and two shells of 1 Å each bring the selection radius to 3.5 Å.
    selection = _crop_selection(start[:10], movable[:10], start[0], cell, pbc, 1.0)
    assert selection is not None
    kept, within_reach = selection
    np.testing.assert_array_equal(kept, np.arange(9))
    assert within_reach(start[[1, 10]])
    assert not within_reach(start[[3]])

    # Adding one more nearby atom takes the retained fraction from 9/10 to 10/11.
    assert _crop_selection(start, movable, start[0], cell, pbc, 1.0) is None


@pytest.mark.parametrize("case", ["injected", "missing", "crop-fallback"])
def test_worker_dispatch_with_frozen_atoms(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """Reach the public worker with a frozen Lanczos row and reject missing atoms."""
    import sys
    from pykmc.engine.artn_search import ArtnSearchExtension

    start = valley_start(3)
    mask = np.array([True, False, True])
    injected = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    opts = dict(
        ninit=0,
        nevalf_max=7,
        first_lanczos_vector=injected,
        push_mode="list",
        reuse_neighbor_list=False,
    )
    central = 0
    if case == "crop-fallback":
        central = 5
        start = valley_start(6)
        mask = np.arange(len(start)) == central
        initial = np.zeros_like(start)
        initial[central, 0] = 1
        opts = dict(
            ninit=3,
            nperp=0,
            push_step_size=0.4,
            nevalf_max=4,
            push_mode="list",
            initial_push=initial,
            reuse_neighbor_list=False,
            eam_force_crop=True,
        )
    trace = [[] for _ in range(4)]
    cropped_trace = [[] for _ in range(4)]

    class Engine:
        def __init__(self, config: Any, comm: ThreadComm, engine_id: str) -> None:
            self.comm = comm
            self.lmp = self
            self.numpy = self
            self.crop = ".crop." in engine_id

        def start(self) -> None:
            pass

        initialize_parameters = initialize_potential = close = start

        def initialize_system(
            self, types: np.ndarray, x: np.ndarray, *args: Any, **kwargs: Any
        ) -> None:
            self.ids = np.arange(self.comm.rank, len(x), self.comm.Get_size())
            self.x = x[self.ids].copy()
            self.n = len(x)

        def get_natoms(self) -> int:
            return self.n - int(case == "missing")

        def extract_global(self, key: str) -> int:
            assert key == "nlocal"
            return len(self.ids)

        def extract_atom(self, key: str) -> np.ndarray:
            return dict(
                id=self.ids + 1,
                x=self.x,
                f=curved_valley(self.x)[1],
                type=np.ones(len(self.ids), int),
                mass=np.array([0.0, 1.0]),
            )[key]

        def command(self, cmd: str) -> None:
            if cmd == "run 0 post no":
                (cropped_trace if self.crop else trace)[self.comm.rank].append(
                    self.x.copy()
                )
            elif cmd.startswith("info communication"):
                if self.comm.rank == 0:
                    Path(shlex.split(cmd)[-1]).write_text(
                        "Communication cutoff = 0.1\n"
                    )
            else:
                assert cmd == "displace_atoms all move 0 0 0 units box"

        def get_thermo(self, key: str) -> float:
            return self.comm.allreduce(curved_valley(self.x)[0], MPI.SUM)

        @staticmethod
        def _positions_to_lammps(x: np.ndarray) -> np.ndarray:
            return x

        _positions_from_lammps = _positions_to_lammps

    monkeypatch.setitem(
        sys.modules, "pykmc.engine.lammps", SimpleNamespace(LammpsEngine=Engine)
    )

    def invoke(comm: ThreadComm) -> Any:
        ext = ArtnSearchExtension(
            SimpleNamespace(
                comm=comm,
                config=SimpleNamespace(pair_style="eam/fs"),
                engine_id=0,
                register=lambda ext: None,
            )
        )
        if case == "missing":
            with pytest.raises(ValueError, match="did not create every atom"):
                ext.artn_search(
                    start,
                    ["Fe"] * len(start),
                    OPEN_CELL,
                    NO_PBC,
                    np.flatnonzero(mask),
                    central,
                    5,
                    opts,
                )
            return None
        return ext.artn_search(
            start,
            ["Fe"] * len(start),
            OPEN_CELL,
            NO_PBC,
            np.flatnonzero(mask),
            central,
            5,
            opts,
        )

    leader, *others = parallel(4, invoke)
    if case == "missing":
        return
    reference = []

    def query(x: np.ndarray) -> tuple[float, np.ndarray]:
        reference.append(x.copy())
        return curved_valley(x)

    serial = artn_search(
        start,
        OPEN_CELL,
        NO_PBC,
        mask,
        central,
        5,
        query,
        **{
            k: v
            for k, v in opts.items()
            if k not in ("reuse_neighbor_list", "eam_force_crop")
        },
    )
    assert leader.reason == serial.reason and leader.force_calls_total == len(reference)
    assert all(x is None for x in others)
    if case == "crop-fallback":
        # Initial full & crop evaluations, one actual cropped move to .4,
        # then movement beyond .5 causes a collective full fallback for good.
        assert all(len(c) == 2 for c in cropped_trace)
        assert all(len(c) == 3 for c in trace)
        np.testing.assert_allclose(cropped_trace[0][-1], reference[1][[central]])
        for r, calls in enumerate(trace):
            np.testing.assert_allclose(calls, np.asarray(reference)[[0, 2, 3], r::4])
        return
    for rank, calls in enumerate(trace):
        np.testing.assert_allclose(
            calls, np.asarray(reference)[:, rank::4], atol=1e-9, rtol=0
        )
    assert np.max(np.abs(np.asarray(trace[1]) - start[1])) > 1e-5


def test_assembled_periodic_endpoints_reach_the_catalogue(
    system_binary_fcc: System, config_system_single_type: Config
) -> None:
    """Centre a two-shard result with a frozen atom on the periodic face."""
    system = system_binary_fcc
    start = system.positions
    start[0, 0] = -1e-17
    mask = np.arange(len(start)) != 0
    movable = np.flatnonzero(mask)
    mover = 1
    saddle = start.copy()
    saddle[mover, 0] += 0.25
    final = start.copy()
    final[mover, 0] += 0.5

    def collect(comm: ThreadComm) -> DimerResult | None:
        ids = movable[comm.rank :: 2]
        space = DistributedSpace(comm, ids, mask, start)
        shard = _raw(start[ids], saddle=saddle[ids], min2=final[ids])
        return space.collect_result(shard, system.cell, system.pbc)

    result, peer = parallel(2, collect)
    assert peer is None
    for endpoint, expected in (
        (result.min1_positions, start),
        (result.saddle_positions, saddle),
        (result.min2_positions, final),
    ):
        np.testing.assert_array_equal(endpoint[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(endpoint[movable], expected[movable])

    config = config_system_single_type
    future = Future()
    future.set_result(result)
    manager = Mock()
    manager.submit.return_value = future
    search = EventSearch.create("artn", config=config)
    search.execute([mover], system, manager, Mock())
    table = ReferenceEventTable(config)
    (added,) = table.add_events(search.get_successes_results())
    assert added.is_ok()
    row = table.table.iloc[0]
    shift = np.diag(system.cell) / 2 - start[mover]
    for stored, source in (
        (row.initial_positions, start),
        (row.saddle_positions, saddle),
        (row.final_positions, final),
    ):
        expected = translate(source[[mover]], shift, system.cell)
        np.testing.assert_array_equal(stored[row.move_atom_idx], expected[0])
