"""Tests for the dimer event search strategy, with no LAMMPS.

A stub manager stands in for the engines and returns prepared dimer results,
so these tests cover what rank 0 does: the requests (cell, periodicity, seed,
movable atoms), the seeds' independence from worker count and completion
order, the corrected move-atom rule, and the mapping of results and failures
through the shared search rules.
"""

import random
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pykmc.config import Config
from pykmc.dimer import DimerResult
from pykmc.engine.dimer_search import DimerSearchExtension
from pykmc.eventsearch import EventSearch, SearchRequest
from pykmc.eventsearch.strategies.dimer import DimerStrategy, dimer_search_result
from pykmc.eventsearch.strategies.partn import PartnStrategy
from pykmc.initializer import Initializer
from pykmc.kmc import KMC
from pykmc.result import SEARCH_DETAIL_FIELDS, ErrorType
from pykmc.search_rules import artn_move_atom, move_atom

INPUT = Path(__file__).parents[1] / "data" / "input.in"
CELL = np.diag([10.0, 10.0, 10.0])
PBC = np.array([True, True, True])


def _input(tmp_path: Path, extra_control: str = "", extra: str = "") -> str:
    text = INPUT.read_text().replace("style = partn", "style = dimer")
    text = text.replace("[Control] \n", f"[Control] \n{extra_control}")
    text += "\n[Dimer]\nstart_radius = 3.0\n" + extra
    path = tmp_path / "input.in"
    path.write_text(text)
    return str(path)


def _system(positions: np.ndarray, pbc=PBC) -> SimpleNamespace:
    return SimpleNamespace(
        positions=np.array(positions, dtype=float),
        cell=CELL.copy(),
        types=np.array(["Ni"] * len(positions)),
        pbc=np.array(pbc, dtype=bool),
    )


def _grid_system(n: int = 12) -> SimpleNamespace:
    rng = np.random.default_rng(3)
    return _system(rng.uniform(0.0, 10.0, size=(n, 3)))


def _raw(
    positions: np.ndarray,
    seed: int = 7,
    success: bool = True,
    reason: str | None = None,
    delr1: float = 0.01,
    delr2: float = 2.0,
    min1: np.ndarray | None = None,
    saddle: np.ndarray | None = None,
    min2: np.ndarray | None = None,
) -> DimerResult:
    if not success:
        return DimerResult(
            success=False,
            reason=reason,
            saddle_positions=None,
            min1_positions=None,
            min2_positions=None,
            e_start=-10.0,
            e_saddle=None,
            e_min1=None,
            e_min2=None,
            delr1=None,
            delr2=None,
            curvature=None,
            mode=None,
            force_calls_saddle=40,
            force_calls_total=40,
            seed=seed,
        )
    min1 = positions.copy() if min1 is None else min1
    saddle = positions + 0.1 if saddle is None else saddle
    min2 = positions + 0.2 if min2 is None else min2
    return DimerResult(
        success=True,
        reason=None,
        saddle_positions=saddle,
        min1_positions=min1,
        min2_positions=min2,
        e_start=-10.0,
        e_saddle=-9.4,
        e_min1=-10.0,
        e_min2=-9.8,
        delr1=delr1,
        delr2=delr2,
        curvature=-2.0,
        mode=np.zeros_like(positions),
        force_calls_saddle=100,
        force_calls_total=150,
        seed=seed,
    )


class StubLoggers:
    def info(self, name: str, message: str) -> None:
        pass

    def progress_bar(self, name: str, done: int, total: int) -> None:
        pass


class OrderedManager:
    """Resolves every op once all are submitted, in a given completion order.

    Each op returns a dimer result that echoes the seed it was given.
    """

    def __init__(self, n_requests: int, order: list[int]) -> None:
        self.n_requests = n_requests
        self.order = order
        self.calls = []
        self.futures = []

    def submit(self, op_name: str, **kwargs) -> Future:
        self.calls.append((op_name, kwargs))
        future = Future()
        self.futures.append(future)
        if len(self.calls) == self.n_requests:
            for index in self.order:
                kw = self.calls[index][1]
                self.futures[index].set_result(_raw(kw["positions"], seed=kw["seed"]))
        return future


class PoolManager:
    """Runs each op on a pool of ``n_workers`` threads, after a random delay."""

    def __init__(self, n_workers: int, delay_seed: int) -> None:
        self.pool = ThreadPoolExecutor(max_workers=n_workers)
        self.rng = random.Random(delay_seed)
        self.seen: dict[int, int] = {}
        self.lock = threading.Lock()
        self.calls = []

    def submit(self, op_name: str, **kwargs) -> Future:
        index = len(self.calls)
        self.calls.append((op_name, kwargs))
        delay = self.rng.uniform(0.0, 0.003)

        def work() -> DimerResult:
            threading.Event().wait(delay)
            with self.lock:
                self.seen[index] = kwargs["seed"]
            return _raw(kwargs["positions"], seed=kwargs["seed"])

        return self.pool.submit(work)


# ---------------------------------------------------------------- wiring


def test_dimer_supplies_its_engine_extension_and_partn_none(tmp_path: Path) -> None:
    dimer = EventSearch.create("dimer", config=Config.from_ini_file(_input(tmp_path)))
    partn = EventSearch.create("partn", config=Config.from_ini_file(str(INPUT)))

    assert isinstance(dimer._strategy, DimerStrategy)
    assert dimer.engine_extensions == [DimerSearchExtension]
    assert isinstance(partn._strategy, PartnStrategy)
    assert partn.engine_extensions == []


def test_dimer_options_are_the_dimer_section(tmp_path: Path) -> None:
    config = Config.from_ini_file(_input(tmp_path))

    assert DimerStrategy(config).options == config.dimer.model_dump()


# ---------------------------------------------------------------- requests


def test_every_request_carries_cell_and_periodicity(tmp_path: Path) -> None:
    system = _system(np.zeros((3, 3)), pbc=[True, True, False])
    for config in (
        Config.from_ini_file(_input(tmp_path)),
        Config.from_ini_file(str(INPUT)),
    ):
        event_search = EventSearch.create(config.eventsearch.style, config=config)
        recorded = []
        event_search._strategy.submit = lambda requests, manager, out=recorded: (
            out.extend(requests) or []
        )

        event_search.execute([0, 2], system, object(), StubLoggers())

        assert len(recorded) == 2
        for request in recorded:
            assert np.array_equal(request.cell, system.cell)
            assert np.array_equal(request.pbc, [True, True, False])
            assert not np.shares_memory(request.pbc, system.pbc)


def test_dimer_request_carries_seed_and_movable_partn_neither(tmp_path: Path) -> None:
    system = _grid_system()
    manager = OrderedManager(2, [0, 1])
    dimer = EventSearch.create("dimer", config=Config.from_ini_file(_input(tmp_path)))
    dimer.execute([4, 9], system, manager, StubLoggers())

    for op_name, kwargs in manager.calls:
        assert op_name == "dimer_search"
        assert set(kwargs) == {
            "positions",
            "types",
            "cell",
            "pbc",
            "movable",
            "central_atom",
            "seed",
            "options",
        }
        assert isinstance(kwargs["seed"], int) and 0 <= kwargs["seed"] < 2**32
        assert np.array_equal(kwargs["movable"], np.arange(len(system.positions)))

    partn = EventSearch(PartnStrategy(Config.from_ini_file(str(INPUT))))
    recorded = []
    partn._strategy.submit = lambda requests, manager: recorded.extend(requests) or []
    partn.execute([4, 9], system, object(), StubLoggers())
    assert [(r.seed, r.movable) for r in recorded] == [(None, None), (None, None)]


def _av_config(rmov: float, frozen: list[int] | None = None) -> SimpleNamespace:
    frozen_atoms = None
    if frozen is not None:
        from pykmc.config import RegionConfig

        frozen_atoms = RegionConfig(indices=frozen)
    return SimpleNamespace(
        control=SimpleNamespace(active_volume=rmov is not None),
        activevolume=SimpleNamespace(rmov=rmov, ract=12.0, artn_forces="full"),
        atomicenvironment=SimpleNamespace(rcut=6.5),
        eventsearch=SimpleNamespace(delr_thr=0.5),
        dimer=SimpleNamespace(model_dump=lambda: {"start_radius": 3.0}),
        frozen_atoms=frozen_atoms,
    )


# Atom 0 sits near the x = 0 face; atom 1 is 0.7 Å away across it, atom 2 is
# 2.5 Å away inside the cell, atom 3 is 4.5 Å away, atom 4 is far.
POSITIONS = np.array(
    [
        [0.5, 5.0, 5.0],
        [9.8, 5.0, 5.0],
        [3.0, 5.0, 5.0],
        [5.0, 5.0, 5.0],
        [5.0, 0.0, 0.0],
    ]
)


def test_active_volume_on_moves_atoms_within_rmov_across_the_boundary() -> None:
    strategy = DimerStrategy(_av_config(rmov=3.0))

    assert strategy.movable_atoms(0, _system(POSITIONS)).tolist() == [0, 1, 2]


def test_rmov_is_the_boundary_of_the_movable_sphere() -> None:
    strategy = DimerStrategy(_av_config(rmov=3.0))
    # Atom 1 is 1e-6 Å inside rmov along x, atom 2 is 1e-6 Å outside along y.
    positions = np.array(
        [[2.0, 5.0, 5.0], [5.0 - 1e-6, 5.0, 5.0], [2.0, 8.0 + 1e-6, 5.0]]
    )

    assert strategy.movable_atoms(0, _system(positions)).tolist() == [0, 1]


def test_active_volume_on_honours_a_non_periodic_axis() -> None:
    strategy = DimerStrategy(_av_config(rmov=3.0))
    system = _system(POSITIONS, pbc=[False, True, True])

    assert strategy.movable_atoms(0, system).tolist() == [0, 2]


def test_active_volume_in_a_skew_cell_uses_the_shortest_image() -> None:
    """Include an atom reached only by a combination of periodic translations."""
    system = _system([[0, 0, 0], [1.5, 0, 0]])
    system.cell = np.array([[10, 0, 0], [4.8, 0.4, 0], [0, 0, 8]])
    strategy = DimerStrategy(_av_config(rmov=1.4))
    assert strategy.movable_atoms(0, system).tolist() == [0, 1]


def test_active_volume_off_moves_every_atom() -> None:
    strategy = DimerStrategy(_av_config(rmov=None))

    assert strategy.movable_atoms(0, _system(POSITIONS)).tolist() == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(("rmov", "expected"), [(3.0, [0, 2]), (None, [0, 2, 3, 4])])
def test_frozen_atoms_never_move(rmov: float | None, expected: list[int]) -> None:
    strategy = DimerStrategy(_av_config(rmov=rmov, frozen=[1, 5]))
    positions = np.vstack([POSITIONS, [[1.0, 5.0, 5.0]]])

    assert strategy.movable_atoms(0, _system(positions)).tolist() == expected


def test_frozen_atoms_from_the_input_file(tmp_path: Path) -> None:
    config = Config.from_ini_file(
        _input(tmp_path, extra="\n[frozen_atoms]\nindices = 1 3\n")
    )

    movable = DimerStrategy(config).movable_atoms(0, _system(POSITIONS))

    assert movable.tolist() == [0, 2, 4]


def test_active_volume_smaller_than_cutoff_is_rejected_before_any_submit() -> None:
    config = _av_config(rmov=3.0)
    config.activevolume.ract = 6.0
    manager = OrderedManager(1, [0])
    request = SearchRequest(0, POSITIONS, CELL, np.array(["Ni"] * 5), PBC, 1, None)

    with pytest.raises(ValueError, match="Active Volume radius is smaller"):
        DimerStrategy(config).submit([request], manager)

    assert manager.calls == []


# ---------------------------------------------------------------- seeds


def _seeds_through_kmc(input_path: str, manager, central_atoms: list[int]) -> list:
    config = Config.from_ini_file(input_path)
    kmc = KMC(config)  # seeds the run-wide stream from [Control] seed
    Initializer(kmc).initialize_event_search()
    kmc.system = _grid_system()
    kmc.manager = manager
    kmc.loggers = StubLoggers()
    kmc.execute_event_searches(central_atoms)
    return kmc.event_search.results


CENTRAL = [3, 0, 7, 3, 11, 5, 0, 9]


def test_one_control_seed_gives_the_same_seeds_for_any_completion_order(
    tmp_path: Path,
) -> None:
    path = _input(tmp_path, extra_control="seed = 20261116\n")
    n = len(CENTRAL)
    orders = [list(range(n)), list(reversed(range(n))), [5, 2, 7, 0, 3, 6, 1, 4]]

    runs = []
    for order in orders:
        manager = OrderedManager(n, order)
        _seeds_through_kmc(path, manager, CENTRAL)
        runs.append([kw["seed"] for _, kw in manager.calls])

    assert runs[0] == runs[1] == runs[2]
    # The i-th request gets the i-th draw of the stream [Control] seed starts.
    stream = random.Random(20261116)
    assert runs[0] == [stream.getrandbits(32) for _ in range(n)]
    assert len(set(runs[0])) == n  # a seed per search, repeats of an atom included


def test_one_control_seed_gives_the_same_seeds_for_any_worker_count(
    tmp_path: Path,
) -> None:
    path = _input(tmp_path, extra_control="seed = 20261116\n")

    per_count = {}
    for n_workers, delay_seed in ((1, 0), (2, 1), (4, 2), (8, 3)):
        manager = PoolManager(n_workers, delay_seed)
        results = _seeds_through_kmc(path, manager, CENTRAL)
        assert all(r.is_ok() for r in results)
        per_count[n_workers] = [manager.seen[i] for i in range(len(CENTRAL))]

    assert per_count[1] == per_count[2] == per_count[4] == per_count[8]
    stream = random.Random(20261116)
    assert per_count[1] == [stream.getrandbits(32) for _ in CENTRAL]


def test_another_control_seed_gives_other_seeds(tmp_path: Path) -> None:
    first = OrderedManager(len(CENTRAL), list(range(len(CENTRAL))))
    second = OrderedManager(len(CENTRAL), list(range(len(CENTRAL))))
    _seeds_through_kmc(_input(tmp_path, extra_control="seed = 1\n"), first, CENTRAL)
    _seeds_through_kmc(_input(tmp_path, extra_control="seed = 2\n"), second, CENTRAL)

    assert [k["seed"] for _, k in first.calls] != [k["seed"] for _, k in second.calls]


def test_partn_requests_draw_nothing_from_the_run_stream() -> None:
    config = Config.from_ini_file(str(INPUT))
    event_search = EventSearch(PartnStrategy(config))
    event_search._strategy.submit = lambda requests, manager: []
    random.seed(5)
    before = random.getstate()

    event_search.execute(CENTRAL, _grid_system(), object(), StubLoggers())

    assert random.getstate() == before


# ---------------------------------------------------------------- move atom


def test_move_atom_picks_the_hopper_across_the_periodic_boundary() -> None:
    # Atom 0 hops 0.5 Å across the x = 0 face (raw displacement +9.5 Å); atom 1,
    # a bystander, moves 0.3 Å; atom 2 jitters 0.1 Å across the y = 0 face (raw
    # displacement -9.9 Å).
    start = np.array([[0.2, 5.0, 5.0], [3.0, 5.0, 5.0], [6.0, 9.95, 5.0]])
    saddle = start.copy()
    saddle[0, 0] = 9.7
    saddle[1, 0] += 0.3
    saddle[2, 1] = 0.05
    raw = _raw(start, min1=start, saddle=saddle, min2=saddle + 0.0)
    request = SearchRequest(1, start, CELL, np.array(["Ni"] * 3), PBC, 7, None)

    result = dimer_search_result(raw, request, delr_threshold=0.5)

    assert result.is_ok()
    assert result.ok_value().move_atom_index == 0
    # ARTn's pick takes the minimum image too (PR #150) and agrees here.
    assert artn_move_atom(start, saddle, CELL, PBC) == 0


def test_move_atom_on_a_non_periodic_axis_takes_the_raw_displacement() -> None:
    start = np.array([[0.2, 5.0, 5.0], [3.0, 5.0, 5.0]])
    saddle = start.copy()
    saddle[0, 0] = 9.7
    saddle[1, 0] += 0.3

    assert move_atom(start, saddle, CELL, np.array([True, True, True])) == 0
    assert move_atom(start, saddle, CELL, np.array([False, True, True])) == 0
    saddle[0, 0] = 0.2 + 0.1
    assert move_atom(start, saddle, CELL, np.array([False, True, True])) == 1


@pytest.mark.parametrize("swapped", [False, True])
def test_triclinic_mover_and_mode_use_the_oriented_minimum(swapped: bool) -> None:
    """Measure the mover and mode from the closest endpoint in either order."""
    cell = np.array([[10.0, 0, 0], [4.0, 10.0, 0], [0, 0, 20.0]])
    start = np.array([[1.0, 1.0, 1.0], [4.0, 2.0, 3.0]])
    saddle = start + [[6.86, 4.9, 0], [0, 0, 6.2]]
    other = saddle.copy()
    other[0, 0] -= 0.7
    mode = np.zeros_like(start)
    mode[0, 0] = 1
    raw = _raw(
        start,
        min1=other if swapped else start,
        min2=start if swapped else other,
        delr1=8.0 if swapped else 0.0,
        delr2=0.0 if swapped else 8.0,
        saddle=saddle,
    )._replace(mode=mode)
    request = SearchRequest(0, start, cell, np.array(["Ni", "Ni"]), PBC, 7, None)

    out = dimer_search_result(raw, request, delr_threshold=0.5).ok_value()

    # Atom 0's shortest displacement is (-3.14, 4.9, 0), length 5.82 Å.
    assert out.move_atom_index == 1
    np.testing.assert_array_equal(out.mode, -mode)
    np.testing.assert_array_equal(out.min1_positions, start)


def test_swapped_return_measures_from_the_oriented_initial_minimum() -> None:
    # The backend's minimum 2 is the start (delr2 < delr1), so the event is
    # swapped. From minimum 2 to the saddle atom 1 moves most (0.5 Å against
    # 0.2 Å); from minimum 1 it would be atom 0 (0.9 Å against 0 Å).
    start = np.array([[2.0, 5.0, 5.0], [5.0, 5.0, 5.0], [8.0, 5.0, 5.0]])
    saddle = start.copy()
    saddle[0, 0] += 0.2
    saddle[1, 0] += 0.5
    min1 = saddle.copy()
    min1[0, 0] -= 0.9
    raw = _raw(start, delr1=1.4, delr2=0.003, min1=min1, saddle=saddle, min2=start)
    request = SearchRequest(1, start, CELL, np.array(["Ni"] * 3), PBC, 7, None)

    result = dimer_search_result(raw, request, delr_threshold=0.5)

    out = result.ok_value()
    assert out.move_atom_index == 1
    assert np.array_equal(out.min1_positions, start)
    assert np.array_equal(out.min2_positions, min1)
    assert out.dE_forward == pytest.approx(-9.4 - -9.8)
    assert out.dE_backward == pytest.approx(-9.4 - -10.0)
    assert move_atom(min1, saddle, CELL, PBC) == 0


# ---------------------------------------------------------------- results


def _request() -> SearchRequest:
    return SearchRequest(2, POSITIONS, CELL, np.array(["Ni"] * 5), PBC, 7, None)


def test_a_connected_saddle_becomes_an_oriented_event() -> None:
    raw = _raw(POSITIONS)

    out = dimer_search_result(raw, _request(), 0.5).ok_value()

    assert out.central_atom_index == 2
    assert out.dE_forward == pytest.approx(0.6)
    assert out.dE_backward == pytest.approx(0.4)
    assert np.array_equal(out.min1_positions, raw.min1_positions)
    assert np.array_equal(out.types, ["Ni"] * 5)


@pytest.mark.parametrize(
    "reason",
    [
        "max_iterations",
        "positive_curvature",
        "zero_force",
        "minimisation_not_converged",
        "higher_order_saddle",
        "saddle_order_unverified",
    ],
)
def test_a_failed_search_is_event_not_found_with_its_reason(reason: str) -> None:
    result = dimer_search_result(
        _raw(POSITIONS, success=False, reason=reason), _request(), 0.5
    )

    assert not result.is_ok()
    err = result.err_value()
    assert err.type == ErrorType.EVENT_NOT_FOUND
    assert err.details == (1, reason)


def test_a_saddle_not_connected_to_the_start_is_minima_not_matching() -> None:
    raw = _raw(POSITIONS, delr1=1.2, delr2=2.4)

    err = dimer_search_result(raw, _request(), 0.5).err_value()

    assert err.type == ErrorType.EVENT_MINIMA_NOT_MATCH_POSITIONS
    assert err.variables == {
        "delr1": 1.2,
        "delr2": 2.4,
        "failure_reason": "minima_not_matching_positions",
        "backend": "dimer",
        "seed": 7,
        "e_saddle": -9.4,
        "curvature": -2.0,
        "force_calls": 100,
        "force_calls_endpoints": 50,
    }


@pytest.mark.parametrize("swapped", [False, True])
def test_a_dimer_event_fills_every_detail_but_the_failure_reason(
    swapped: bool,
) -> None:
    """A dimer event fills every detail but the failure reason."""
    raw = _raw(
        POSITIONS, delr1=2.0 if swapped else 0.01, delr2=0.01 if swapped else 2.0
    )
    raw = raw._replace(mode=np.arange(15.0).reshape(5, 3))

    out = dimer_search_result(raw, _request(), 0.5).ok_value()

    assert out.backend == "dimer"
    assert out.seed == 7 and type(out.seed) is int
    assert out.e_saddle == -9.4
    assert (out.e_initial, out.e_final) == ((-9.8, -10.0) if swapped else (-10.0, -9.8))
    assert out.dE_forward == pytest.approx(out.e_saddle - out.e_initial)
    assert (out.force_calls, out.force_calls_endpoints) == (100, 50)
    assert out.curvature == -2.0
    # Normalised and measured from the initial minimum: in the swapped case
    # that is minimum 2 (start + 0.2 A), past the saddle (start + 0.1 A).
    sign = -1.0 if swapped else 1.0
    assert np.allclose(out.mode, sign * raw.mode / np.linalg.norm(raw.mode))
    assert out.failure_reason is None
    assert all(
        getattr(out, name) is not None
        for name in SEARCH_DETAIL_FIELDS
        if name != "failure_reason"
    )


def test_the_dimer_mode_points_from_the_oriented_initial_minimum() -> None:
    """The dimer mode points from the oriented initial minimum."""
    # Atom 0: start x = 0.5, initial minimum x = 0.8 (0.3 A from the start),
    # saddle x = 0.7. The core's mode (+x) points from the start to the saddle,
    # but from the initial minimum the saddle lies in -x.
    start = POSITIONS.copy()
    initial, saddle = start.copy(), start.copy()
    initial[0, 0] += 0.3
    saddle[0, 0] += 0.2
    mode = np.zeros_like(start)
    mode[0, 0] = 1.0
    raw = _raw(start, delr1=0.3, delr2=2.0, min1=initial, saddle=saddle)._replace(
        mode=mode
    )

    out = dimer_search_result(raw, _request(), 0.5).ok_value()

    assert np.array_equal(out.mode, -mode)


def test_a_failed_dimer_search_carries_reason_force_calls_and_seed() -> None:
    """A failed dimer search carries reason force calls and seed."""
    raw = _raw(POSITIONS, seed=11, success=False, reason="max_iterations")

    err = dimer_search_result(raw, _request(), 0.5).err_value()

    assert err.variables == {
        "failure_reason": "max_iterations",
        "backend": "dimer",
        "seed": 11,
        "force_calls": 40,
        "force_calls_endpoints": 0,
    }


def test_results_come_back_in_request_order() -> None:
    config = _av_config(rmov=None)
    n = 4
    manager = OrderedManager(n, [3, 1, 0, 2])
    requests = [
        SearchRequest(i, POSITIONS, CELL, np.array(["Ni"] * 5), PBC, 100 + i, None)
        for i in range(n)
    ]

    futures = DimerStrategy(config).submit(requests, manager)

    assert [f.result().ok_value().central_atom_index for f in futures] == [0, 1, 2, 3]
    assert [kw["seed"] for _, kw in manager.calls] == [100, 101, 102, 103]


def test_an_error_raised_by_the_op_is_raised_not_turned_into_a_failure() -> None:
    class FailingManager:
        def submit(self, op_name: str, **kwargs) -> Future:
            future = Future()
            future.set_exception(RuntimeError("ERROR: Non-numeric atom coords"))
            return future

    request = _request()

    (future,) = DimerStrategy(_av_config(rmov=None)).submit([request], FailingManager())

    with pytest.raises(RuntimeError, match="Non-numeric"):
        future.result()


# ------------------------------------------------------- saddle order gate


def _two_wells(positions: np.ndarray) -> tuple[float, np.ndarray]:
    """Return the energy and forces of two wells on d = x0 - x1 - (0, 2, 0).

    First-order saddles at 0.9375 eV; an index-two saddle at 1 eV, Hessian
    eigenvalues -4, -1 and 10 eV/Å².
    """
    x, y, z = positions[0] - positions[1] - [0, 2, 0]
    energy = (x * x - 1) ** 2 + (6 * x * x - 0.5) * y * y + y**4 + 5 * z * z
    gradient = np.array(
        [
            4 * x * (x * x - 1) + 12 * x * y * y,
            2 * (6 * x * x - 0.5) * y + 4 * y**3,
            10 * z,
        ]
    )
    forces = np.zeros_like(positions)
    forces[0], forces[1] = -gradient, gradient
    return energy, forces


@pytest.mark.parametrize(("seed", "rows"), [(4, []), (1, [0.9375])])
def test_only_first_order_saddles_reach_the_catalogue(
    tmp_path: Path, seed: int, rows: list
) -> None:
    """Seed 4 climbs to the index-two saddle and adds no row; seed 1 adds its hop."""
    from pykmc.dimer import dimer_search
    from pykmc.event_table import ReferenceEventTable
    from pykmc.system import System

    start = np.array(
        [
            [5.999, 7.0, 5.0],
            [5.0, 5.0, 5.0],
            [6.3, 6.1, 5.9],
            [4.3, 6.2, 5.1],
            [5.5, 6.8, 6.3],
        ]
    )
    cell = np.eye(3) * 12
    pbc = np.zeros(3, bool)
    types = np.array(["Ni", "Cu", "Ag", "Al", "Fe"])
    request = SearchRequest(
        0, start, cell, types, pbc, seed=seed, movable=np.array([0])
    )
    raw = dimer_search(
        start,
        cell,
        pbc,
        np.arange(len(start)) == 0,
        0,
        seed,
        _two_wells,
        start="region",
        remove_translation=True,
    )
    config = Config.from_ini_file(_input(tmp_path, extra="start = region\n"))
    search = EventSearch.create("dimer", config=config)
    search._cell = cell
    search.system = System(types=types, positions=start, cell=cell, pbc=pbc)
    search.results = [dimer_search_result(raw, request, delr_threshold=0.5)]
    table = ReferenceEventTable(config)
    table.add_events(search.get_successes_results())

    barriers = list(table.table["energy_barrier"]) if not table.table.empty else []
    assert barriers == pytest.approx(rows, abs=1e-5)
    if not rows:
        assert raw.reason == "higher_order_saddle"
