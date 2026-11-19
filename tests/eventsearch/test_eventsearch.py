"""Tests for the EventSearch facade.

A stub strategy stands in for pARTn, so these tests exercise only the facade:
construction without a manager, a system or loggers, one request per central
atom, results kept in request order, and centring of the successful results.
"""

from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pykmc.config import Config
from pykmc.eventsearch import EventSearch, EventSearchStrategy, SearchRequest
from pykmc.eventsearch.strategies.partn import PartnStrategy
from pykmc.initializer import Initializer
from pykmc.kmc import KMC
from pykmc.result import Err, ErrorInfo, ErrorType, EventSearchOutput, Ok

INPUT = Path(__file__).parents[1] / "data" / "input.in"


class StubStrategy(EventSearchStrategy):
    """Records the requests and returns one prepared result per request."""

    name = "stub_search_test_strategy"

    def __init__(self, config: object = None) -> None:
        self.config = config
        self.calls = []
        self.results = None

    def submit(self, requests: list, manager: object) -> list:
        self.calls.append((requests, manager))
        futures = []
        for i, request in enumerate(requests):
            future = Future()
            if self.results is None:
                future.set_result(request.central_atom)
            else:
                future.set_result(self.results[i])
            futures.append(future)
        return futures


class StubLoggers:
    """Records the log lines and the progress bar updates."""

    def __init__(self) -> None:
        self.lines = []
        self.progress = []

    def info(self, name: str, message: str) -> None:
        self.lines.append((name, message))

    def progress_bar(self, name: str, done: int, total: int) -> None:
        self.progress.append((name, done, total))


def _system(n_atoms: int = 4) -> SimpleNamespace:
    rng = np.random.default_rng(0)
    return SimpleNamespace(
        positions=rng.uniform(0.0, 10.0, size=(n_atoms, 3)),
        cell=np.diag([10.0, 10.0, 10.0]),
        types=np.array(["Ni"] * n_atoms),
        pbc=np.array([True, True, True]),
    )


def test_create_builds_without_manager_system_or_loggers() -> None:
    event_search = EventSearch.create("stub_search_test_strategy", config="cfg")

    assert isinstance(event_search._strategy, StubStrategy)
    assert event_search._strategy.config == "cfg"
    assert event_search.system is None
    assert event_search.results is None


def test_create_dispatches_on_eventsearch_style() -> None:
    config = Config.from_ini_file(INPUT)

    event_search = EventSearch.create(config.eventsearch.style, config=config)

    assert config.eventsearch.style == "partn"
    assert isinstance(event_search._strategy, PartnStrategy)
    assert event_search._strategy.config is config


def test_kmc_reuses_strategy_and_preserves_each_search_batch() -> None:
    class Manager:
        def __init__(self) -> None:
            self.calls = []

        def submit(self, op_name: str, **kwargs) -> Future:
            self.calls.append((op_name, kwargs))
            atom = kwargs["central_atom_idx"]
            positions = kwargs["positions"]
            future = Future()
            future.set_result(
                Ok(
                    EventSearchOutput(
                        central_atom_index=atom,
                        min1_positions=positions,
                        saddle_positions=positions + 0.1,
                        min2_positions=positions + 0.2,
                        dE_forward=0.5,
                        dE_backward=0.4,
                        move_atom_index=atom,
                    )
                )
            )
            return future

    config = Config.from_ini_file(str(INPUT))
    kmc = KMC(config)
    Initializer(kmc).initialize_event_search()
    facade = kmc.event_search
    kmc.system = _system()
    kmc.manager = Manager()
    kmc.loggers = StubLoggers()

    first = kmc.execute_event_searches([1, 0])
    kmc.system.cell *= 2
    second = kmc.execute_event_searches([3])

    assert isinstance(facade._strategy, PartnStrategy)
    assert facade._strategy.config is config
    assert first._strategy is second._strategy is facade._strategy
    assert [r.ok_value().central_atom_index for r in first.results] == [1, 0]
    assert [r.ok_value().central_atom_index for r in second.results] == [3]
    for batch, atoms, center in [(first, [1, 0], 5.0), (second, [3], 10.0)]:
        successes = batch.get_successes_results()
        assert [out.central_atom_index for out in successes] == atoms
        for out in successes:
            np.testing.assert_allclose(out.min1_positions[out.move_atom_index], center)
            np.testing.assert_array_equal(out.cell, np.eye(3) * 2 * center)
    assert [c[0] for c in kmc.manager.calls] == ["partn_search"] * 3
    assert [c[1]["central_atom_idx"] for c in kmc.manager.calls] == [1, 0, 3]
    assert all(c[1]["config"] is config for c in kmc.manager.calls)
    assert facade.system is kmc.system
    assert kmc.event_search.results is second.results
    assert kmc.loggers.lines == [
        ("log", "\t :=> Searching 2 reference events"),
        ("log", "\t :=> Searching 1 reference events"),
    ]


def test_create_rejects_an_unknown_style() -> None:
    with pytest.raises(ValueError, match="unknown"):
        EventSearch.create("no_such_search_style")


def test_execute_sends_one_request_per_central_atom_with_own_copies() -> None:
    strategy = StubStrategy()
    event_search = EventSearch(strategy)
    system = _system()
    manager = object()

    event_search.execute([3, 0, 3], system, manager, StubLoggers())

    assert len(strategy.calls) == 1
    requests, got_manager = strategy.calls[0]
    assert got_manager is manager
    assert [r.central_atom for r in requests] == [3, 0, 3]
    for request in requests:
        assert isinstance(request, SearchRequest)
        assert request.seed is None and request.movable is None
        for field in ("positions", "cell", "types", "pbc"):
            array = getattr(request, field)
            assert np.array_equal(array, getattr(system, field))
            assert not np.shares_memory(array, getattr(system, field))
    for field in ("positions", "cell", "types", "pbc"):
        assert not np.shares_memory(
            getattr(requests[0], field), getattr(requests[2], field)
        )


def test_execute_keeps_results_in_request_order_and_logs_as_before() -> None:
    strategy = StubStrategy()
    strategy.results = [object(), object(), object()]
    event_search = EventSearch(strategy)
    system = _system()
    loggers = StubLoggers()

    event_search.execute([2, 1, 0], system, object(), loggers)

    assert event_search.system is system
    assert len(event_search.results) == 3
    assert all(
        a is b for a, b in zip(event_search.results, strategy.results, strict=True)
    )
    assert loggers.lines == [("log", "\t :=> Searching 3 reference events")]
    assert loggers.progress == [
        ("progress", 1, 3),
        ("progress", 2, 3),
        ("progress", 3, 3),
    ]


def test_execute_replaces_the_previous_step_results() -> None:
    strategy = StubStrategy()
    event_search = EventSearch(strategy)

    second_system = _system()

    event_search.execute([0, 1], _system(), object(), StubLoggers())
    event_search.execute([2], second_system, object(), StubLoggers())

    assert event_search.results == [2]
    assert event_search.system is second_system


def test_successes_are_centred_on_the_move_atom() -> None:
    system = _system()
    min1 = system.positions.copy()
    saddle = min1 + 0.1
    min2 = min1 + 0.2
    ok = Ok(
        EventSearchOutput(
            central_atom_index=1,
            min1_positions=min1.copy(),
            saddle_positions=saddle.copy(),
            min2_positions=min2.copy(),
            dE_forward=0.5,
            dE_backward=0.4,
            move_atom_index=2,
        )
    )
    failed = Err(ErrorInfo(type=ErrorType.EVENT_NOT_FOUND, message="not found"))
    strategy = StubStrategy()
    strategy.results = [failed, ok]
    event_search = EventSearch(strategy)

    event_search.execute([0, 1], system, object(), StubLoggers())
    successes = event_search.get_successes_results()

    assert len(successes) == 1
    out = successes[0]
    assert np.allclose(out.min1_positions[2], [5.0, 5.0, 5.0])
    shift = np.array([5.0, 5.0, 5.0]) - min1[2]
    for got, start in (
        (out.min1_positions, min1),
        (out.saddle_positions, saddle),
        (out.min2_positions, min2),
    ):
        assert np.allclose(got, (start + shift) % 10.0)
    np.testing.assert_array_equal(out.cell, system.cell)
