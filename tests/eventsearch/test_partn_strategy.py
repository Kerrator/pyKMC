"""Tests for the pARTn event search strategy.

A stub manager stands in for the engines, so no LAMMPS or pARTn is needed: the
strategy must submit one ``partn_search`` op per central atom with the
arguments the op takes, and the results must come back in request order.
"""

import subprocess
import sys
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pykmc.config import Config
from pykmc.eventsearch import EventSearch, SearchRequest
from pykmc.eventsearch.strategies.partn import PartnStrategy

ROOT = Path(__file__).parents[2]
INPUT = ROOT / "tests" / "data" / "input.in"
INPUT_ACTIVE_VOLUME = ROOT / "examples" / "Ni_fcc_4000at_monovacancy+sia" / "input.in"


class StubManager:
    """Records each submitted op and returns a future the test resolves."""

    def __init__(self) -> None:
        self.calls = []
        self.futures = []

    def submit(self, op_name: str, **kwargs) -> Future:
        self.calls.append((op_name, kwargs))
        future = Future()
        self.futures.append(future)
        return future


class ReverseStubManager(StubManager):
    """Resolves every op once the last one is submitted, later requests first."""

    def __init__(self, n_requests: int) -> None:
        super().__init__()
        self.n_requests = n_requests
        self.completion_order = []

    def submit(self, op_name: str, **kwargs) -> Future:
        future = super().submit(op_name, **kwargs)
        if len(self.calls) == self.n_requests:
            for index in reversed(range(self.n_requests)):
                self.completion_order.append(index)
                atom = self.calls[index][1]["central_atom_idx"]
                self.futures[index].set_result(("result", atom, index))
        return future


class StubLoggers:
    """Records the facade's log lines."""

    def __init__(self) -> None:
        self.lines = []

    def info(self, name: str, message: str) -> None:
        self.lines.append((name, message))

    def progress_bar(self, name: str, done: int, total: int) -> None:
        pass


def _config(active_volume: bool, ract: float = 9.0, rcut: float = 6.5):
    return SimpleNamespace(
        control=SimpleNamespace(active_volume=active_volume),
        activevolume=SimpleNamespace(ract=ract),
        atomicenvironment=SimpleNamespace(rcut=rcut),
    )


def _requests(central_atoms: list[int]) -> list[SearchRequest]:
    rng = np.random.default_rng(1)
    return [
        SearchRequest(
            central_atom=atom,
            positions=rng.uniform(0.0, 10.0, size=(5, 3)),
            cell=np.diag([10.0, 10.0, 10.0]),
            types=np.array(["Ni"] * 5),
        )
        for atom in central_atoms
    ]


def test_without_active_volume_one_partn_search_per_central_atom() -> None:
    config = _config(active_volume=False)
    requests = _requests([4, 1, 4])
    manager = StubManager()

    futures = PartnStrategy(config).submit(requests, manager)

    assert futures == manager.futures
    assert len(manager.calls) == 3
    for (op_name, kwargs), request in zip(manager.calls, requests, strict=True):
        assert op_name == "partn_search"
        assert set(kwargs) == {"config", "central_atom_idx", "positions", "types"}
        assert kwargs["config"] is config
        assert kwargs["central_atom_idx"] == request.central_atom
        assert kwargs["positions"] is request.positions
        assert kwargs["types"] is request.types


def test_with_active_volume_the_op_also_gets_the_cell() -> None:
    config = _config(active_volume=True)
    requests = _requests([0, 3])
    manager = StubManager()

    PartnStrategy(config).submit(requests, manager)

    assert len(manager.calls) == 2
    for (op_name, kwargs), request in zip(manager.calls, requests, strict=True):
        assert op_name == "partn_search"
        assert set(kwargs) == {
            "config",
            "central_atom_idx",
            "positions",
            "cell",
            "types",
        }
        assert kwargs["config"] is config
        assert kwargs["central_atom_idx"] == request.central_atom
        assert kwargs["positions"] is request.positions
        assert kwargs["cell"] is request.cell
        assert kwargs["types"] is request.types


@pytest.mark.parametrize("ract", [6.5, 5.0])
def test_active_volume_smaller_than_cutoff_is_rejected_before_any_submit(
    ract: float,
) -> None:
    manager = StubManager()

    with pytest.raises(ValueError, match="Active Volume radius is smaller"):
        PartnStrategy(_config(active_volume=True, ract=ract)).submit(
            _requests([0]), manager
        )

    assert manager.calls == []


def test_active_volume_check_runs_at_search_time_after_the_log_line() -> None:
    event_search = EventSearch(PartnStrategy(_config(active_volume=True, ract=5.0)))
    system = SimpleNamespace(
        positions=np.zeros((4, 3)),
        cell=np.diag([10.0, 10.0, 10.0]),
        types=np.array(["Ni"] * 4),
    )
    manager = ReverseStubManager(2)
    loggers = StubLoggers()

    with pytest.raises(ValueError, match="Active Volume radius is smaller"):
        event_search.execute([0, 1], system, manager, loggers)

    assert loggers.lines == [("log", "\t :=> Searching 2 reference events")]
    assert manager.calls == []


def test_results_come_back_in_request_order_when_searches_finish_out_of_order() -> None:
    central_atoms = [7, 2, 7, 5]
    manager = ReverseStubManager(len(central_atoms))
    event_search = EventSearch(PartnStrategy(_config(active_volume=False)))
    system = SimpleNamespace(
        positions=np.zeros((8, 3)),
        cell=np.diag([10.0, 10.0, 10.0]),
        types=np.array(["Ni"] * 8),
    )

    event_search.execute(central_atoms, system, manager, StubLoggers())

    assert manager.completion_order == [3, 2, 1, 0]
    assert event_search.results == [
        ("result", atom, i) for i, atom in enumerate(central_atoms)
    ]


@pytest.mark.parametrize(
    ("path", "active_volume"), [(INPUT, False), (INPUT_ACTIVE_VOLUME, True)]
)
def test_works_with_the_real_config(path: Path, active_volume: bool) -> None:
    config = Config.from_ini_file(str(path))
    manager = StubManager()

    PartnStrategy(config).submit(_requests([0]), manager)

    op_name, kwargs = manager.calls[0]
    assert bool(config.control.active_volume) is active_volume
    assert op_name == "partn_search"
    assert kwargs["config"] is config
    assert ("cell" in kwargs) is active_volume


def test_event_search_imports_without_lammps_or_partn() -> None:
    """The facade and its strategies import with ``lammps`` and ``pypARTn`` unavailable."""
    code = (
        "import sys; sys.modules['lammps'] = None; sys.modules['pypARTn'] = None; "
        "from pykmc.eventsearch import EventSearchStrategy; "
        "assert 'partn' in EventSearchStrategy._registry, EventSearchStrategy._registry; "
        "assert not EventSearchStrategy._import_errors, EventSearchStrategy._import_errors"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)
