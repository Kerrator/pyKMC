"""Tests that `run.py` builds the event search on every rank and wires it in.

The engine factory and the KMC class are replaced by recorders, so no MPI
workers, LAMMPS or pARTn are started.
"""

import sys
from pathlib import Path

import pytest

import pykmc.run as run
from pykmc.engine.dimer_search import DimerSearchExtension
from pykmc.eventsearch.strategies.dimer import DimerStrategy
from pykmc.eventsearch.strategies.partn import PartnStrategy

INPUT = Path(__file__).parent / "data" / "input.in"


def _dimer_input(tmp_path: Path) -> str:
    text = INPUT.read_text().replace("style = partn", "style = dimer")
    path = tmp_path / "input.in"
    path.write_text(text + "\n[Dimer]\nstart_radius = 3.0\n")
    return str(path)


class RecordingFactory:
    """Stands in for `EngineManagerFactory`; `launch` returns ``manager``."""

    manager = None
    instances: list = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        RecordingFactory.instances.append(self)

    def launch(self):
        return RecordingFactory.manager


class RecordingKMC:
    """Stands in for `KMC`; records the event search it is given."""

    instances: list = []

    def __init__(self, config) -> None:
        self.config = config
        self.manager = None
        self.event_search = None
        self.calls = []
        RecordingKMC.instances.append(self)

    def _initialize(self) -> None:
        self.calls.append(("initialize", self.event_search))

    def run(self) -> None:
        self.calls.append(("run", self.event_search))


@pytest.fixture
def recorders(monkeypatch):
    RecordingFactory.instances = []
    RecordingFactory.manager = None
    RecordingKMC.instances = []
    monkeypatch.setattr(run, "EngineManagerFactory", RecordingFactory)
    monkeypatch.setattr(run, "KMC", RecordingKMC)
    monkeypatch.setattr(run, "check_pypartn", lambda config: None)
    return monkeypatch


def _main(monkeypatch, path: str) -> None:
    monkeypatch.setattr(sys, "argv", ["pykmc", "-in", path])
    run.main()


def test_worker_rank_builds_the_facade_and_passes_the_dimer_extension(
    recorders, tmp_path: Path
) -> None:
    _main(recorders, _dimer_input(tmp_path))

    (factory,) = RecordingFactory.instances
    assert factory.kwargs["engine_extensions"] == [DimerSearchExtension]
    assert RecordingKMC.instances == []  # not rank 0


def test_partn_adds_no_extension(recorders) -> None:
    _main(recorders, str(INPUT))

    (factory,) = RecordingFactory.instances
    assert factory.kwargs["engine_extensions"] == []


@pytest.mark.parametrize(
    ("dimer", "strategy"), [(True, DimerStrategy), (False, PartnStrategy)]
)
def test_rank_0_gives_the_same_facade_to_kmc(
    recorders, tmp_path: Path, dimer: bool, strategy: type
) -> None:
    manager = object()
    RecordingFactory.manager = manager

    _main(recorders, _dimer_input(tmp_path) if dimer else str(INPUT))

    (kmc,) = RecordingKMC.instances
    assert kmc.manager is manager
    assert isinstance(kmc.event_search._strategy, strategy)
    assert kmc.calls == [("initialize", kmc.event_search), ("run", kmc.event_search)]
    (factory,) = RecordingFactory.instances
    assert factory.kwargs["engine_extensions"] == kmc.event_search.engine_extensions


def test_initializer_keeps_a_given_event_search(tmp_path: Path) -> None:
    from pykmc.config import Config
    from pykmc.eventsearch import EventSearch
    from pykmc.initializer import Initializer
    from pykmc.kmc import KMC

    config = Config.from_ini_file(_dimer_input(tmp_path))
    kmc = KMC(config)
    given = EventSearch.create("dimer", config=config)
    kmc.event_search = given

    Initializer(kmc).initialize_event_search()

    assert kmc.event_search is given
