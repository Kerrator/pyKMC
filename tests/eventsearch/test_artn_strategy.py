"""Tests for the `artn` event search strategy, with no LAMMPS.

A stub manager stands in for the engines. Most stubs return prepared results;
one runs the real ARTn core on an analytic surface, on a pool of threads, so
the events of a whole batch can be compared across worker counts.
"""

import inspect
import logging
import random
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

import pykmc.run as run
from pykmc.artn import artn_search, perp_limits
from pykmc.config import ActiveVolume, Config, PartnConfig
from pykmc.engine.artn_search import ArtnSearchExtension
from pykmc.eventsearch import EventSearch, SearchRequest
from pykmc.eventsearch.strategies.artn import (
    ArtnStrategy,
    artn_options,
    ignored_partn_parameters,
)
from pykmc.eventsearch.strategies.dimer import DimerStrategy
from pykmc.result import ErrorType, Ok

from ..test_dimer import GRID, SHIFT, periodic_wells
from ..test_run_event_search_wiring import (  # noqa: F401 (fixture)
    RecordingFactory,
    RecordingKMC,
    recorders,
)
from .test_dimer_strategy import (
    CELL,
    PBC,
    POSITIONS,
    OrderedManager,
    PoolManager,
    StubLoggers,
    _av_config,
    _raw,
    _seeds_through_kmc,
    _system,
)

INPUT = Path(__file__).parents[1] / "data" / "input.in"


def _input(tmp_path: Path, extra_control: str = "", partn: str = "") -> str:
    text = INPUT.read_text().replace("style = partn", "style = artn")
    text = text.replace("[Control] \n", f"[Control] \n{extra_control}")
    text = text.replace("[pARTn] \n", f"[pARTn] \n{partn}")
    path = tmp_path / "input.in"
    path.write_text(text)
    return str(path)


# ---------------------------------------------------------------- config


def test_artn_with_a_partn_section_parses(tmp_path: Path) -> None:
    """The ARTn style reads search parameters from the pARTn section."""
    config = Config.from_ini_file(_input(tmp_path))

    assert config.eventsearch.style == "artn"
    assert config.partn is not None


def test_artn_without_a_partn_section_fails(tmp_path: Path) -> None:
    """Configuration parsing requires the pARTn section for ARTn searches."""
    text = INPUT.read_text().replace("style = partn", "style = artn")
    path = tmp_path / "input.in"
    path.write_text(text.replace("[pARTn] \n", ""))

    with pytest.raises(ValueError, match="partn"):
        Config.from_ini_file(str(path))


def test_strategy_rejects_mixed_nperp_limitation(tmp_path: Path) -> None:
    """Mixing -2 with an explicit limit fails when the strategy is built."""
    config = Config.from_ini_file(_input(tmp_path, partn="nperp_limitation = 4, -2\n"))

    with pytest.raises(ValueError, match="nperp_limitation"):
        ArtnStrategy(config)


@pytest.mark.parametrize("limits", [None, [-2, -2]], ids=["unset", "all-default"])
def test_strategy_accepts_default_nperp_limitation(limits: list[int] | None) -> None:
    """An unset list and an all -2 list reach the core unchanged."""
    config = _av_config(rmov=None)
    config.partn = PartnConfig(nperp_limitation=limits)

    strategy = ArtnStrategy(config)

    assert strategy.options["nperp_limitation"] == limits


def test_options_are_the_core_keywords_from_the_partn_section(tmp_path: Path) -> None:
    """The strategy passes supported pARTn settings under their core keyword names."""
    config = Config.from_ini_file(_input(tmp_path, partn="push_step_size = 0.3\n"))
    options = ArtnStrategy(config).options
    params = inspect.signature(artn_search).parameters

    assert options == artn_options(config.partn)
    assert options["push_step_size"] == 0.3
    assert set(options) <= set(params)
    assert "zseed" not in options and "neigen" not in options
    assert options["nsmooth"] == config.partn.nsmooth
    assert options["nperp_limitation"] == config.partn.nperp_limitation


@pytest.mark.parametrize(
    ("active_volume", "artn_forces", "radii"),
    [(True, "av", (3.0, 12.0)), (True, "full", None), (False, "av", None)],
)
def test_the_av_switch_sends_the_active_volume_radii(
    active_volume: bool, artn_forces: str, radii: tuple | None
) -> None:
    config = _av_config(rmov=3.0 if active_volume else None)
    config.partn = PartnConfig()
    config.activevolume.artn_forces = artn_forces
    options = ArtnStrategy(config).options

    assert options.get("av_radii") == radii
    assert {k: v for k, v in options.items() if k != "av_radii"} == artn_options(
        config.partn
    )


def test_artn_forces_is_unset_unless_given() -> None:
    """Omitted, it is None: searches use the full system, the refiner picks its own."""
    assert ActiveVolume().artn_forces is None
    assert ActiveVolume(artn_forces="full").artn_forces == "full"
    assert ActiveVolume(artn_forces="av").artn_forces == "av"
    with pytest.raises(ValidationError):
        ActiveVolume(artn_forces="mask")


# ---------------------------------------------------------------- wiring


def test_artn_supplies_its_engine_extension(tmp_path: Path) -> None:
    """Selecting ARTn registers the engine operation for its search requests."""
    event_search = EventSearch.create(
        "artn", config=Config.from_ini_file(_input(tmp_path))
    )

    assert isinstance(event_search._strategy, ArtnStrategy)
    assert event_search.engine_extensions == [ArtnSearchExtension]


def test_run_passes_the_artn_extension_to_every_rank(
    recorders: pytest.MonkeyPatch,  # noqa: F811
    tmp_path: Path,
) -> None:
    """Run setup includes the ARTn extension when building the manager."""
    recorders.setattr(sys, "argv", ["pykmc", "-in", _input(tmp_path)])
    run.main()

    (factory,) = RecordingFactory.instances
    assert factory.kwargs["engine_extensions"] == [ArtnSearchExtension]


def test_request_carries_seed_and_movable_and_the_op_its_options(
    tmp_path: Path,
) -> None:
    """Each worker request carries the seed, movable indices and search options."""
    rng = np.random.default_rng(3)
    system = _system(rng.uniform(0.0, 10.0, size=(12, 3)))
    manager = OrderedManager(2, [0, 1])
    config = Config.from_ini_file(_input(tmp_path))
    EventSearch.create("artn", config=config).execute(
        [4, 9], system, manager, StubLoggers()
    )

    for op_name, kwargs in manager.calls:
        assert op_name == "artn_search"
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
        assert kwargs["options"] == artn_options(config.partn)
    assert [kw["central_atom"] for _, kw in manager.calls] == [4, 9]


@pytest.mark.parametrize("rmov", [3.0, None])
def test_movable_atoms_are_the_dimers(rmov: float | None) -> None:
    """ARTn and the dimer choose the same movable atoms with frozen rows."""
    config = _av_config(rmov=rmov, frozen=[1, 5])
    config.partn = PartnConfig()
    positions = np.vstack([POSITIONS, [[1.0, 5.0, 5.0]]])

    assert np.array_equal(
        ArtnStrategy(config).movable_atoms(0, _system(positions)),
        DimerStrategy(config).movable_atoms(0, _system(positions)),
    )


def test_active_volume_smaller_than_cutoff_is_rejected_before_any_submit() -> None:
    """An undersized active volume fails before the manager receives a request."""
    config = _av_config(rmov=3.0)
    config.partn = PartnConfig()
    config.activevolume.ract = 6.0
    manager = OrderedManager(1, [0])
    request = SearchRequest(0, POSITIONS, CELL, np.array(["Ni"] * 5), PBC, 1, None)

    with pytest.raises(ValueError, match="Active Volume radius is smaller"):
        ArtnStrategy(config).submit([request], manager)

    assert manager.calls == []


@pytest.mark.parametrize("artn_forces", ["av", "full"])
def test_av_searches_carry_the_frozen_atoms(artn_forces: str) -> None:
    """On the active volume each request names [frozen_atoms] for the buffer."""
    config = _av_config(rmov=3.0, frozen=[1, 3])
    config.partn = PartnConfig()
    config.activevolume.artn_forces = artn_forces
    strategy = ArtnStrategy(config)
    system = _system(POSITIONS)
    movable = strategy.movable_atoms(0, system)
    request = SearchRequest(0, POSITIONS, CELL, system.types, PBC, 1, movable)
    manager = OrderedManager(1, [0])
    strategy.submit([request], manager)

    ((_, kwargs),) = manager.calls
    np.testing.assert_array_equal(kwargs["movable"], [0, 2])
    if artn_forces == "av":
        np.testing.assert_array_equal(kwargs["options"]["av_frozen"], [1, 3])
    else:
        assert "av_frozen" not in kwargs["options"]
    assert "av_frozen" not in strategy.options


# ---------------------------------------------------------------- seeds


CENTRAL = [3, 0, 7, 3, 11, 5, 0, 9]


def test_one_control_seed_gives_the_same_seeds_for_any_worker_count(
    tmp_path: Path,
) -> None:
    """The run-wide seed sequence holds for any worker count and completion order."""
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


# Atoms on the periodic wells' minima, 2.5 Å or more apart in a 10 Å cell.
SITES = [(2, 1, 1), (0, 0, 0), (1, 3, 2), (2, 1, 2), (3, 2, 0), (0, 2, 3)]
WELLS = np.array([[SHIFT + GRID * i, SHIFT + GRID * j, GRID * k] for i, j, k in SITES])


class CoreManager:
    """Runs each op with the real ARTn core on the periodic wells, on threads."""

    def __init__(self, n_workers: int, delay_seed: int) -> None:
        self.pool = ThreadPoolExecutor(max_workers=n_workers)
        self.rng = random.Random(delay_seed)

    def submit(self, op_name: str, **kw: object) -> Future:
        """Run one search on the pool."""
        assert op_name == "artn_search"
        delay = self.rng.uniform(0.0, 0.003)
        mask = np.zeros(len(kw["positions"]), bool)
        mask[kw["movable"]] = True

        def work() -> object:
            threading.Event().wait(delay)
            return artn_search(
                kw["positions"],
                kw["cell"],
                kw["pbc"],
                mask,
                kw["central_atom"],
                kw["seed"],
                periodic_wells,
                **kw["options"],
            )

        return self.pool.submit(work)


def test_same_control_seed_and_other_worker_counts_give_the_same_events(
    tmp_path: Path,
) -> None:
    """The seeded analytic batch reproduces events and failures across pool sizes."""
    from pykmc.initializer import Initializer
    from pykmc.kmc import KMC

    path = _input(tmp_path, extra_control="seed = 20261116\n")
    central = [3, 0, 5, 3, 1, 4, 3, 2]
    batches = []
    for n_workers, delay_seed in ((1, 0), (3, 1), (8, 2)):
        kmc = KMC(Config.from_ini_file(path))  # seeds the run-wide stream
        Initializer(kmc).initialize_event_search()
        kmc.system = _system(WELLS)
        kmc.manager = CoreManager(n_workers, delay_seed)
        kmc.loggers = StubLoggers()
        kmc.execute_event_searches(central)
        batches.append(kmc.event_search.results)

    first = batches[0]
    assert sum(isinstance(r, Ok) for r in first) >= 4  # mostly events, not failures
    for other in batches[1:]:
        for a, b in zip(first, other, strict=True):
            assert type(a) is type(b)
            if isinstance(a, Ok):
                ea, eb = a.ok_value(), b.ok_value()
                assert ea.seed == eb.seed and ea.backend == eb.backend == "artn"
                assert (ea.dE_forward, ea.dE_backward) == (
                    eb.dE_forward,
                    eb.dE_backward,
                )
                assert ea.force_calls == eb.force_calls
                for name in ("min1_positions", "saddle_positions", "min2_positions"):
                    assert np.array_equal(getattr(ea, name), getattr(eb, name))
            else:
                assert a.err_value().variables == b.err_value().variables


# ---------------------------------------------------------------- results


def test_results_go_through_the_shared_rules_as_artn() -> None:
    """Accepted ARTn events keep their seed, barrier and force-call counts."""
    manager = OrderedManager(2, [1, 0])
    config = _av_config(rmov=None)
    config.partn = PartnConfig()
    requests = [
        SearchRequest(2, POSITIONS, CELL, np.array(["Ni"] * 5), PBC, 7, np.arange(5)),
        SearchRequest(3, POSITIONS, CELL, np.array(["Ni"] * 5), PBC, 8, np.arange(5)),
    ]
    futures = ArtnStrategy(config).submit(requests, manager)
    out = [f.result() for f in futures]

    event = out[0].ok_value()
    assert event.central_atom_index == 2
    assert event.backend == "artn" and event.seed == 7
    assert event.dE_forward == pytest.approx(0.6)
    assert event.force_calls == 100 and event.force_calls_endpoints == 50


@pytest.mark.parametrize(
    ("moved_threshold", "accepted_threshold", "npart", "accepted"),
    [(0.25, 0.6, 1, True), (0.5, 0.6, 0, True), (0.25, 0.5, 1, False)],
)
def test_participant_report_and_event_acceptance_use_separate_thresholds(
    moved_threshold: float,
    accepted_threshold: float,
    npart: int,
    accepted: bool,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The report counts strictly larger atomic displacements; acceptance uses the 3N distance."""
    start = np.array([[9.75, 1.0, 1.0], [1.0, 2.0, 3.0]])
    min1 = np.array([[0.25, 1.0, 1.0], [1.0, 2.0, 3.0]])
    raw = _raw(start, delr1=0.5, delr2=2.0, min1=min1)
    future = Future()
    future.set_result(raw)
    manager = OrderedManager(1, [0])
    monkeypatch.setattr(manager, "submit", lambda *args, **kwargs: future)
    config = _av_config(rmov=None)
    config.partn = PartnConfig(delr_thr=moved_threshold)
    config.eventsearch.delr_thr = accepted_threshold
    request = SearchRequest(0, start, CELL, np.array(["Ni", "Ni"]), PBC, raw.seed)
    monkeypatch.setattr(logging.getLogger("log"), "propagate", True)
    with caplog.at_level(logging.DEBUG, logger="log"):
        result = ArtnStrategy(config).submit([request], manager)[0].result()

    assert result.is_ok() == accepted
    if not accepted:
        assert result.err_value().type == ErrorType.EVENT_MINIMA_NOT_MATCH_POSITIONS
    assert (
        f"ARTn atom 0 seed {raw.seed} min1: delr=0.5 Å, npart={npart}"
        in caplog.messages
    )


def test_a_failed_search_maps_to_event_not_found_with_its_reason() -> None:
    """A failed worker search carries its reason and seed into the event error."""

    class FailingManager:
        def submit(self, op_name: str, **kw: object) -> Future:
            future = Future()
            future.set_result(
                _raw(
                    kw["positions"],
                    seed=kw["seed"],
                    success=False,
                    reason="eigenvalue_lost",
                )
            )
            return future

    config = _av_config(rmov=None)
    config.partn = PartnConfig()
    request = SearchRequest(
        1, POSITIONS, CELL, np.array(["Ni"] * 5), PBC, 9, np.arange(5)
    )
    (future,) = ArtnStrategy(config).submit([request], FailingManager())
    info = future.result().err_value()

    assert info.type == ErrorType.EVENT_NOT_FOUND
    assert info.variables["failure_reason"] == "eigenvalue_lost"
    assert info.variables["backend"] == "artn" and info.variables["seed"] == 9


# ---------------------------------------------------------------- ignored settings


def test_pykmc_defaults_and_the_gate_fe_block_ignore_nothing() -> None:
    """Smoothing, the out-of-basin limits and new chances are followed (issue 06)."""
    assert ignored_partn_parameters(PartnConfig()) == {}
    fe = PartnConfig(
        push_dist_thr=4.0, push_step_size=0.3, eigval_thr=-0.5, nnewchance=3, zseed=0
    )
    assert ignored_partn_parameters(fe) == {}


def test_settings_the_core_follows_are_not_listed() -> None:
    """Supported smoothing and perpendicular limits produce no ignored settings."""
    plain = PartnConfig(nsmooth=0, nperp=None, nperp_limitation=None)
    assert ignored_partn_parameters(plain) == {}
    unlimited = PartnConfig(nsmooth=0, nperp=-1, nperp_limitation=[-1])
    assert ignored_partn_parameters(unlimited) == {}


@pytest.mark.parametrize(
    ("nperp", "limitation", "expected"),
    [
        # pyKMC's defaults: the basin's 3, then 3, 4, 8, 12, 16 and none out of it
        (3, [4, 8, 12, 16, -1], [3, 4, 8, 12, 16, -1, -1]),
        # pARTn then starts the list with its first value
        (None, [4, 8, 12, 16, -1], [4, 8, 12, 16, -1, -1]),
        (-1, [4, 8, 12, 16, -1], [4, 8, 12, 16, -1, -1]),
        (5, None, [5, -1]),
        (None, None, [-1, -1]),
        (-1, None, [-1, -1]),
        (3, [], [3, 8, 12, 16, -1]),  # an empty list takes pARTn's default
    ],
)
def test_the_core_gets_partns_perp_limits(
    nperp: int | None, limitation: list[int] | None, expected: list[int]
) -> None:
    """The forwarded basin and climb limits reproduce the expected pARTn schedule."""
    partn = PartnConfig(nperp=nperp, nperp_limitation=limitation)
    options = artn_options(partn)

    assert options["nperp"] == nperp
    assert options["nperp_limitation"] == limitation
    assert perp_limits(options["nperp"], options["nperp_limitation"]) == expected


def test_an_unset_nperp_with_a_list_still_ends_the_basin() -> None:
    """With nperp unset, the basin stops after the list's first value, as in pARTn."""
    from ..test_dimer import NO_PBC, OPEN_CELL, curved_valley, valley_start

    options = artn_options(PartnConfig(nperp=None))
    options.update(push_mode="list", push_step_size=0.3, nevalf_max=400)
    result = artn_search(
        valley_start(),
        OPEN_CELL,
        NO_PBC,
        np.ones(4, bool),
        0,
        5,
        curved_valley,
        **options,
    )

    assert result.success


def test_a_set_zseed_and_neigen_are_listed() -> None:
    """Unsupported eigenvector counts and a pARTn seed appear in the warning."""
    ignored = ignored_partn_parameters(PartnConfig(nsmooth=0, zseed=12345, neigen=2))

    assert "zseed" in ignored and "[Control] seed" in ignored["zseed"]
    assert "neigen" in ignored


def test_ignored_settings_are_logged_once() -> None:
    """Several submits through one strategy log the settings warning once."""
    config = _av_config(rmov=None)
    config.partn = PartnConfig(neigen=2, zseed=7)
    strategy = ArtnStrategy(config)
    request = SearchRequest(
        1, POSITIONS, CELL, np.array(["Ni"] * 5), PBC, 9, np.arange(5)
    )

    class Recorder(logging.Handler):
        def __init__(self) -> None:
            super().__init__(logging.WARNING)
            self.lines = []

        def emit(self, record: logging.LogRecord) -> None:
            self.lines.append(record.getMessage())

    logger = logging.getLogger("log")
    recorder = Recorder()
    logger.addHandler(recorder)
    try:
        strategy.submit([request], OrderedManager(1, [0]))
        strategy.submit([request], OrderedManager(1, [0]))
    finally:
        logger.removeHandler(recorder)

    assert len(recorder.lines) == 1
    assert "neigen" in recorder.lines[0] and "zseed" in recorder.lines[0]


def test_default_perp_limit_sentinel_reaches_lanczos() -> None:
    """With both defaults requested, four basin PERP steps lead to Lanczos."""
    options = artn_options(
        PartnConfig(nperp=-1, nperp_limitation=[-2], ninit=1, nevalf_max=7)
    )
    moves = []
    artn_search(
        np.zeros((1, 3)),
        CELL,
        PBC,
        np.ones(1, bool),
        0,
        4,
        lambda x: (0.5 * float(np.sum(x * x)), -x),
        observe=lambda move, state: moves.append(move),
        **options,
    )
    assert moves == ["start", "init", *(["perp"] * 4), "lanczos"]
