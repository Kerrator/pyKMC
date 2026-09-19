"""oracle: valid seeds retain draws, launched failures preserve cleanup policy."""

import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

import pykmc
import pykmc.run as run
from pykmc.config import Config, ControlConfig
from pykmc.kmc import KMC


def config(seed):
    cfg = Config.from_ini_file(
        str(Path(pykmc.__file__).resolve().parent.parent / "tests/data/input.in")
    )
    cfg.control = ControlConfig.model_validate(
        {**cfg.control.model_dump(), "seed": seed}
    )
    return cfg


@pytest.mark.parametrize("seed", [-1, 2**32, -100000000000, 2**64, 1.5, "invalid"])
def test_invalid_seed_before_factory(monkeypatch, seed):
    events = []
    monkeypatch.setattr(
        run.Config,
        "from_ini_file",
        lambda _: ControlConfig(
            initial_config="unused", n_steps=1, engine="lammps", seed=seed
        ),
    )
    monkeypatch.setattr(
        run, "EngineManagerFactory", lambda **_: events.append("factory")
    )
    monkeypatch.setattr(sys, "argv", ["pykmc", "-in", "unused"])
    with pytest.raises(ValidationError):
        run.main()
    assert events == []


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_valid_boundary_draws_match_independent_generators(seed):
    cfg = config(seed)
    KMC(cfg, manager=object())
    assert random.random() == random.Random(seed).random()
    assert np.random.random() == np.random.RandomState(seed).random_sample()


def install(monkeypatch, cls, shutdown_error=None):
    cfg = config(12345)
    events = []

    def shutdown():
        events.append(("shutdown",))
        if shutdown_error is not None:
            raise shutdown_error

    manager = SimpleNamespace(shutdown=shutdown)
    comm = SimpleNamespace(
        Get_size=lambda: 4, Abort=lambda code: events.append(("abort", code))
    )

    def launch():
        events.append(("launch",))
        return manager

    monkeypatch.setattr(run, "MPI", SimpleNamespace(COMM_WORLD=comm))
    monkeypatch.setattr(run.Config, "from_ini_file", lambda _: cfg)
    monkeypatch.setattr(
        run, "EngineManagerFactory", lambda **_: SimpleNamespace(launch=launch)
    )
    monkeypatch.setattr(run, "KMC", cls)
    monkeypatch.setattr(sys, "argv", ["pykmc", "-in", "unused"])
    return events


@pytest.mark.parametrize(
    "failure", [RuntimeError("constructor error"), KeyboardInterrupt("interrupt")]
)
def test_valid_seed_constructor_failure_aborts(monkeypatch, failure, capsys):
    def constructor(*args, **kwargs):
        raise failure

    events = install(monkeypatch, constructor)
    with pytest.raises(type(failure)) as raised:
        run.main()
    assert raised.value is failure
    assert events == [("launch",), ("abort", 1)]
    assert str(failure) in capsys.readouterr().err


@pytest.mark.parametrize("where", ["constructor", "initialize", "run"])
@pytest.mark.parametrize("status", [0, 1, 7])
def test_systemexit_preserves_status_and_shutdown(monkeypatch, where, status):
    class FakeKMC:
        def __init__(self, *args, **kwargs):
            if where == "constructor":
                raise SystemExit(status)

        def _initialize(self):
            if where == "initialize":
                raise SystemExit(status)

        def run(self):
            raise SystemExit(status)

    events = install(monkeypatch, FakeKMC)
    with pytest.raises(SystemExit) as raised:
        run.main()
    assert raised.value.code == status
    assert events == [("launch",), ("shutdown",)]


def test_normal_return_shuts_down(monkeypatch):
    class FakeKMC:
        def __init__(self, *args, **kwargs):
            pass

        def _initialize(self):
            pass

        def run(self):
            pass

    events = install(monkeypatch, FakeKMC)
    run.main()
    assert events == [("launch",), ("shutdown",)]


def test_shutdown_failure_aborts(monkeypatch):
    def constructor(*args, **kwargs):
        raise SystemExit(3)

    error = RuntimeError("shutdown failure")
    events = install(monkeypatch, constructor, error)
    with pytest.raises(RuntimeError) as raised:
        run.main()
    assert raised.value is error
    assert isinstance(raised.value.__context__, SystemExit)
    assert events == [("launch",), ("shutdown",), ("abort", 1)]
