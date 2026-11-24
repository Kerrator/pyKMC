"""The shorthand and explicit Manager APIs share one observation point."""

from concurrent.futures import Future
from unittest.mock import Mock

import numpy as np
import pytest
from pykmc.manager import Manager

from .conftest import BASIN, MIN, ORDER, SAVED_MIN, TYPES
from .recorder import recorder


def test_scripted_manager_accepts_direct_and_shorthand(make_case):
    """The controlled worker runs each job exactly once regardless of dispatch syntax."""
    sim, _ = make_case()
    worker = sim.manager
    request = dict(
        config=sim.config,
        central_atom_idx=3,
        positions=MIN,
        saddle_idx=ORDER,
        types=TYPES,
    )
    old = worker.partn_refine(**request)
    explicit = worker.submit("partn_refine", **request)
    assert isinstance(old, Future) and isinstance(explicit, Future)
    assert len(worker.calls) == len(worker.artns) == 2
    np.testing.assert_array_equal(
        old.result().ok_value().saddle_positions,
        explicit.result().ok_value().saddle_positions,
    )
    assert worker.get_total_energy(positions=MIN).result() == sim.total_energy
    assert worker.submit("get_total_energy", positions=SAVED_MIN).result() == BASIN
    np.testing.assert_array_equal(worker.min_calls, [MIN, SAVED_MIN])

    worker.group_minimizer = Mock(return_value="scripted group")
    assert worker.group_minimize_with_results(positions=MIN) == "scripted group"
    assert (
        worker.submit_group("minimize_with_results", positions=MIN) == "scripted group"
    )
    assert worker.group_minimizer.call_count == 2


def test_native_recorder_preserves_return_values_and_caps(monkeypatch, tmp_path):
    """Drive the dispatch recorder without ctypes, engines, MPI or a force calculation."""
    monkeypatch.chdir(tmp_path)
    transport = Mock(spec=Manager)
    transport.submit.return_value = Future()
    wrapped, counts = recorder(transport)
    request = dict(central_atom_idx=3, positions=MIN, saddle_idx=ORDER)
    assert wrapped.partn_refine(**request) is transport.submit.return_value
    assert wrapped.submit("partn_refine", **request) is transport.submit.return_value
    with pytest.raises(RuntimeError, match="local refinement budget"):
        wrapped.submit("partn_refine", **request)
    assert [c.args for c in transport.submit.call_args_list] == [("partn_refine",)] * 2
    for i in (1, 2):
        with np.load(tmp_path / f"local-{i}.npz") as received:
            np.testing.assert_array_equal(received["positions"], MIN)
            np.testing.assert_array_equal(received["ids"], ORDER)
            assert received["center"] == 3

    assert wrapped.group_minimize_with_results() is transport.submit_group.return_value
    assert (
        wrapped.submit_group("minimize_with_results", positions=MIN)
        is transport.submit_group.return_value
    )
    assert (
        wrapped.group_minimize_with_results(positions=SAVED_MIN)
        is transport.submit_group.return_value
    )
    with pytest.raises(RuntimeError, match="group minimization budget"):
        wrapped.submit_group("minimize_with_results", positions=MIN)
    assert [c.args for c in transport.submit_group.call_args_list] == [
        ("minimize_with_results",)
    ] * 3
    assert not (tmp_path / "group-1-start.npy").exists()
    np.testing.assert_array_equal(np.load(tmp_path / "group-2-start.npy"), MIN)
    np.testing.assert_array_equal(np.load(tmp_path / "group-3-start.npy"), SAVED_MIN)
    assert counts == {"local": 2, "group": 3}
