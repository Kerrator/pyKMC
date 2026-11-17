"""Tests for the optional search details of an event search.

The details are the energies, backend, seed, force calls, curvature, mode and
failure reason. They go on the search result, ride on the error of a failed search, and are
written to the log and the info file only. No LAMMPS or pARTn is needed: pARTn
is replaced by a stand-in handle.
"""

from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from pykmc.info_simulation import event_search_log_line, info_event_search_records
from pykmc.kmc import KMC
from pykmc.result import (
    SEARCH_DETAIL_FIELDS,
    Err,
    EventSearchOutput,
    KMCLoopInfo,
    Ok,
    ReferenceEventSearchInfo,
)
from pykmc.search_rules import (
    SearchEndpoints,
    add_search_details,
    lammps_box,
    map_to_full_system,
    orient_mode,
    partn_search_details,
    search_result,
)

CELL = np.diag([10.0, 10.0, 10.0])
PBC = np.array([True, True, True])


def _output() -> EventSearchOutput:
    positions = np.zeros((3, 3))
    return EventSearchOutput(
        0, positions, positions + 0.1, positions + 0.2, 0.5, 0.3, 1
    )


def _result(delr1: float, delr2: float, error: tuple = (0, "")) -> object:
    start = np.zeros((3, 3))
    return search_result(
        central_atom_index=2,
        error=error,
        read_distances=lambda: (delr1, delr2),
        read_endpoints=lambda: SearchEndpoints(
            e_min1=-10.0,
            e_saddle=-9.4,
            e_min2=-9.8,
            min1_positions=start,
            saddle_positions=start + 0.1,
            min2_positions=start + 0.2,
            move_atom_index=1,
        ),
        delr_threshold=0.5,
    )


# ------------------------------------------------------------ result fields


def test_the_details_are_unset_by_default() -> None:
    """The details are unset by default."""
    out = _output()

    assert all(getattr(out, name) is None for name in SEARCH_DETAIL_FIELDS)
    assert out.cell is None and out.types is None
    names = [f.name for f in fields(EventSearchOutput)]
    assert set(names[names.index("backend") :]) == set(SEARCH_DETAIL_FIELDS)


def test_search_result_orients_the_energies() -> None:
    """Search result orients the energies."""
    plain = _result(0.01, 2.0).ok_value()
    swapped = _result(2.0, 0.01).ok_value()

    assert (plain.e_initial, plain.e_saddle, plain.e_final) == (-10.0, -9.4, -9.8)
    assert (swapped.e_initial, swapped.e_saddle, swapped.e_final) == (-9.8, -9.4, -10.0)
    assert swapped.dE_forward == pytest.approx(swapped.e_saddle - swapped.e_initial)
    assert plain.backend is None and plain.mode is None


def test_search_result_leaves_the_errors_as_they_were() -> None:
    """Search result leaves the errors as they were."""
    not_found = _result(0.01, 2.0, error=(-10, "boom")).err_value()
    rejected = _result(1.2, 2.4).err_value()

    assert not_found.details == (-10, "boom") and not_found.variables is None
    assert rejected.variables == {"delr1": 1.2, "delr2": 2.4}


# ------------------------------------------------------- add_search_details


def test_details_are_set_on_an_event_and_none_is_skipped() -> None:
    """Details are set on an event and none is skipped."""
    mode = np.ones((3, 3))
    result = add_search_details(
        _result(0.01, 2.0), backend="x", seed=5, curvature=None, mode=mode
    )

    out = result.ok_value()
    assert (out.backend, out.seed, out.curvature) == ("x", 5, None)
    assert out.mode is mode
    with pytest.raises(AttributeError):
        add_search_details(_result(0.01, 2.0), nonsense=1)


def test_a_failed_search_carries_its_details_on_the_error() -> None:
    """A failed search carries its details on the error."""
    result = add_search_details(
        _result(0.01, 2.0, error=(1, "max_iterations")),
        backend="dimer",
        seed=7,
        force_calls=40,
        force_calls_endpoints=0,
        mode=np.ones((3, 3)),
        e_saddle=None,
    )

    assert isinstance(result, Err)
    assert result.err_value().variables == {
        "failure_reason": "max_iterations",
        "backend": "dimer",
        "seed": 7,
        "force_calls": 40,
        "force_calls_endpoints": 0,
    }


def test_a_rejected_search_keeps_its_distances_and_gets_a_reason() -> None:
    """A rejected search keeps its distances and gets a reason."""
    result = add_search_details(_result(1.2, 2.4), backend="dimer", e_saddle=-9.4)

    assert result.err_value().variables == {
        "delr1": 1.2,
        "delr2": 2.4,
        "failure_reason": "minima_not_matching_positions",
        "backend": "dimer",
        "e_saddle": -9.4,
    }


def test_an_empty_backend_message_still_gives_a_failure_reason() -> None:
    """An empty backend message still gives a failure reason."""
    result = add_search_details(_result(0.0, 0.0, error=(-1, "")), backend="partn")

    assert result.err_value().variables["failure_reason"] == "no_event_found"


# --------------------------------------------------------------- orient_mode


def test_orient_mode_uses_the_minimum_image_and_normalises() -> None:
    """Orient mode uses the minimum image and normalises."""
    # Atom 0 hops +0.2 A in x across the boundary (9.9 -> 0.1); the raw
    # difference, -9.8 A, would point the other way.
    initial = np.array([[9.9, 5.0, 5.0], [5.0, 5.0, 5.0]])
    saddle = np.array([[0.1, 5.0, 5.0], [5.0, 5.0, 5.0]])
    mode = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    assert np.allclose(orient_mode(mode, initial, saddle, CELL, PBC), mode / 2)
    assert np.allclose(orient_mode(-mode, initial, saddle, CELL, PBC), mode / 2)
    no_pbc = np.array([False, True, True])
    assert np.allclose(orient_mode(mode, initial, saddle, CELL, no_pbc), -mode / 2)
    assert np.array_equal(
        orient_mode(np.zeros((2, 3)), initial, saddle, CELL, PBC), np.zeros((2, 3))
    )


# --------------------------------------------------------------------- pARTn


class FakeArtn:
    """A pARTn handle after a search; logs what is read."""

    def __init__(self, values: dict) -> None:
        self.values = values
        self.read: list[str] = []

    def extract(self, name: str) -> object:
        """Return a value as pypARTn would, and log the read."""
        self.read.append(name)
        return self.values[name]


# pARTn's sentinel for a count it never set (-NAN_INT).
UNSET = -2147483647


def _artn(
    has_sad: bool = True,
    last_step: int = 75,
    swapped: bool = False,
    stepped: bool = True,
) -> FakeArtn:
    """Return pARTn's state after a search: saddle at step 41, last call at ``last_step``.

    Two active-volume atoms (full-system atoms 3 and 1); atom 3 hops +0.3 A in
    x across the boundary. pARTn's eigenvector points the other way.
    """
    min1 = np.array([[9.9, 5.0, 5.0], [2.0, 2.0, 2.0]])
    saddle = np.array([[0.2, 5.0, 5.0], [2.0, 2.0, 2.0]])
    min2 = np.array([[0.6, 5.0, 5.0], [2.0, 2.0, 2.0]])
    if swapped:
        min1, min2 = min2, min1
    return FakeArtn(
        {
            "nevalf": np.int32(last_step if stepped else -1),
            "has_sad": has_sad,
            "nevalf_sad": np.int32(41) if has_sad else UNSET,
            "etot_sad": np.float64(-9.4),
            "eigval_sad": np.float64(-2.5),
            "delr_min1": 2.0 if swapped else 0.01,
            "delr_min2": 0.01 if swapped else 2.0,
            "tau_min1": min1,
            "tau_min2": min2,
            "tau_sad": saddle,
            "eigen_sad": np.array([[-0.6, 0.0, 0.0], [0.0, 0.0, -0.8]]),
        }
    )


def _to_full(array: np.ndarray) -> np.ndarray:
    return map_to_full_system(array, np.array([3, 1]), np.zeros((5, 3)))


@pytest.mark.parametrize("swapped", [False, True])
def test_partn_details_of_an_accepted_search(swapped: bool) -> None:
    """Partn details of an accepted search."""
    details = partn_search_details(
        _artn(swapped=swapped), True, 12345, CELL, PBC, _to_full
    )

    mode = details.pop("mode")
    assert details == {
        "backend": "partn",
        "seed": 12345,
        "e_saddle": -9.4,
        "curvature": -2.5,
        "force_calls": 42,
        "force_calls_endpoints": 34,
    }
    assert all(type(v) in (str, int, float) for v in details.values())
    # Oriented from the initial minimum (the closer one, pARTn's second when
    # swapped) to the saddle: +x for atom 3, through the boundary. Measured
    # from the other minimum it would be -x. Mapped to the full system.
    want = np.zeros((5, 3))
    want[3] = [0.6, 0.0, 0.0]
    want[1] = [0.0, 0.0, 0.8]
    assert np.allclose(mode, want)


def test_partn_seed_is_unset_when_zseed_is_zero() -> None:
    """Partn seed is unset when zseed is zero."""
    details = partn_search_details(_artn(), True, 0, CELL, PBC, _to_full)

    assert details["seed"] is None


def test_partn_failure_before_a_saddle_counts_to_the_failure() -> None:
    """Partn failure before a saddle counts to the failure."""
    artn = _artn(has_sad=False, last_step=299)

    details = partn_search_details(artn, False, 0, CELL, PBC, _to_full)

    assert details == {
        "backend": "partn",
        "seed": None,
        "force_calls": 300,
        "force_calls_endpoints": 0,
    }
    assert artn.read == ["nevalf", "has_sad"]


def test_partn_search_without_a_step_reports_no_count() -> None:
    """Partn search without a step reports no count."""
    artn = _artn(has_sad=False, stepped=False)

    details = partn_search_details(artn, False, 7, CELL, PBC, _to_full)

    assert details == {"backend": "partn", "seed": 7}
    assert artn.read == ["nevalf"]


def test_partn_failure_after_the_saddle_counts_both_parts() -> None:
    """Partn failure after the saddle counts both parts."""
    details = partn_search_details(_artn(last_step=50), False, 0, CELL, PBC, _to_full)

    assert details["force_calls"] == 42
    assert details["force_calls_endpoints"] == 9
    assert "mode" not in details


def test_lammps_box_puts_the_tilts_in_place() -> None:
    """Lammps box puts the tilts in place."""
    box = lammps_box(
        np.array([1.0, 2.0, 3.0]), np.array([5.0, 7.0, 9.0]), 0.1, 0.2, 0.3
    )

    assert np.array_equal(box, [[4.0, 0.0, 0.0], [0.1, 5.0, 0.0], [0.3, 0.2, 6.0]])


# ---------------------------------------------------- records and log lines


def _searches() -> tuple[list[int], list]:
    mode = np.zeros((5, 3))
    mode[3] = [0.123456789, -0.5, 0.0]
    mode[1] = [0.0, 0.0, 1e-7]
    ok = add_search_details(
        _result(0.01, 2.0),
        backend="dimer",
        seed=np.int64(7),
        e_saddle=np.float64(-9.4),
        curvature=np.float64(-2.0),
        force_calls=np.int64(100),
        force_calls_endpoints=50,
        mode=mode,
    )
    failed = add_search_details(
        _result(0.0, 0.0, error=(1, "max_iterations")),
        backend="dimer",
        seed=8,
        force_calls=40,
        force_calls_endpoints=0,
    )
    rejected = add_search_details(
        _result(1.2, 2.4),
        backend="dimer",
        seed=9,
        e_saddle=-9.1,
        curvature=-1.0,
        force_calls=90,
        force_calls_endpoints=60,
        mode=mode,
    )
    bare = Ok(_output())
    return [4, 0, 2, 1], [ok, failed, rejected, bare]


def _plain(value: object) -> bool:
    if isinstance(value, dict):
        return all(type(k) is str and _plain(v) for k, v in value.items())
    if isinstance(value, list):
        return all(_plain(v) for v in value)
    return type(value) in (int, float, str)


def test_records_hold_plain_values_and_the_sparse_mode() -> None:
    """Records hold plain values and the sparse mode."""
    records = info_event_search_records(*_searches())

    assert _plain(records)
    assert records[0] == {
        "central_atom": 4,
        "backend": "dimer",
        "seed": 7,
        "e_initial": -10.0,
        "e_saddle": -9.4,
        "e_final": -9.8,
        "force_calls": 100,
        "force_calls_endpoints": 50,
        "curvature": -2.0,
        "mode": [[1, 0.0, 0.0, 1e-07], [3, 0.123457, -0.5, 0.0]],
    }
    assert records[1] == {
        "central_atom": 0,
        "backend": "dimer",
        "seed": 8,
        "failure_reason": "max_iterations",
        "force_calls": 40,
        "force_calls_endpoints": 0,
    }
    assert records[2]["failure_reason"] == "minima_not_matching_positions"
    assert "mode" not in records[2] and "e_initial" not in records[2]
    assert records[3] == {"central_atom": 1}
    assert list(records[0]) == ["central_atom", *SEARCH_DETAIL_FIELDS[:1]] + [
        "seed",
        "e_initial",
        "e_saddle",
        "e_final",
        "force_calls",
        "force_calls_endpoints",
        "curvature",
        "mode",
    ]


def test_info_file_gains_the_records_after_the_existing_lines() -> None:
    """Info file gains the records after the existing lines."""
    central_atoms, results = _searches()
    records = info_event_search_records(central_atoms, results)
    counts = ReferenceEventSearchInfo(4, 2, {"no_event_found": 1})
    before = KMCLoopInfo(step=3, reference_event_searches_info=counts).output_msg()

    after = KMCLoopInfo(
        step=3, reference_event_searches_info=counts, reference_event_searches=records
    ).output_msg()

    assert after.startswith(before.rstrip("\n"))
    assert "!!" not in after  # no numpy or python tags
    assert "  - [1, 0.0, 0.0, 1.0e-07]" in after  # one flow row per atom
    doc = yaml.safe_load(after)
    assert doc["reference_event_searches"] == records
    # Force calls per saddle and the success rate, from the info file alone.
    searches = doc["reference_event_searches"]
    saddles = [s for s in searches if "e_saddle" in s]
    assert sum(s["force_calls"] for s in saddles) / len(saddles) == 95
    successes = [s for s in searches if "failure_reason" not in s]
    assert len(successes) / len(searches) == 0.5
    # A step without searches writes no new key.
    empty = KMCLoopInfo(
        step=3, reference_event_searches_info=counts, reference_event_searches=[]
    ).output_msg()
    assert empty == before


def test_log_lines_give_every_field_a_search_set() -> None:
    """Log lines give every field a search set."""
    records = info_event_search_records(*_searches())

    lines = [event_search_log_line(r) for r in records]

    assert lines[0] == (
        "\t :=> Search atom 4 (dimer, seed 7): saddle -9.400000 eV, initial "
        "-10.000000 eV, final -9.800000 eV, curvature -2.0000 eV/A^2, force calls "
        "100 + 50, mode 3 0.515, 1 0.000"
    )
    assert lines[1] == (
        "\t :=> Search atom 0 (dimer, seed 8): failed: max_iterations, force calls 40 + 0"
    )
    assert lines[3] == "\t :=> Search atom 1"


def test_kmc_logs_one_debug_line_per_search() -> None:
    """Kmc logs one debug line per search."""
    lines = []
    loggers = SimpleNamespace(debug=lambda name, msg: lines.append((name, msg)))
    central_atoms, results = _searches()

    def kmc(verbosity: int) -> SimpleNamespace:
        control = SimpleNamespace(verbosity=verbosity)
        return SimpleNamespace(loggers=loggers, config=SimpleNamespace(control=control))

    records = KMC.log_event_searches(kmc(2), central_atoms, results)

    assert records == info_event_search_records(central_atoms, results)
    assert lines == [("log", event_search_log_line(r)) for r in records]
    # With verbosity 0 the info file is not written: no record is built.
    lines.clear()
    assert KMC.log_event_searches(kmc(0), central_atoms, results) == []
    assert lines == []


def test_the_restart_file_takes_no_search_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The restart file takes no search data."""
    monkeypatch.chdir(tmp_path)
    kmc = SimpleNamespace(loggers=SimpleNamespace(_start_wall_time=0.0))

    KMC._save_restart_file(kmc, 3, 1.5)

    assert set(np.load(tmp_path / "restart_3.npz").files) == {
        "last_step",
        "last_time",
        "total_wall_time",
    }
