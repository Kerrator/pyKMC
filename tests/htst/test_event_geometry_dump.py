"""``control.event_geometry_output``: full min1/saddle/min2 dump per accepted event.

Reference-table rows keep only neighbour-subset positions, which are unusable
for a Hessian, so once a run ends its prefactors can never be recomputed. With
``event_geometry_output`` set, ``ReferenceEventTable.add_events`` writes one
``event_<idx_ref>.npz`` per accepted event holding the FULL geometry, the
species, the cell and both barriers. The dump is independent of the rate
style (a constant run can dump too), off by default, and never raises. No
LAMMPS, no MPI — a FakeManager resolves the htst futures.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pytest import LogCaptureFixture, MonkeyPatch

from pykmc.event_table import ReferenceEventTable
from pykmc.rate_constant.prefactor import EventPrefactors
from pykmc.result import EventSearchOutput


class FakeManager:
    """Return one pre-resolved future per payload (htst style only)."""

    def __init__(self, results: list[EventPrefactors]) -> None:
        self.calls: list[list[dict[str, Any]]] = []
        self._results = results

    def compute_event_prefactors(
        self, config: Any, events: list[dict[str, Any]]
    ) -> list[Future]:
        """Record the payloads and resolve them from ``results``."""
        self.calls.append(list(events))
        futures: list[Future] = []
        for pre in self._results[: len(events)]:
            f: Future = Future()
            f.set_result(pre)
            futures.append(f)
        return futures


def _patch_numpy_nl(monkeypatch: MonkeyPatch) -> None:
    """NeighborsList lists -> numpy arrays (same workaround as the nu0 tests)."""
    import pykmc.event_table as _et
    from pykmc.neighbors_list import NeighborsList as _RealNL

    class _NumpyNL(_RealNL):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            for key in ("rnei", "rcut"):
                if key in self.neighbors_list:
                    self.neighbors_list[key] = [
                        np.array(lst, dtype=np.int64)
                        for lst in self.neighbors_list[key]
                    ]

    monkeypatch.setattr(_et, "NeighborsList", _NumpyNL)


def _event(system: Any, config: Any) -> EventSearchOutput:
    """Trivial self-backward event (min1 == saddle == min2) at atom 0."""
    de = 0.5 * (config.eventsearch.emin_event + config.eventsearch.emax_event)
    return EventSearchOutput(
        central_atom_index=0,
        min1_positions=system.positions,
        saddle_positions=system.positions,
        min2_positions=system.positions,
        dE_forward=de,
        dE_backward=de + 0.01,
        move_atom_index=0,
        cell=system.cell,
    )


def _dump_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("event_*.npz"))


def test_dump_is_off_by_default(config_Ni_4000at_monovacancy_sia: Any) -> None:
    """A run that does not ask for the dump never gets one."""
    assert config_Ni_4000at_monovacancy_sia.control.event_geometry_output is None


def test_constant_style_dumps_full_geometry(
    config_Ni_4000at_monovacancy_sia: Any,
    system_single_type_fcc: Any,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Constant style: the dump holds the FULL geometry, types, cell, barriers."""
    _patch_numpy_nl(monkeypatch)
    config = config_Ni_4000at_monovacancy_sia  # style=constant
    out = tmp_path / "geom"
    config.control.event_geometry_output = str(out)
    fake = FakeManager([])
    table = ReferenceEventTable(config, manager=fake)
    sys_ = system_single_type_fcc
    ev = _event(sys_, config)

    table.add_events([ev], types=list(sys_.types))

    assert fake.calls == []  # the dump never triggers a prefactor fan-out
    fwd_ref = int(table.table.iloc[0]["idx_ref"])
    bwd_ref = int(table.table.iloc[1]["idx_ref"]) if len(table.table) > 1 else -1
    files = _dump_files(out)
    assert [f.name for f in files] == [f"event_{fwd_ref:06d}.npz"]
    d = np.load(files[0])
    assert int(d["idx_ref_forward"]) == fwd_ref
    assert int(d["idx_ref_backward"]) == bwd_ref
    assert int(d["central_atom_idx"]) == 0
    for key in ("min1_positions", "saddle_positions", "min2_positions"):
        np.testing.assert_array_equal(d[key], sys_.positions)
    assert d["types"].tolist() == list(sys_.types)
    np.testing.assert_array_equal(d["cell"], sys_.cell)
    assert float(d["dE_forward"]) == ev.dE_forward
    assert float(d["dE_backward"]) == ev.dE_backward


def test_htst_style_dumps_and_still_backfills(
    config_Ni_4000at_monovacancy_sia: Any,
    system_single_type_fcc: Any,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """HTST style: the same payload feeds both the dump and the nu0 backfill."""
    _patch_numpy_nl(monkeypatch)
    config = config_Ni_4000at_monovacancy_sia
    config.rateconstant.style = "htst"
    out = tmp_path / "geom"
    config.control.event_geometry_output = str(out)
    pre = EventPrefactors(
        nu0_forward=5.0e12,
        nu0_backward=3.0e12,
        n_free=5,
        n_neg_saddle=1,
        ok_forward=True,
        ok_backward=True,
        reason="",
    )
    fake = FakeManager([pre])
    table = ReferenceEventTable(config, manager=fake)
    sys_ = system_single_type_fcc

    table.add_events([_event(sys_, config)], types=list(sys_.types))

    assert len(fake.calls) == 1
    assert len(_dump_files(out)) == 1
    assert table.table.iloc[0]["nu0"] == pytest.approx(5.0e12)


def test_dump_failure_is_logged_not_raised(
    config_Ni_4000at_monovacancy_sia: Any,
    system_single_type_fcc: Any,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    caplog: LogCaptureFixture,
) -> None:
    """An unwritable dump directory warns and leaves the table intact."""
    _patch_numpy_nl(monkeypatch)
    config = config_Ni_4000at_monovacancy_sia
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    config.control.event_geometry_output = str(blocker / "geom")
    table = ReferenceEventTable(config, manager=FakeManager([]))
    sys_ = system_single_type_fcc

    with caplog.at_level(logging.WARNING, logger="pykmc.event_table"):
        table.add_events([_event(sys_, config)], types=list(sys_.types))

    assert len(table.table) >= 1
    assert any("event geometry dump" in r.getMessage() for r in caplog.records)
