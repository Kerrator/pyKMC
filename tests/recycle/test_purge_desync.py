"""Regression tests for the recycle x purge desynchronisation.

With event recycling the active table persists across KMC steps, but a
reconstruction failure purges reference events (``ReferenceEventTable.remove``)
without revalidating the rows carried over to the next step. A recycled row
pointing at a purged ref -- routine, since one generic ref maps onto many
sites -- then crashes the next step (``KeyError`` in ``info_active_events``,
``IndexError`` in ``KMC.reconstruction``).

These tests pin:

* ``ReferenceEventTable.remove`` reports the full set it deleted (forward refs
  and their backward siblings) so the caller can resync the active table.
* ``ActiveEventTable.drop_orphans`` evicts rows pointing at a purged ref.
* ``KMC.run`` evicts orphans only AFTER the step log and the recycling prune,
  so the executed row is still readable when it shares the purged ref.
* ``DetectorThreshold.detect`` tolerates a ref that is already gone.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Optional
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from ase.geometry import find_mic

from pykmc.basins.detection import DetectorThreshold
from pykmc.config import EventRecyclingConfig
from pykmc.event_table import ReferenceEventTable
from pykmc.kmc import KMC
from pykmc.result import Err, ErrorInfo, ErrorType, EventRefinementOutput, Ok

from .conftest import make_active_table, row


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _active_row(atom_index: int, num_reference_event: int) -> dict:
    """Build a stub active row pointing at a chosen reference event."""
    return {**row(atom_index), "num_reference_event": num_reference_event}


def _ref_row(idx_ref: int, idx_backward: int, event_id: str) -> dict:
    """Build a minimal reference row with the columns the lookups consume."""
    return {
        "idx_ref": idx_ref,
        "event_id": event_id,
        "idx_backward": idx_backward,
        "id_final": f"final_{event_id}",
        "energy_barrier": 0.5,
        "dra": 1.0,
    }


def _stub_reference_table(rows: list[dict]) -> ReferenceEventTable:
    """Build a ReferenceEventTable whose ``table`` is a tiny stub frame."""
    config = Mock()
    config.control.reference_table = None
    rt = ReferenceEventTable(config)
    rt.table = pd.DataFrame(rows)
    return rt


def _err(error_type: ErrorType) -> Err:
    """Wrap an ``ErrorType`` in the ``Err`` payload reconstruction returns."""
    return Err(ErrorInfo(type=error_type, message="stub", variables={}))


# --------------------------------------------------------------------------- #
# ReferenceEventTable.remove reports what it deleted (fwd + backward)
# --------------------------------------------------------------------------- #
def test_reference_remove_returns_removed_refs_including_backward() -> None:
    """remove() returns the requested refs together with their backward siblings."""
    rt = _stub_reference_table(
        [
            _ref_row(0, 1, "topo0"),
            _ref_row(1, 0, "topo1"),
            _ref_row(2, 3, "topo2"),
            _ref_row(3, 2, "topo3"),
        ]
    )

    removed = rt.remove([0])

    assert removed == {0, 1}
    assert set(rt.table["idx_ref"]) == {2, 3}


# --------------------------------------------------------------------------- #
# ActiveEventTable.drop_orphans
# --------------------------------------------------------------------------- #
def test_drop_orphans_evicts_matching_rows_and_resets_index() -> None:
    """drop_orphans() removes rows by ref, resets the index, returns the count."""
    active = make_active_table(
        [_active_row(10, 0), _active_row(11, 1), _active_row(12, 0)]
    )

    n = active.drop_orphans({0})

    assert n == 2
    assert list(active.table["num_reference_event"]) == [1]
    assert list(active.table.index) == [0]


# --------------------------------------------------------------------------- #
# DetectorThreshold.detect tolerates an already-purged ref
# --------------------------------------------------------------------------- #
def test_detect_is_false_when_executed_ref_was_purged() -> None:
    """Basin detection on an executed row whose ref was purged this step."""
    executed = pd.Series({**_active_row(3, 5), "energy_barrier": 0.1})
    reference = pd.DataFrame([_ref_row(0, 0, "topo0")])  # ref 5 purged

    assert not DetectorThreshold().detect(executed, reference, 0.5, True)


# --------------------------------------------------------------------------- #
# KMC.run: two real steps with recycling
# --------------------------------------------------------------------------- #
class _FakeManager:
    """Engine-pool stand-in exposing the calls ``KMC.run`` makes."""

    def broadcast(self, *args: Any, **kwargs: Any) -> None:
        """Pretend to broadcast to the engines."""


class _RecordingLoggers:
    """No-op logger that records the step-output lines."""

    def __init__(self) -> None:
        self.step_lines: list[dict[str, Any]] = []

    def info(self, *args: Any, **kwargs: Any) -> None:
        """Swallow info messages."""

    def error(self, *args: Any, **kwargs: Any) -> None:
        """Swallow error messages."""

    def events_file_step_first_line(self, *args: Any, **kwargs: Any) -> None:
        """Swallow events-file header lines."""

    def events_applicable_info_line(self, *args: Any, **kwargs: Any) -> None:
        """Swallow events-file applicable-event lines."""

    def table_line_info_kmc(
        self,
        name: str,
        step: int,
        delta_t: float,
        total_time: float,
        num_reference_event: Any,
        energy_barrier: Any,
        k_event: Any,
        *args: Any,
    ) -> None:
        """Record the step line exactly as ``KMC.run`` evaluated it."""
        self.step_lines.append(
            {"step": step, "num_reference_event": int(num_reference_event)}
        )


class _Canned:
    """Search / refinement output returning a fixed list of successes."""

    def __init__(self, outputs: Optional[list[Any]] = None) -> None:
        self._outputs = outputs or []
        self.results: list[Any] = []

    def get_successes_results(self) -> list[Any]:
        """Return the canned outputs."""
        return self._outputs


def test_run_evicts_recycled_orphan_after_prune(
    config_Cu: Any,
    system_Cu: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Two recycled steps: a purge must not orphan rows or break the step log.

    Step 1: B (ref 0) fails INVALID_MIN1, so ref 0 and its backward 1 are
    purged; A, which shares ref 0, is executed; C (ref 0) is far and unmoved,
    so the recycler carries it over. Step 2 adds D (ref 2) and executes it.
    """
    config = config_Cu
    config.control.n_steps = 2
    config.control.basin = False
    config.control.recycle = True
    config.control.reference_table = None

    system = system_Cu
    positions = np.array(system.positions, copy=True)
    _, dist = find_mic(positions - positions[0], system.cell, pbc=True)
    a = 0
    c = int(np.argmax(dist))
    b, d = (int(i) for i in np.argsort(dist)[1:3])
    config.eventrecycling = EventRecyclingConfig(
        style="displacement", movement_thr=0.02, distance_thr=0.5 * dist[c]
    )

    sim = KMC(config)
    sim.system = system
    sim.manager = _FakeManager()
    sim.loggers = _RecordingLoggers()
    sim.visited_environments = set()
    sim.reference_table = ReferenceEventTable(config)
    sim.reference_table.table = pd.DataFrame(
        [
            _ref_row(0, 1, "topo0"),
            _ref_row(1, 0, "topo1"),
            _ref_row(2, 3, "topo2"),
            _ref_row(3, 2, "topo3"),
        ]
    )

    def _event(atom: int, ref: int) -> EventRefinementOutput:
        return EventRefinementOutput(
            central_atom_index=atom,
            saddle_positions=positions.copy(),
            E_saddle=1.0,
            min2_positions=positions.copy(),
            dE_forward=0.5,
            num_reference_event=ref,
            refined="T",
        )

    refinements = iter(
        [
            [_event(b, 0), _event(a, 0), _event(c, 0)],  # step 1
            [_event(d, 2)],  # step 2
        ]
    )

    def _select_atom(atoms: list[int]) -> Callable[[Any], tuple[int, float, float]]:
        order = iter(atoms)

        def _select(active_table: Any) -> tuple[int, float, float]:
            atom = next(order)
            idx = int(np.flatnonzero(active_table.table["atom_index"] == atom)[0])
            return idx, 1.0, 1.0

        return _select

    def _reconstruct(idx: int, active_table: Any) -> Any:
        atom = int(active_table.table.loc[idx].at["atom_index"])
        if atom == b:
            return _err(ErrorType.RECONSTRUCTION_INVALID_MIN1)
        moved = np.array(sim.system.positions, copy=True)
        moved[atom] += np.array([0.3, 0.0, 0.0])
        return Ok(SimpleNamespace(min2_positions=moved, min2_etot=-1.0))

    seen_pairs: list[set[tuple[int, int]]] = []

    def _refine(subset: Any, existing_pairs: Any = None) -> _Canned:
        seen_pairs.append(set(existing_pairs or ()))
        return _Canned(next(refinements))

    monkeypatch.setattr(sim, "minimize_system", lambda: None)
    monkeypatch.setattr(sim, "execute_event_searches", lambda atoms: _Canned())
    monkeypatch.setattr(sim, "add_reference_events", lambda results: [])
    monkeypatch.setattr(sim, "execute_refinements", _refine)
    monkeypatch.setattr(sim, "_select_event", _select_atom([b, a, d]))
    monkeypatch.setattr(sim, "_reconstruction_active_event", _reconstruct)
    monkeypatch.setattr(sim, "_save", lambda: None)
    monkeypatch.setattr(sim, "_append_snapshot_to_trajectory", lambda: None)
    monkeypatch.setattr(sim, "_save_restart_file", lambda *args: None)
    monkeypatch.setattr(sim, "_close", lambda: None)
    monkeypatch.chdir(tmp_path)

    sim.run()

    # Step 1 logged the executed row A although its ref 0 was purged that step.
    assert sim.loggers.step_lines == [
        {"step": 1, "num_reference_event": 0},
        {"step": 2, "num_reference_event": 2},
    ]
    assert set(sim.reference_table.table["idx_ref"]) == {2, 3}
    # The orphan C (ref 0) was evicted after the prune: step 2 never saw it.
    assert (c, 0) not in seen_pairs[1]
