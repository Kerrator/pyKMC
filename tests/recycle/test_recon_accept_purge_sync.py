"""Regression tests for recycle x purge desynchronisation (cluster 1).

A persistent active table carries recycled rows across KMC steps, but a
reconstruction failure purges reference events (``ReferenceEventTable.remove``)
without revalidating those carried-over active rows. Orphaned rows -- active
rows whose ``num_reference_event`` was deleted -- then crash the next step or
silently distort the events log.

These tests pin the four guarantees of the cluster-1 fix:

* ``ReferenceEventTable.remove`` reports the full set it deleted (forward refs
  *and* their backward siblings) so the caller can resync the active table.
* ``ActiveEventTable.drop_orphans`` evicts rows pointing at a purged ref.
* ``info_active_events`` tolerates an orphaned ref (forward or backward) instead
  of raising ``KeyError`` (#3 / #5).
* ``KMC.reconstruction`` only purges a reference event for a genuine
  reconstruction *defect* (wrong minimum) -- not for a per-row mapping mismatch
  or a transient engine hiccup (#2 / #9) -- and never raises ``IndexError`` when
  the failing row's ref is already gone (#1).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from pykmc import System
from pykmc.event_recycling import DistanceRecycling
from pykmc.event_table import ReferenceEventTable
from pykmc.info_simulation import info_active_events
from pykmc.kmc import KMC
from pykmc.result import Err, ErrorInfo, ErrorType

from .conftest import make_active_table, row


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _active_row(atom_index: int, num_reference_event: int) -> dict:
    """Build a stub active row pointing at a chosen reference event."""
    return {**row(atom_index), "num_reference_event": num_reference_event}


def _reference_table(rows: list[dict]) -> ReferenceEventTable:
    """Build a ReferenceEventTable whose ``table`` is a tiny stub frame."""
    config = Mock()
    config.rateconstant.style = "constant"
    config.rateconstant.T = 300.0
    config.rateconstant.k0 = 10.0
    config.control.reference_table = None
    rt = ReferenceEventTable(config)
    rt.table = pd.DataFrame(rows)
    return rt


def _ref_row(idx_ref: int, idx_backward: int, event_id: str) -> dict:
    """Build a minimal reference row with the columns the lookups consume."""
    return {
        "idx_ref": idx_ref,
        "event_id": event_id,
        "idx_backward": idx_backward,
        "energy_barrier": 0.5,
        "dra": 1.0,
    }


def _kmc_for_loop(active_ref_event: int, ref_rows: list[dict], err: Err) -> KMC:
    """Wire a KMC so ``reconstruction`` runs one failing iteration.

    ``_select_event`` and ``_reconstruction_active_event`` are stubbed; the
    single active row points at ``active_ref_event``; ``err`` is the failure the
    reconstruction returns.
    """
    kmc = KMC(config=Mock())
    kmc.loggers = Mock()
    kmc._close = Mock()
    kmc.reference_table = SimpleNamespace(table=pd.DataFrame(ref_rows))
    kmc._select_event = Mock(return_value=(0, 1.0, 2.0))
    kmc._reconstruction_active_event = Mock(return_value=err)
    return kmc


def _err(error_type: ErrorType) -> Err:
    """Wrap an ``ErrorType`` in the ``Err`` payload reconstruction returns."""
    return Err(ErrorInfo(type=error_type, message="stub", variables={}))


# --------------------------------------------------------------------------- #
# ReferenceEventTable.remove reports what it deleted (fwd + backward)
# --------------------------------------------------------------------------- #
def test_reference_remove_returns_removed_refs_including_backward() -> None:
    """remove() returns the requested refs together with their backward siblings."""
    rt = _reference_table(
        [
            _ref_row(0, 1, "topo0"),
            _ref_row(1, 0, "topo1"),
            _ref_row(2, 3, "topo2"),
            _ref_row(3, 2, "topo3"),
        ]
    )

    removed = rt.remove([0])

    # 0 plus its backward sibling 1 are deleted; 2/3 survive.
    assert removed == {0, 1}
    assert set(rt.table["idx_ref"]) == {2, 3}


def test_reference_remove_returns_empty_set_for_no_match() -> None:
    """remove() of an absent ref deletes nothing and reports an empty set."""
    rt = _reference_table([_ref_row(0, 0, "topo0")])
    assert rt.remove([99]) == set()


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
    # index must be a clean RangeIndex so .loc[i] stays positional downstream.
    assert list(active.table.index) == [0]


def test_drop_orphans_is_noop_for_empty_inputs() -> None:
    """drop_orphans() is a no-op for an empty ref set or an empty table."""
    active = make_active_table([_active_row(10, 0)])
    assert active.drop_orphans(set()) == 0
    assert len(active.table) == 1

    empty = make_active_table([_active_row(10, 0)])
    empty.table = empty.table.iloc[0:0]
    assert empty.drop_orphans({0}) == 0


# --------------------------------------------------------------------------- #
# info_active_events tolerates orphaned refs (#3 forward, #5 backward)
# --------------------------------------------------------------------------- #
def test_info_active_events_tolerates_purged_forward_ref() -> None:
    """An active row whose forward ref was purged must not raise KeyError (#3)."""
    active = make_active_table([_active_row(3, 7)])  # ref 7 absent
    reference_table = SimpleNamespace(table=pd.DataFrame([_ref_row(0, 0, "topo0")]))
    system_types = ["Ni"] * 8

    info = info_active_events(system_types, reference_table, active)

    assert len(info.reference_events) == 1
    assert info.reference_events[0] == 7


def test_info_active_events_tolerates_purged_backward_ref() -> None:
    """A surviving forward ref with a purged backward must not raise (#5)."""
    active = make_active_table([_active_row(3, 0)])
    reference_table = SimpleNamespace(
        table=pd.DataFrame([_ref_row(0, 1, "topo0")])  # idx_backward=1 missing
    )
    system_types = ["Ni"] * 8

    info = info_active_events(system_types, reference_table, active)

    # No KeyError on mapping_energy[backward]; dE_backward falls back to NaN.
    assert np.isnan(info.dE_backward[0])


# --------------------------------------------------------------------------- #
# KMC.reconstruction error classification (#1 / #2 / #9)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "per_row_error",
    [
        ErrorType.RECONSTRUCTION_MINIMIZE_FAILED,  # transient engine / bad columns (#2)
        ErrorType.RECONSTRUCTION_EVENT_NOT_CONTAINED,  # site-specific geometry (#9)
    ],
)
def test_per_row_failure_does_not_purge_reference(per_row_error: ErrorType) -> None:
    """A per-row/transient miss drops only the active row, never purges the ref."""
    kmc = _kmc_for_loop(
        active_ref_event=5,
        ref_rows=[_ref_row(5, 6, "topoX")],
        err=_err(per_row_error),
    )
    active = make_active_table([_active_row(7, 5)])

    _result, _dt, _ktot, _idx, err_reference, err_ae = kmc.reconstruction(active)

    assert err_reference == []
    assert err_ae == []
    assert len(active.table) == 0


@pytest.mark.parametrize(
    "defect_error",
    [ErrorType.RECONSTRUCTION_INVALID_MIN1, ErrorType.RECONSTRUCTION_INVALID_MIN2],
)
def test_min_defect_purges_reference(defect_error: ErrorType) -> None:
    """Landing on the wrong minimum schedules the reference event for purge."""
    kmc = _kmc_for_loop(
        active_ref_event=5,
        ref_rows=[_ref_row(5, 6, "topoX")],
        err=_err(defect_error),
    )
    active = make_active_table([_active_row(7, 5)])

    _result, _dt, _ktot, _idx, err_reference, err_ae = kmc.reconstruction(active)

    assert err_reference == [5]
    assert err_ae == ["topoX"]


def test_orphan_ref_failure_does_not_raise_indexerror() -> None:
    """A defect-type failure on a row whose ref is already gone must not crash (#1)."""
    kmc = _kmc_for_loop(
        active_ref_event=5,
        ref_rows=[_ref_row(0, 0, "topo0")],  # ref 5 absent
        err=_err(ErrorType.RECONSTRUCTION_INVALID_MIN1),
    )
    active = make_active_table([_active_row(7, 5)])

    # Must not raise IndexError on the empty .values[0] lookup.
    _result, _dt, _ktot, _idx, err_reference, err_ae = kmc.reconstruction(active)

    # Nothing left to purge; the row is simply dropped.
    assert err_reference == []
    assert err_ae == []
    assert len(active.table) == 0


# --------------------------------------------------------------------------- #
# Deferred drop_orphans: the eviction must run AFTER the positional consumers
# of idx_selected_event (step log, detector, prune), never in the purge block.
# These mirror the main-loop call order in KMC.run (purge -> log/prune ->
# drop_orphans) around the drop_orphans/prune_for_recycling helpers.
# --------------------------------------------------------------------------- #
def test_executed_row_survives_purge_of_shared_ref_then_evicted_after_prune(
    ni_fcc_3vacancies: tuple[System, list[int]],
) -> None:
    """Executed row sharing its ref with a purged sibling stays readable, then goes.

    An earlier active row failing INVALID_MIN1/MIN2 this step purges a reference
    event that the *executed* row (a different site) also points at -- routine,
    since one generic ref maps onto many sites. The fix defers ``drop_orphans``
    to after every positional consumer of ``idx_selected_event``, so:

    (i) the ``:=> table_line_info_kmc`` lookup ``active.table.loc[idx]`` still
        reads the executed row between purge and prune (no KeyError), and
    (ii) after prune + the deferred ``drop_orphans`` the next-step table holds no
         row pointing at a removed ref (the orphaned sibling never carries over).
    """
    system, central = ni_fcc_3vacancies
    positions_pre = system.positions.copy()
    # Execute the event at vacancy A (central[0]); shift its central atom so the
    # recycler's movement check fires on it (it is the executed row anyway).
    system.positions[central[0]] = positions_pre[central[0]] + np.array([0.3, 0.0, 0.0])

    # Executed row (A) shares ref 5 with the far/unmoved sibling row (C); B holds
    # a distinct ref 9. ref 5 is the one purged this step.
    active = make_active_table(
        [
            _active_row(central[0], 5),  # executed (idx 0)
            _active_row(central[1], 9),  # close sibling, distinct ref
            _active_row(central[2], 5),  # far/unmoved sibling, SHARED purged ref
        ]
    )
    active.recycler = DistanceRecycling(movement_thr=0.02, distance_thr=10.0)
    idx_selected_event = 0

    # -- purge (mirrors kmc.py: reference_table.remove -> removed_refs) --
    removed_refs = {5}

    # (i) purge is done; drop_orphans has NOT run yet. The step-log lookup must
    # still resolve the executed row at its captured positional index.
    executed = active.table.loc[idx_selected_event]
    assert int(executed.at["atom_index"]) == central[0]
    assert int(executed.at["num_reference_event"]) == 5

    # -- prune for the next step (recycler keeps the far/unmoved C, ref 5) --
    active.prune_for_recycling(idx_selected_event, system, positions_pre)
    assert 5 in set(active.table["num_reference_event"].astype(int))  # sibling carried

    # -- deferred eviction (mirrors the new call site after the prune block) --
    n_evicted = active.drop_orphans(removed_refs)

    # (ii) the orphaned sibling is gone; no next-step row points at a removed ref.
    assert n_evicted == 1
    assert removed_refs.isdisjoint(set(active.table["num_reference_event"].astype(int)))


def test_orphan_below_executed_index_does_not_shift_label_before_prune() -> None:
    """An orphan at a position below idx_selected_event must not shift its label early.

    If ``drop_orphans`` ran in the purge block it would ``reset_index`` and slide
    every label above the evicted orphan down by one, so the POSITIONAL
    ``idx_selected_event`` captured from ``reconstruction`` would resolve to the
    wrong row for the step log / basin detect / recycling anchor. Deferring the
    eviction to after those consumers keeps the executed row's label stable.
    """
    # Orphan (ref 5) at label 0, executed row (ref 9) at label 1.
    active = make_active_table(
        [
            _active_row(10, 5),  # orphan -- ref purged this step
            _active_row(11, 9),  # executed row
        ]
    )
    idx_selected_event = 1
    removed_refs = {5}

    # With the eviction deferred, the executed row is still at its captured label.
    executed = active.table.loc[idx_selected_event]
    assert int(executed.at["atom_index"]) == 11
    assert int(executed.at["num_reference_event"]) == 9

    # Guard the premise: an *early* drop_orphans would evict label 0 and
    # reset_index, sliding the executed row down to label 0 -- so the captured
    # label 1 no longer exists and the step-log lookup raises KeyError. This is
    # exactly why the real loop must defer the eviction. (Operate on a copy so
    # `active` stays intact.)
    early = make_active_table(
        [_active_row(10, 5), _active_row(11, 9)]
    )
    early.drop_orphans(removed_refs)
    assert int(early.table.loc[0].at["atom_index"]) == 11  # executed row slid down
    with pytest.raises(KeyError):
        early.table.loc[idx_selected_event]

    # Deferred path: eviction only after the log/prune stage leaves the executed
    # row correctly identified and drops the orphan from the next-step table.
    active.drop_orphans(removed_refs)
    assert removed_refs.isdisjoint(set(active.table["num_reference_event"].astype(int)))
