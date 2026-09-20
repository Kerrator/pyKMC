"""Batch-order identity oracle; geometric/typed-protocol evidence only.

Imports the frozen early-admission fixture by a registered module name. Actual
table admission, service dispatch, archive and finalizers run; no native MEP or
numerical Hessian is represented by the literal protocol spectra.
"""

from concurrent.futures import Future
from dataclasses import replace

import numpy as np
import pytest

from pykmc.event_table import ReferenceEventTable
from pykmc.htst.provenance import RequestSnapshot
from pykmc.rate_constant import create_rate_constant
from pykmc.rate_constant.prefactors import PrefactorService


from tests.lifecycle import test_whole_event_admission as base


class Worker:
    def __init__(self, frequencies):
        self.frequencies = frequencies
        self.requests = []
        self.results = []

    def submit(self, operation, *, request, compute_backward=True):
        assert operation == "compute_event_prefactors" and compute_backward is True
        frequency = self.frequencies[len(self.requests)]
        self.requests.append(request)
        result = base.protocol_result(request)
        result = replace(
            result,
            forward=replace(result.forward, nu0_hz=frequency),
            backward=replace(result.backward, nu0_hz=frequency),
        )
        self.results.append(result)
        future = Future()
        future.set_result(result)
        return future


def table_and_worker(monkeypatch, frequencies):
    cfg = base.configuration("htst")
    worker = Worker(frequencies)
    service = PrefactorService(
        cfg,
        worker,
        create_rate_constant(cfg.rateconstant),
        species_masses=(base.TYPES, base.MASSES),
        global_constraints=base.full_constraints(base.I),
        method="protocol",
    )
    table = ReferenceEventTable(cfg, prefactor_service=service)

    def build_series(
        *,
        min1_positions,
        saddle_positions,
        min2_positions,
        index_move,
        dE_forward,
        dE_backward,
        cell,
        types,
        pbc=None,
    ):
        assert tuple(pbc) == base.PBC
        forward = np.array_equal(min1_positions, base.I)
        expected = (base.I, base.F) if forward else (base.F, base.I)
        assert np.array_equal(min1_positions, expected[0])
        assert np.array_equal(min2_positions, expected[1])
        assert np.array_equal(saddle_positions, base.S)
        assert np.array_equal(cell, base.CELL) and tuple(types) == base.TYPES
        assert index_move == 0
        initial_id, final_id = ("A", "B") if forward else ("B", "A")
        return (
            base.row(
                min1_positions,
                saddle_positions,
                min2_positions,
                initial_id=initial_id,
                final_id=final_id,
                barrier=dE_forward,
            ),
            base.row(
                min2_positions,
                saddle_positions,
                min1_positions,
                initial_id=final_id,
                final_id=initial_id,
                barrier=dE_backward,
            ),
        )

    monkeypatch.setattr(table, "_build_event_series", build_series)
    original_resolve = table._resolve_prefactors

    def with_unrelated_labels(*args, **kwargs):
        # Exercise logical IDs independently of positions or DataFrame labels.
        table.table.index = [103 + 11 * i for i in range(len(table.table))]
        return original_resolve(*args, **kwargs)

    monkeypatch.setattr(table, "_resolve_prefactors", with_unrelated_labels)
    return table, worker


def make_events(reverse_second):
    first = base.event(base.I, base.F)
    if reverse_second:
        second = base.event(base.F, base.I)
        second.dE_forward, second.dE_backward = 0.7, 0.5
    else:
        second = base.event(base.I, base.F)
    return [first, second]


def input_copy(events):
    return [
        tuple(
            np.array(p, copy=True)
            for p in (
                ev.min1_positions,
                ev.saddle_positions,
                ev.min2_positions,
                *ev.prefactor_geometry,
            )
        )
        for ev in events
    ]


def physical_records(table):
    return sorted(
        (
            str(row.event_id),
            str(row.id_final),
            float(row.energy_barrier),
            float(row.nu0),
            float(row.k),
        )
        for _, row in table.table.iterrows()
    )


def audit(table, worker, events, snapshots):
    for ev, before in zip(events, snapshots, strict=True):
        for actual, expected in zip(
            (
                ev.min1_positions,
                ev.saddle_positions,
                ev.min2_positions,
                *ev.prefactor_geometry,
            ),
            before,
            strict=True,
        ):
            assert np.array_equal(actual, expected)
        assert ev.constraints.atom_ids == base.IDS
        assert ev.constraints.fixed_ids == ()
    ids = set(table.table.idx_ref.astype(int))
    assert set(table.table.idx_backward.astype(int)) <= ids
    for _, row in table.table.iterrows():
        calc = table.prefactor_archive.calculation_for(int(row.idx_ref), row)
        assert calc is not None and calc.estimate.ok
        assert calc.provenance.source.is_complete
    for request, result in zip(worker.requests, worker.results, strict=True):
        assert result.provenance.source == RequestSnapshot.capture(request)
        for direction in ("forward", "backward"):
            calculation = result.calculation(direction)
            assert (
                table.prefactor_archive.calculations[calculation.calculation_id]
                == calculation
            )
    archived_ids = set(table.prefactor_archive.references)
    assert table.max_idx_ref() > max(ids | archived_ids)
    for discarded in archived_ids - ids:
        assert table.prefactor_archive.history.get(discarded)
        assert table.prefactor_archive.references[discarded] is not None


@pytest.mark.parametrize(
    "reverse_second", [False, True], ids=["same-direction", "exact-reverse"]
)
def test_same_batch_and_sequential_have_one_physical_pair(reverse_second, monkeypatch):
    batch, batch_worker = table_and_worker(monkeypatch, [5e12, 5e12])
    batch_events = make_events(reverse_second)
    batch_before = input_copy(batch_events)
    batch_results = batch.add_events(batch_events, pbc=base.PBC)
    sequential, sequential_worker = table_and_worker(monkeypatch, [5e12])
    sequential_events = make_events(reverse_second)
    sequential_before = input_copy(sequential_events)
    for ev in sequential_events:
        sequential.add_events([ev], pbc=base.PBC)
    audit(batch, batch_worker, batch_events, batch_before)
    audit(sequential, sequential_worker, sequential_events, sequential_before)
    # contracts 7f policy 6: the geometric gate rejects the exact repeat (or
    # its exact reverse) before dispatch, in the same batch as well.
    assert [r.is_ok() for r in batch_results] == [True, False]
    assert len(batch_worker.requests) == 1
    assert len(sequential_worker.requests) == 1  # Known exact repeat adds no work.
    assert len(sequential.table) == 2
    assert len(batch.table) == 2, "The same batch retained a second exact physical pair"
    assert physical_records(batch) == physical_records(sequential)
    assert sum(batch.table.k.astype(float)) == sum(sequential.table.k.astype(float))
    for table in (batch, sequential):
        links = {int(r.idx_ref): int(r.idx_backward) for _, r in table.table.iterrows()}
        assert all(
            links.get(back) == forward and forward != back
            for forward, back in links.items()
        )


def test_same_batch_disagreeing_actual_spectra_are_not_merged(monkeypatch):
    table, worker = table_and_worker(monkeypatch, [5e12, 7e12])
    events = make_events(False)
    before = input_copy(events)
    results = table.add_events(events, pbc=base.PBC)
    audit(table, worker, events, before)
    # contracts 7f policy 6: the exact repeat never reaches the worker, so no
    # second spectrum exists to disagree with; the first pair stands alone.
    assert [r.is_ok() for r in results] == [True, False]
    assert len(worker.requests) == 1
    assert len(table.table) == 2
    assert sorted(table.table.nu0.astype(float)) == [5e12, 5e12]
    links = {int(r.idx_ref): int(r.idx_backward) for _, r in table.table.iterrows()}
    assert all(
        links.get(back) == forward and forward != back
        for forward, back in links.items()
    )
