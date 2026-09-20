"""Two bounded batch-merge regressions; typed protocol, not native spectra.

Use real public add_events, service requests, producing records and finalizers.
Literal directional frequencies intentionally isolate association and tolerance
logic. No production import or execution was performed while authoring.
"""

from concurrent.futures import Future
from dataclasses import replace
import math

import numpy as np

from pykmc.result import EventSearchOutput


from tests.lifecycle import test_whole_event_batch as batch

base = batch.base


class DirectionalWorker:
    def __init__(self, frequencies):
        self.frequencies = frequencies
        self.requests = []
        self.results = []

    def submit(self, operation, *, request, compute_backward=True):
        assert operation == "compute_event_prefactors" and compute_backward is True
        forward, backward = self.frequencies[len(self.requests)]
        self.requests.append(request)
        result = base.protocol_result(request)
        result = replace(
            result,
            forward=replace(result.forward, nu0_hz=forward),
            backward=replace(result.backward, nu0_hz=backward),
        )
        self.results.append(result)
        future = Future()
        future.set_result(result)
        return future


def table_and_worker(monkeypatch, frequencies):
    table, _ = batch.table_and_worker(monkeypatch, [])
    worker = DirectionalWorker(frequencies)
    table.prefactor_service.manager = worker
    return table, worker


def links_and_values(table):
    return {
        int(row.idx_ref): (int(row.idx_backward), float(row.nu0))
        for _, row in table.table.iterrows()
    }


def test_two_direction_matches_must_target_one_existing_pair(monkeypatch):
    # A and B remain different accepted directional pairs. C's individual
    # matches would select A.forward and B.backward, never a coherent pair.
    table, worker = table_and_worker(
        monkeypatch,
        [(5e12, 7e12), (7e12, 5e12), (5e12, 5e12)],
    )
    events = [base.event(base.I, base.F) for _ in range(3)]
    before = batch.input_copy(events)
    results = table.add_events(events, pbc=base.PBC)
    batch.audit(table, worker, events, before)
    print("cross_pair_rows", links_and_values(table))
    assert len(worker.requests) == 3
    assert len(table.table) == 6, (
        "Independent matches spliced two older directional pairs"
    )
    assert all(result.is_ok() and len(result.ok_value()) == 2 for result in results)
    assert links_and_values(table) == {
        0: (1, 5e12),
        1: (0, 7e12),
        2: (3, 7e12),
        3: (2, 5e12),
        4: (5, 5e12),
        5: (4, 5e12),
    }
    # The nonsymmetric full labeled saddle fixes R=I, so opposite orientation
    # cannot be used to manufacture a third older matching pair.
    assert math.isclose(
        np.linalg.det(base.S[1:] - base.S[0]), 3.0, rel_tol=0.0, abs_tol=1e-12
    )


def test_shared_self_reverse_survivor_needs_direct_spectral_agreement(monkeypatch):
    # Reflection x -> -x is one exact physical map for all three full frames.
    # Distinct species fix atom ordering; no geometry threshold is adjusted.
    initial = np.array(
        [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    )
    saddle = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.6, 0.7], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    )
    reflection = np.diag([-1.0, 1.0, 1.0])
    final = initial @ reflection.T
    assert np.array_equal(final @ reflection.T, initial)
    assert np.array_equal(saddle @ reflection.T, saddle)
    assert math.isclose(4.8e12, 5e12, rel_tol=0.05, abs_tol=0.0)
    assert math.isclose(5.2e12, 5e12, rel_tol=0.05, abs_tol=0.0)
    assert not math.isclose(4.8e12, 5.2e12, rel_tol=0.05, abs_tol=0.0)
    table, worker = table_and_worker(monkeypatch, [(5e12, 5e12), (4.8e12, 5.2e12)])

    def symmetric_series(
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
        assert np.array_equal(min1_positions, initial)
        assert np.array_equal(saddle_positions, saddle)
        assert np.array_equal(min2_positions, final)
        assert np.array_equal(cell, base.CELL) and tuple(types) == base.TYPES
        assert index_move == 0 and dE_forward == dE_backward == 0.5
        return (
            base.row(initial, saddle, final, initial_id="A", final_id="A", barrier=0.5),
            base.row(final, saddle, initial, initial_id="A", final_id="A", barrier=0.5),
        )

    monkeypatch.setattr(table, "_build_event_series", symmetric_series)
    events = [
        EventSearchOutput(
            central_atom_index=0,
            min1_positions=initial.copy(),
            saddle_positions=saddle.copy(),
            min2_positions=final.copy(),
            dE_forward=0.5,
            dE_backward=0.5,
            move_atom_index=0,
            cell=base.CELL.copy(),
            types=list(base.TYPES),
            constraints=base.full_constraints(initial),
            prefactor_geometry=(initial.copy(), saddle.copy(), final.copy()),
        )
        for _ in range(2)
    ]
    before = batch.input_copy(events)
    results = table.add_events(events, pbc=base.PBC)
    batch.audit(table, worker, events, before)
    print("nontransitive_rows", links_and_values(table))
    assert len(worker.requests) == 2
    # The first exact positive must collapse; the disagreeing actual pair
    # remains reciprocal even though both estimates are close to old 5 THz.
    assert len(table.table) == 3, (
        "Two approximate matches bypassed direct directional disagreement"
    )
    assert all(result.is_ok() for result in results)
    assert [len(result.ok_value()) for result in results] == [1, 2]
    assert links_and_values(table) == {0: (0, 5e12), 2: (3, 4.8e12), 3: (2, 5.2e12)}
