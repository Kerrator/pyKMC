"""Reference-table prefactor lifecycle (architecture rule 1).

Accepted events only cost a Hessian; one request per accepted event yields
both directions; rows are patched by logical id (never by position); the
batch is resolved before ``add_events`` returns. No LAMMPS, no MPI.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pykmc.event_table import (
    REFERENCE_BASE_COLUMNS,
    REFERENCE_HTST_COLUMNS,
    ReferenceEventTable,
)
from pykmc.rate_constant import create_rate_constant, rate_from_prefactor
from pykmc.rate_constant.prefactors import PrefactorService
from pykmc.result import EventSearchOutput
from tests.lifecycle.conftest import (
    FakeManager,
    accepted,
    rejected,
)

from .protocol_producers import protocol_event_prefactors, protocol_patch

HOP = np.array([1.2, 0.3, 0.0])


def _event(
    system: Any,
    move: int,
    dE_forward: float,
    dE_backward: float,
    shift: np.ndarray | None = None,
    types: list[str] | None = "system",
) -> EventSearchOutput:
    """Build a search output on ``system``; ``shift`` None means trivial."""
    pos = np.asarray(system.positions, dtype=float)
    min2 = pos.copy()
    saddle = pos.copy()
    if shift is not None:
        min2[move] += shift
        saddle[move] += 0.5 * shift
    return EventSearchOutput(
        central_atom_index=move,
        min1_positions=pos.copy(),
        saddle_positions=saddle,
        min2_positions=min2,
        dE_forward=dE_forward,
        dE_backward=dE_backward,
        move_atom_index=move,
        cell=np.asarray(system.cell, dtype=float),
        types=list(system.types) if types == "system" else types,
    )


def _table_with_service(
    config: Any, responses: dict[tuple, tuple]
) -> tuple[ReferenceEventTable, FakeManager]:
    """Build a reference table wired to a fake manager answering by event key."""

    def responder(req: Any) -> Any:
        fwd, bwd = responses[req.event_key]
        return protocol_event_prefactors(req, fwd, bwd)

    fake = FakeManager(responder)
    service = PrefactorService(
        config, fake, create_rate_constant(config.rateconstant), method="fd"
    )
    return ReferenceEventTable(config, prefactor_service=service), fake


def _row(table: ReferenceEventTable, idx_ref: int) -> pd.Series:
    rows = table.table[table.table["idx_ref"] == idx_ref]
    assert len(rows) == 1, idx_ref
    return rows.iloc[0]


def _insert(table: ReferenceEventTable, row: pd.Series, idx_ref: int) -> None:
    row = row.copy()
    row["idx_ref"] = idx_ref
    row["idx_backward"] = idx_ref
    table.table = pd.concat([table.table, row.to_frame().T], ignore_index=True)


class TestSchema:
    """Constant tables keep the S0 schema; htst/rpa append exactly four columns."""

    def test_constant_schema_is_the_baseline(self, constant_config: Any) -> None:
        """Exactly the 15 baseline columns, in order."""
        table = ReferenceEventTable(constant_config)
        assert list(table.table.columns) == list(REFERENCE_BASE_COLUMNS)
        assert len(REFERENCE_BASE_COLUMNS) == 15

    @pytest.mark.parametrize("style", ["htst", "rpa"])
    def test_htst_schema_is_baseline_plus_four(
        self, style: str, htst_config: Any, rpa_config: Any
    ) -> None:
        """Baseline 15 plus k_prefactor, nu0, nu0_status, nu0_reason."""
        config = htst_config if style == "htst" else rpa_config
        table = ReferenceEventTable(config)
        assert list(table.table.columns) == list(REFERENCE_BASE_COLUMNS) + list(
            REFERENCE_HTST_COLUMNS
        )
        assert REFERENCE_HTST_COLUMNS == (
            "k_prefactor",
            "nu0",
            "nu0_status",
            "nu0_reason",
        )

    def test_built_series_carry_pending_placeholders(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """Before resolution a row holds k0, NaN nu0 and status pending."""
        table = ReferenceEventTable(htst_config)
        pos = system_single_type_fcc.positions
        fwd, bwd = table._build_event_series(
            min1_positions=pos,
            saddle_positions=pos,
            min2_positions=pos,
            index_move=0,
            dE_forward=0.5,
            dE_backward=0.7,
            cell=system_single_type_fcc.cell,
            types=list(system_single_type_fcc.types),
        )
        k0 = htst_config.rateconstant.k0
        T = htst_config.rateconstant.T
        for series, dE in ((fwd, 0.5), (bwd, 0.7)):
            assert series["k_prefactor"] == k0
            assert math.isnan(series["nu0"])
            assert series["nu0_status"] == "pending"
            assert series["nu0_reason"] == ""
            assert series["k"] == rate_from_prefactor(k0, dE, T)


class TestTwoEventsDistinctDirections:
    """Two events, four directional rows, four distinct outcomes."""

    def test_forward_and_backward_rows_get_their_own_estimate(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """Each direction's row is patched from its own DirectionalPrefactor."""
        responses = {
            (0, 1): (accepted(5.0e12), accepted(3.0e12)),
            (2, 3): (accepted(7.0e12), rejected("backward window")),
        }
        table, fake = _table_with_service(htst_config, responses)
        sys_ = system_single_type_fcc
        results = table.add_events(
            [_event(sys_, 0, 2.0, 1.5, HOP), _event(sys_, 5, 3.0, 2.5, HOP)],
            pbc=sys_.pbc,
        )
        assert [r.is_ok() for r in results] == [True, True]
        T = htst_config.rateconstant.T
        k0 = htst_config.rateconstant.k0

        r0, r1, r2, r3 = (_row(table, i) for i in range(4))
        assert (r0["nu0"], r0["k_prefactor"], r0["nu0_status"]) == (5.0e12, 5.0, "ok")
        assert r0["k"] == rate_from_prefactor(5.0, 2.0, T)
        assert r0["nu0_reason"] == ""
        assert (r1["nu0"], r1["k_prefactor"], r1["nu0_status"]) == (3.0e12, 3.0, "ok")
        assert r1["k"] == rate_from_prefactor(3.0, 1.5, T)
        assert (r2["nu0"], r2["k_prefactor"], r2["nu0_status"]) == (7.0e12, 7.0, "ok")
        assert r2["k"] == rate_from_prefactor(7.0, 3.0, T)
        assert math.isnan(r3["nu0"])
        assert r3["k_prefactor"] == k0
        assert r3["nu0_status"] == "rejected"
        assert r3["nu0_reason"] == "out_of_window: backward window"
        assert r3["k"] == rate_from_prefactor(k0, 2.5, T)
        # reciprocal links untouched by the patch
        assert [int(_row(table, i)["idx_backward"]) for i in range(4)] == [1, 0, 3, 2]

    def test_one_request_per_accepted_event_with_full_geometry(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """The request carries the full search geometry, not the table crops."""
        responses = {(0, 1): (accepted(5.0e12), accepted(3.0e12))}
        table, fake = _table_with_service(htst_config, responses)
        sys_ = system_single_type_fcc
        ev = _event(sys_, 0, 2.0, 1.5, HOP)
        table.add_events([ev], pbc=sys_.pbc)

        requests = fake.prefactor_requests
        assert len(requests) == 1
        req = requests[0]
        assert req.event_key == (0, 1)
        n = len(sys_.positions)
        assert req.min1_positions.shape == (n, 3)
        assert np.array_equal(req.min1_positions, ev.min1_positions)
        assert np.array_equal(req.saddle_positions, ev.saddle_positions)
        assert np.array_equal(req.min2_positions, ev.min2_positions)
        assert req.center_index == 0
        assert req.types == tuple(sys_.types)
        assert req.pbc == tuple(bool(p) for p in sys_.pbc)
        assert np.array_equal(req.cell, sys_.cell)

    def test_mixed_direction_forward_rejected(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """A forward rejection never clears the accepted backward value."""
        responses = {(0, 1): (rejected("fwd"), accepted(4.0e12))}
        table, _ = _table_with_service(htst_config, responses)
        sys_ = system_single_type_fcc
        table.add_events([_event(sys_, 0, 2.0, 1.5, HOP)], pbc=sys_.pbc)
        r0, r1 = _row(table, 0), _row(table, 1)
        assert r0["nu0_status"] == "rejected" and r0["k_prefactor"] == 1.0
        assert r1["nu0_status"] == "ok" and r1["k_prefactor"] == 4.0


class TestOnlyAcceptedEventsCostRequests:
    """Rejected and duplicate events never reach the manager."""

    def test_energy_rejected_and_duplicate_cost_nothing(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """Energy rejection is free; coarse barrier candidates need physical proof."""
        responses = {
            (0, 1): (accepted(5.0e12), accepted(3.0e12)),
            (2, 3): (accepted(5.0e12), accepted(3.0e12)),
        }
        table, fake = _table_with_service(htst_config, responses)
        sys_ = system_single_type_fcc
        emax = htst_config.eventsearch.emax_event
        results = table.add_events(
            [
                _event(sys_, 0, 2.0, 1.5, HOP),
                _event(sys_, 0, emax + 1.0, 1.5, HOP),  # energy gate
                _event(
                    sys_, 0, 2.05, 1.5, HOP
                ),  # same coarse topology; 0.05 eV is not numerical equality
            ],
            pbc=sys_.pbc,
        )
        assert [r.is_ok() for r in results] == [True, False, True]
        assert len(fake.prefactor_requests) == 2
        assert len(table.table) == 3
        assert list(table.table.energy_barrier) == [2.0, 1.5, 2.05]
        assert list(
            zip(table.table.idx_ref, table.table.idx_backward, strict=True)
        ) == [(0, 1), (1, 0), (2, 1)]
        # Directional equality permits the shared reverse. The two different
        # forward barriers/rates remain distinct; this protocol has no actual U.
        for idx, nu, barrier in ((0, 5.0, 2.0), (1, 3.0, 1.5), (2, 5.0, 2.05)):
            row = _row(table, idx)
            assert row.nu0 == nu * 1e12 and row.k_prefactor == nu
            assert row.energy_barrier == barrier
            assert row.k == rate_from_prefactor(nu, barrier, htst_config.rateconstant.T)
        assert _row(table, 0).k > _row(table, 2).k
        assert table.prefactor_archive.history[3][-1]["nu0"] == 3.0e12
        # A truly identical, already-resolved whole event still costs no work.
        duplicate = table.add_events([_event(sys_, 0, 2.0, 1.5, HOP)], pbc=sys_.pbc)
        assert not duplicate[0].is_ok()
        assert len(fake.prefactor_requests) == 2
        assert len(table.table) == 3

    def test_no_accepted_event_submits_nothing(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """An all-rejected batch never builds a request."""
        table, fake = _table_with_service(htst_config, {})
        sys_ = system_single_type_fcc
        emax = htst_config.eventsearch.emax_event
        table.add_events([_event(sys_, 0, emax + 1.0, 1.5, HOP)], pbc=sys_.pbc)
        assert fake.prefactor_requests == []


class TestResolutionCompletesInsideAddEvents:
    """No row is left pending once add_events returns (refinement reads k next)."""

    def test_no_pending_rows_after_add_events(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """Every accepted row is ok or rejected, and the summary counts them."""
        responses = {(0, 1): (accepted(5.0e12), rejected("bwd"))}
        table, _ = _table_with_service(htst_config, responses)
        sys_ = system_single_type_fcc
        table.add_events([_event(sys_, 0, 2.0, 1.5, HOP)], pbc=sys_.pbc)
        assert set(table.table["nu0_status"]) == {"ok", "rejected"}
        assert table.prefactor_summary() == {
            "ok": 1,
            "rejected": 1,
            "pending": 0,
            "legacy": 0,
            "stale": 0,
        }

    def test_reference_estimate_seeds_refinement(
        self, htst_config: Any, constant_config: Any, system_single_type_fcc: Any
    ) -> None:
        """reference_estimate exposes the stored estimate by logical id."""
        responses = {(0, 1): (accepted(5.0e12), rejected("bwd"))}
        table, _ = _table_with_service(htst_config, responses)
        sys_ = system_single_type_fcc
        table.add_events([_event(sys_, 0, 2.0, 1.5, HOP)], pbc=sys_.pbc)
        assert table.reference_estimate(0) == {
            "nu0_hz": 5.0e12,
            "nu0_status": "ok",
            "nu0_reason": "",
            "nu0_source": "reference",
        }
        assert table.reference_estimate(1)["nu0_hz"] is None
        assert table.reference_estimate(1)["nu0_status"] == "rejected"
        with pytest.raises(ValueError, match="not in the reference table"):
            table.reference_estimate(42)
        assert ReferenceEventTable(constant_config).reference_estimate(0) == {}


class TestPatchByLogicalId:
    """Rows are located by ``idx_ref``; positional indexing would fail."""

    def test_patch_on_shuffled_sparse_table(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """Patching id 3 touches the row at position 1 and nothing else."""
        table = ReferenceEventTable(htst_config)
        pos = system_single_type_fcc.positions

        def series(dE: float) -> pd.Series:
            fwd, _ = table._build_event_series(
                min1_positions=pos,
                saddle_positions=pos,
                min2_positions=pos,
                index_move=0,
                dE_forward=dE,
                dE_backward=dE,
                cell=system_single_type_fcc.cell,
                types=list(system_single_type_fcc.types),
            )
            return fwd

        _insert(table, series(0.5), 12)
        _insert(table, series(1.0), 3)
        _insert(table, series(1.6), 7)
        assert list(table.table["idx_ref"]) == [12, 3, 7]

        table._protocol_source = system_single_type_fcc
        protocol_patch(table, 3, accepted(4.0e12))

        patched = table.table.iloc[1]
        assert int(patched["idx_ref"]) == 3
        assert patched["nu0"] == 4.0e12 and patched["k_prefactor"] == 4.0
        assert patched["k"] == rate_from_prefactor(4.0, 1.0, htst_config.rateconstant.T)
        for position in (0, 2):
            untouched = table.table.iloc[position]
            assert untouched["nu0_status"] == "pending"
            assert untouched["k_prefactor"] == htst_config.rateconstant.k0
        with pytest.raises(ValueError, match="not in the reference table"):
            table._patch_row(99, accepted(1e12))


class TestSelfReverseThroughAddEvents:
    """Self-reverse collapse follows accepted directional results and physical proof."""

    def test_agreeing_spectra_leave_the_reason_empty(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """A trivial self-reverse event is one self-linked row with the forward value."""
        responses = {(0, 1): (accepted(5.0e12), accepted(5.02e12))}
        table, fake = _table_with_service(htst_config, responses)
        sys_ = system_single_type_fcc
        table.add_events([_event(sys_, 0, 0.5, 0.5)], pbc=sys_.pbc)
        assert len(fake.prefactor_requests) == 1
        assert fake.prefactor_requests[0].event_key == (0, 1)
        assert list(
            zip(table.table["idx_ref"], table.table["idx_backward"], strict=True)
        ) == [(0, 0)]
        assert _row(table, 0)["nu0"] == 5.0e12
        assert _row(table, 0)["nu0_reason"] == ""
        assert table.max_idx_ref() == 2

    def test_unequal_spectra_keep_one_row_with_the_backward_recorded(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """Equal topologies with different spectra retain both accepted directions."""
        responses = {(0, 1): (accepted(5.0e12), accepted(3.0e12))}
        table, _ = _table_with_service(htst_config, responses)
        sys_ = system_single_type_fcc
        table.add_events([_event(sys_, 0, 0.5, 0.5)], pbc=sys_.pbc)
        links = list(
            zip(table.table["idx_ref"], table.table["idx_backward"], strict=True)
        )
        assert [(int(a), int(b)) for a, b in links] == [(0, 1), (1, 0)]
        row = _row(table, 0)
        assert row["k_prefactor"] == 5.0 and row["nu0_status"] == "ok"
        assert row["nu0_reason"] == (
            "self-reverse unproven: backward nu0 = 3.0000e+12 Hz, differs by 40.0%"
        )
        backward = _row(table, 1)
        assert backward["nu0"] == 3.0e12 and backward["k_prefactor"] == 3.0
        assert backward["nu0_status"] == "ok" and backward["nu0_reason"] == ""
        assert table.prefactor_summary() == {
            "ok": 2,
            "rejected": 0,
            "pending": 0,
            "legacy": 0,
            "stale": 0,
        }


class TestMisconfiguration:
    """Incomplete htst wiring fails loudly instead of degrading to k0."""

    def test_missing_service_raises(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """An htst table without a service cannot accept events."""
        table = ReferenceEventTable(htst_config)
        sys_ = system_single_type_fcc
        with pytest.raises(RuntimeError, match="PrefactorService"):
            table.add_events([_event(sys_, 0, 2.0, 1.5, HOP)], pbc=sys_.pbc)

    def test_missing_pbc_raises(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """The request needs the actual pbc; None is not defaulted."""
        table, _ = _table_with_service(htst_config, {})
        sys_ = system_single_type_fcc
        with pytest.raises(RuntimeError, match="pbc"):
            table.add_events([_event(sys_, 0, 2.0, 1.5, HOP)])

    def test_missing_types_raises(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """A search result without types cannot become a request."""
        table, _ = _table_with_service(htst_config, {})
        sys_ = system_single_type_fcc
        with pytest.raises(RuntimeError, match="types"):
            table.add_events([_event(sys_, 0, 2.0, 1.5, HOP, types=None)], pbc=sys_.pbc)

    def test_constant_table_ignores_pbc_and_never_submits(
        self, constant_config: Any, system_single_type_fcc: Any
    ) -> None:
        """Constant mode accepts events with no service and no request."""
        table = ReferenceEventTable(constant_config)
        sys_ = system_single_type_fcc
        results = table.add_events([_event(sys_, 0, 2.0, 1.5, HOP)], pbc=sys_.pbc)
        assert results[0].is_ok()
        assert list(table.table.columns) == list(REFERENCE_BASE_COLUMNS)
        assert table.prefactor_summary() == {}
