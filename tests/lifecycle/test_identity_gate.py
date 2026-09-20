"""Directional identity gate of the reference catalogue.

Constant mode retains its topology-based representation. HTST/RPA use the
same geometric admission gate (contracts 7f policy 6), retain provisional
reciprocal rows until accepted directional calculations and a single
whole-event physical map authorize a merge, and collapse an unproven
same-topology candidate back to one self-linked forward row (two rows sharing
one ``event_id`` would be applied twice per site). The declared worker
frequencies in these protocol tests are synthetic; geometry, source context,
logical links and lifecycle actions exercise the actual catalogue/service.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pykmc.event_table import (
    SELF_REVERSE_NU0_RTOL,
    ReferenceEventTable,
    self_reverse_prefactors_agree,
)

# Contract 7c (Diagnostics): every [htst] reference line ends with the free-atom
# count and the wall time of the batch that produced it.
from pykmc.htst.free_region import common_free_indices
from pykmc.rate_constant import create_rate_constant, rate_from_prefactor
from pykmc.rate_constant.prefactors import PrefactorService
from pykmc.result import ErrorType, EventSearchOutput
from tests.lifecycle.conftest import (
    FakeManager,
    accepted,
    rejected,
    skipped,
)

from .protocol_producers import archived_frequency, protocol_event_prefactors

_BATCH_SUFFIX = re.compile(r"\(n_free (\d+), batch (\d+\.\d{3}) s\)$")


def _series(
    table: ReferenceEventTable,
    system: Any,
    move: int,
    dE_forward: float,
    dE_backward: float,
    shift: np.ndarray | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Build the directional series of a (possibly trivial) event on ``system``.

    With ``shift`` None the event is trivial (min1 == saddle == min2), so both
    endpoint topologies and both saddle crops coincide: the base collapses it.
    """
    pos = np.asarray(system.positions, dtype=float)
    min2 = pos.copy()
    saddle = pos.copy()
    if shift is not None:
        min2[move] += shift
        saddle[move] += 0.5 * np.asarray(shift)
    return table._build_event_series(
        min1_positions=pos,
        saddle_positions=saddle,
        min2_positions=min2,
        index_move=move,
        dE_forward=dE_forward,
        dE_backward=dE_backward,
        cell=system.cell,
        types=list(system.types),
    )


def _insert(
    table: ReferenceEventTable, row: pd.Series, idx_ref: int, idx_backward: int
) -> None:
    """Append ``row`` with explicit logical ids (sparse catalogues in tests)."""
    row = row.copy()
    row["idx_ref"] = idx_ref
    row["idx_backward"] = idx_backward
    table.table = pd.concat([table.table, row.to_frame().T], ignore_index=True)


def _links(table: ReferenceEventTable) -> list[tuple[int, int]]:
    """Return ``(idx_ref, idx_backward)`` per row in table order."""
    return [
        (int(r), int(b))
        for r, b in zip(
            table.table["idx_ref"], table.table["idx_backward"], strict=True
        )
    ]


def _trivial_event(system: Any, move: int = 0, dE: float = 0.5) -> EventSearchOutput:
    """Search output of a trivial (self-reverse) event on ``system``."""
    pos = np.asarray(system.positions, dtype=float)
    return EventSearchOutput(
        central_atom_index=move,
        min1_positions=pos.copy(),
        saddle_positions=pos.copy(),
        min2_positions=pos.copy(),
        dE_forward=dE,
        dE_backward=dE,
        move_atom_index=move,
        cell=np.asarray(system.cell, dtype=float),
        types=list(system.types),
    )


def _hop_event(
    system: Any, move: int, dE_forward: float, dE_backward: float, shift: np.ndarray
) -> EventSearchOutput:
    """Search output of the hop that ``_series`` builds with the same ``shift``."""
    pos = np.asarray(system.positions, dtype=float)
    min2 = pos.copy()
    saddle = pos.copy()
    min2[move] += shift
    saddle[move] += 0.5 * np.asarray(shift)
    return EventSearchOutput(
        central_atom_index=move,
        min1_positions=pos,
        saddle_positions=saddle,
        min2_positions=min2,
        dE_forward=dE_forward,
        dE_backward=dE_backward,
        move_atom_index=move,
        cell=np.asarray(system.cell, dtype=float),
        types=list(system.types),
    )


def _table_with_service(
    config: Any, forward: Any, backward: Any
) -> tuple[ReferenceEventTable, FakeManager]:
    """Build a reference table whose fake worker answers every request with the pair."""
    fake = FakeManager(lambda req: protocol_event_prefactors(req, forward, backward))
    service = PrefactorService(
        config, fake, create_rate_constant(config.rateconstant), method="fd"
    )
    return ReferenceEventTable(config, prefactor_service=service), fake


@pytest.fixture
def htst_log_records() -> list[logging.LogRecord]:
    """Capture the records of the ``log`` logger the catalogue writes to."""
    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Collect(level=logging.DEBUG)
    logger = logging.getLogger("log")
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


class TestPrefactorAgreement:
    """``self_reverse_prefactors_agree`` is the numerical half of the comparison."""

    def test_within_tolerance_agrees(self) -> None:
        """Two accepted estimates inside the relative tolerance agree."""
        assert self_reverse_prefactors_agree(
            accepted(5.0e12), accepted(5.0e12 * (1.0 + 0.5 * SELF_REVERSE_NU0_RTOL))
        )

    def test_outside_tolerance_disagrees(self) -> None:
        """Two accepted estimates outside the tolerance do not agree."""
        assert not self_reverse_prefactors_agree(accepted(5.0e12), accepted(3.0e12))

    @pytest.mark.parametrize(
        "forward,backward",
        [
            (accepted(5.0e12), rejected()),
            (rejected(), accepted(5.0e12)),
            (rejected(), rejected()),
            (accepted(5.0e12), skipped()),
        ],
    )
    def test_rejected_or_skipped_direction_never_agrees(
        self, forward: Any, backward: Any
    ) -> None:
        """Equal fallbacks (or no estimate at all) do not demonstrate equal spectra."""
        assert not self_reverse_prefactors_agree(forward, backward)


class TestSameTopologyIsOneRow:
    """Equal endpoint topologies propose a candidate; HTST proof is deferred."""

    def test_htst_admits_a_single_self_linked_candidate(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """Matching crops propose a candidate; admission retains both directions."""
        table = ReferenceEventTable(htst_config)
        fwd, bwd = _series(table, system_single_type_fcc, 0, 0.5, 0.5)
        assert fwd["event_id"] == fwd["id_final"]

        admission = table._admit_series(fwd, bwd).ok_value()
        assert admission.same_topology is True
        assert admission.self_reverse_candidate is True
        assert admission.reverse_idx_ref is None
        assert len(admission.frame) == 2

        table.add(admission.frame, reverse_idx_ref=admission.reverse_idx_ref)
        assert _links(table) == [(0, 1), (1, 0)]

    def test_mismatching_saddle_crops_still_admit_one_row(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """Coarse crop mismatch never removes a provisional direction."""
        table = ReferenceEventTable(htst_config)
        fwd, bwd = _series(table, system_single_type_fcc, 0, 0.5, 0.5)
        rng = np.random.default_rng(7)
        bwd["saddle_positions"] = np.asarray(bwd["saddle_positions"]) + rng.uniform(
            -1.5, 1.5, size=np.asarray(bwd["saddle_positions"]).shape
        )
        assert table._saddle_crops_match(fwd, bwd) is False
        admission = table._admit_series(fwd, bwd).ok_value()
        assert admission.same_topology is True
        assert admission.self_reverse_candidate is True
        assert len(admission.frame) == 2
        table.add(admission.frame)
        assert _links(table) == [(0, 1), (1, 0)]

    def test_constant_collapses_as_the_base(
        self, constant_config: Any, system_single_type_fcc: Any
    ) -> None:
        """Constant mode keeps the base representation: one self-linked row."""
        table = ReferenceEventTable(constant_config)
        fwd, bwd = _series(table, system_single_type_fcc, 0, 0.5, 0.5)
        admission = table._admit_series(fwd, bwd).ok_value()
        assert admission.self_reverse_candidate is False
        assert len(admission.frame) == 1
        table.add(admission.frame)
        assert _links(table) == [(0, 0)]

    def test_distinct_topologies_still_admit_two_rows(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """A hop between different topologies keeps its two directional rows."""
        table = ReferenceEventTable(htst_config)
        fwd, bwd = _series(
            table, system_single_type_fcc, 0, 2.0, 1.5, np.array([1.2, 0.3, 0.0])
        )
        assert fwd["event_id"] != fwd["id_final"]
        admission = table._admit_series(fwd, bwd).ok_value()
        assert admission.same_topology is False
        assert admission.self_reverse_candidate is False
        assert len(admission.frame) == 2
        table.add(admission.frame)
        assert _links(table) == [(0, 1), (1, 0)]


class TestSelfReverseRecording:
    """Both directional results are stored before a proven self-reverse merge."""

    def test_symmetric_spectra_keep_an_empty_reason(
        self, htst_config: Any, system_single_type_fcc: Any, htst_log_records: Any
    ) -> None:
        """Agreement within the tolerance: one row, forward estimate, no reason."""
        table, fake = _table_with_service(
            htst_config, accepted(5.0e12), accepted(5.02e12)
        )
        table.add_events(
            [_trivial_event(system_single_type_fcc)], pbc=system_single_type_fcc.pbc
        )
        assert _links(table) == [(0, 0)]
        assert len(fake.prefactor_requests) == 1
        assert fake.prefactor_requests[0].event_key == (0, 1)
        assert fake.prefactor_backward_flags == [True]
        row = table.table.iloc[0]
        assert row["nu0"] == 5.0e12 and row["k_prefactor"] == 5.0
        assert row["nu0_status"] == "ok" and row["nu0_reason"] == ""
        assert row["k"] == rate_from_prefactor(5.0, 0.5, htst_config.rateconstant.T)
        assert not [r for r in htst_log_records if r.levelno >= logging.WARNING]
        assert any(
            "whole-event self-reverse" in r.getMessage() for r in htst_log_records
        )

    def test_asymmetric_spectra_keep_one_row_and_record_the_backward(
        self, htst_config: Any, system_single_type_fcc: Any, htst_log_records: Any
    ) -> None:
        """Disagreement keeps one forward row; the backward value is archived."""
        # contracts 7f policy 6: an unproven candidate is one self-linked row
        table, fake = _table_with_service(
            htst_config, accepted(5.0e12), accepted(3.0e12)
        )
        table.add_events(
            [_trivial_event(system_single_type_fcc)], pbc=system_single_type_fcc.pbc
        )
        assert _links(table) == [(0, 0)]
        assert len(table.table) == 1
        row = table.table.iloc[0]
        assert (
            row["nu0"] == 5.0e12 and row["k_prefactor"] == 5.0
        )  # forward, no averaging
        assert row["nu0_status"] == "ok"
        assert (
            row["nu0_reason"]
            == "self-reverse unproven: backward nu0 = 3.0000e+12 Hz, differs by 40.0%"
        )
        warnings = [r for r in htst_log_records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "differs by 40.0%" in warnings[0].getMessage()
        assert "collapsed to one self-linked row" in warnings[0].getMessage()
        assert archived_frequency(table, 1, 3.0e12)
        assert any(
            entry["k_prefactor"] == 3.0 and entry["nu0_status"] == "ok"
            for entry in table.prefactor_archive.history[1]
        )
        assert row["k"] == rate_from_prefactor(5.0, 0.5, htst_config.rateconstant.T)
        assert table.max_idx_ref() == 2

    def test_rejected_backward_is_recorded_on_the_single_row(
        self, htst_config: Any, system_single_type_fcc: Any, htst_log_records: Any
    ) -> None:
        """Rejected backward: one accepted forward row, the rejection archived."""
        # contracts 7f policy 6: an unproven candidate is one self-linked row
        table, _ = _table_with_service(
            htst_config, accepted(5.0e12), rejected("bwd window")
        )
        table.add_events(
            [_trivial_event(system_single_type_fcc)], pbc=system_single_type_fcc.pbc
        )
        assert _links(table) == [(0, 0)]
        row = table.table.iloc[0]
        assert row["nu0_status"] == "ok" and row["nu0"] == 5.0e12
        assert row["nu0_reason"] == (
            "self-reverse unproven: backward prefactor rejected (out_of_window: bwd window)"
        )
        backward = table.prefactor_archive.history[1][-1]
        assert backward["nu0_status"] == "rejected"
        assert backward["nu0_reason"] == "out_of_window: bwd window"
        assert backward["k_prefactor"] == htst_config.rateconstant.k0
        assert backward["nu0"] is None
        assert sum(r.levelno == logging.WARNING for r in htst_log_records) == 1

    def test_rejected_forward_keeps_k0_and_notes_the_backward(
        self, htst_config: Any, system_single_type_fcc: Any, htst_log_records: Any
    ) -> None:
        """Rejected forward keeps k0 on the single row; the backward is archived."""
        # contracts 7f policy 6: an unproven candidate is one self-linked row
        table, _ = _table_with_service(
            htst_config, rejected("fwd window"), accepted(4.0e12)
        )
        table.add_events(
            [_trivial_event(system_single_type_fcc)], pbc=system_single_type_fcc.pbc
        )
        assert _links(table) == [(0, 0)]
        row = table.table.iloc[0]
        assert row["nu0_status"] == "rejected"
        assert row["k_prefactor"] == htst_config.rateconstant.k0
        assert row["nu0_reason"] == (
            "out_of_window: fwd window; self-reverse unproven: backward nu0 = 4.0000e+12 Hz "
            "(forward rejected)"
        )
        assert archived_frequency(table, 1, 4.0e12)
        assert sum(r.levelno == logging.WARNING for r in htst_log_records) == 1

    def test_skipped_estimate_is_never_written(self, htst_config: Any) -> None:
        """``_patch_row`` refuses a skipped direction: it is no estimate."""
        table = ReferenceEventTable(htst_config)
        with pytest.raises(ValueError, match="skipped"):
            table._patch_row(0, skipped())

    def test_htst_log_lines_carry_n_free_and_batch_time(
        self, htst_config: Any, system_single_type_fcc: Any, htst_log_records: Any
    ) -> None:
        """Every reference line reports the free-atom count and the batch wall time.

        The forward line and the whole-event comparison line both
        end with the ``(n_free N, batch T s)`` suffix of contract 7c.
        """
        table, fake = _table_with_service(
            htst_config, accepted(5.0e12), accepted(5.0e12)
        )
        table.add_events(
            [_trivial_event(system_single_type_fcc)], pbc=system_single_type_fcc.pbc
        )
        lines = [
            r.getMessage()
            for r in htst_log_records
            if r.getMessage().startswith("[htst] reference event 0")
        ]
        assert len(lines) == 2
        forward, comparison = lines
        assert forward.startswith(
            "[htst] reference event 0 (forward): nu0 = 5.0000e+12 Hz"
        )
        assert (
            "whole-event self-reverse, backward nu0 = 5.0000e+12 Hz agrees within 5%"
            in comparison
        )
        backward_lines = [
            r.getMessage()
            for r in htst_log_records
            if r.getMessage().startswith("[htst] reference event 1 (backward)")
        ]
        assert len(backward_lines) == 1
        assert "nu0 = 5.0000e+12 Hz" in backward_lines[0]
        for line in lines + backward_lines:
            match = _BATCH_SUFFIX.search(line)
            assert match is not None, line
            assert int(match.group(1)) == len(
                common_free_indices(fake.prefactor_requests[0])
            )
            assert float(match.group(2)) >= 0.0

    @pytest.mark.parametrize(
        "forward,backward,expected",
        [
            (accepted(5.0e12), accepted(3.0e12), "differs by 40.0%"),
            (accepted(5.0e12), rejected("bwd window"), "backward prefactor rejected"),
            (rejected("fwd window"), accepted(4.0e12), "(forward rejected)"),
        ],
    )
    def test_self_reverse_warning_carries_n_free_and_batch_time(
        self,
        htst_config: Any,
        system_single_type_fcc: Any,
        htst_log_records: Any,
        forward: Any,
        backward: Any,
        expected: str,
    ) -> None:
        """The discrepancy warning is a reference line too: it ends with the suffix."""
        table, fake = _table_with_service(htst_config, forward, backward)
        table.add_events(
            [_trivial_event(system_single_type_fcc)], pbc=system_single_type_fcc.pbc
        )
        warnings = [
            r.getMessage() for r in htst_log_records if r.levelno == logging.WARNING
        ]
        assert len(warnings) == 1
        assert warnings[0].startswith("[htst] reference event 0: self-reverse")
        assert expected in warnings[0]
        match = _BATCH_SUFFIX.search(warnings[0])
        assert match is not None, warnings[0]
        assert int(match.group(1)) == len(
            common_free_indices(fake.prefactor_requests[0])
        )


class TestLinkingNotesCarryBatchData:
    """Retained directions keep complete diagnostic free-count and batch metadata."""

    def test_unmapped_saddle_crops_note_carries_n_free_and_batch_time(
        self,
        htst_config: Any,
        system_single_type_fcc: Any,
        htst_log_records: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A coarse matcher seam cannot discard an unequal accepted direction."""
        # contracts 7f policy 6: the unequal backward value survives in the archive
        monkeypatch.setattr(
            ReferenceEventTable, "_saddle_crops_match", lambda self, fwd, bwd: False
        )
        table, fake = _table_with_service(
            htst_config, accepted(5.0e12), accepted(3.0e12)
        )
        table.add_events(
            [_trivial_event(system_single_type_fcc)], pbc=system_single_type_fcc.pbc
        )
        assert _links(table) == [(0, 0)]
        assert fake.prefactor_backward_flags == [True]
        row = table.table.iloc[0]
        assert row["nu0"] == 5.0e12
        assert "differs by 40.0%" in row["nu0_reason"]
        assert archived_frequency(table, 1, 3.0e12)
        assert sum(r.levelno == logging.WARNING for r in htst_log_records) == 1
        notes = [
            r.getMessage()
            for r in htst_log_records
            if "collapsed to one self-linked row" in r.getMessage()
        ]
        assert len(notes) == 1
        assert notes[0].startswith("[htst] reference event 0: self-reverse unproven")
        match = _BATCH_SUFFIX.search(notes[0])
        assert match is not None, notes[0]
        assert int(match.group(1)) == len(
            common_free_indices(fake.prefactor_requests[0])
        )

    def test_reverse_already_catalogued_note_carries_n_free_and_batch_time(
        self, htst_config: Any, system_single_type_fcc: Any, htst_log_records: Any
    ) -> None:
        """A catalogued reverse (even bare) links the lone forward row, as the base."""
        # contracts 7f policy 6: the geometric gate decides, whatever row 7's status
        table, fake = _table_with_service(
            htst_config, accepted(5.0e12), accepted(3.0e12)
        )
        shift = np.array([1.2, 0.3, 0.0])
        _, bwd = _series(table, system_single_type_fcc, 0, 2.0, 1.5, shift)
        _insert(table, bwd, idx_ref=7, idx_backward=7)
        table.add_events(
            [_hop_event(system_single_type_fcc, 0, 2.0, 1.5, shift)],
            pbc=system_single_type_fcc.pbc,
        )
        assert _links(table) == [(7, 7), (8, 7)]
        assert [r.event_key for r in fake.prefactor_requests] == [(8, None)]
        row = table.table[table.table["idx_ref"] == 8].iloc[0]
        assert row["nu0"] == 5.0e12 and row["nu0_reason"] == ""
        assert not [r for r in htst_log_records if r.levelno >= logging.WARNING]
        # Row 7 keeps its own (absent) estimate; this search's backward is
        # archived under the forward id and never becomes a selectable row.
        assert table.prefactor_archive.references.get(7) is None
        assert archived_frequency(table, 8, 3.0e12)
        assert any(
            "reverse already catalogued as reference 7" in entry["reason"]
            for entry in table.prefactor_archive.history[8]
        )
        notes = [
            r.getMessage()
            for r in htst_log_records
            if r.getMessage().startswith("[htst] reference event 8")
        ]
        assert len(notes) == 2
        assert "reverse already catalogued as event 7" in notes[1]
        for line in notes:
            match = _BATCH_SUFFIX.search(line)
            assert match is not None, line
            assert int(match.group(1)) == len(
                common_free_indices(fake.prefactor_requests[0])
            )
            assert float(match.group(2)) >= 0.0


class TestReverseAlreadyCatalogued:
    """Case (b): the backward direction of a new event is already a row."""

    def _catalogue_with_reverse(
        self, table: ReferenceEventTable, system: Any
    ) -> tuple[pd.Series, pd.Series]:
        """Insert the reverse (topology B, 0.7 eV) at the sparse id 7; return F."""
        fwd, bwd = _series(table, system, 0, 0.5, 0.7)
        fwd["event_id"] = "TOPO_A"
        fwd["id_final"] = "TOPO_B"
        bwd["event_id"] = "TOPO_B"
        bwd["id_final"] = "TOPO_A"
        existing = bwd.copy()
        _insert(table, existing, idx_ref=7, idx_backward=7)
        return fwd, bwd

    def _resolved_catalogue_with_reverse(self, config, system):
        """Resolve a full-source typed record before linking a sparse reverse."""
        table, fake = _table_with_service(config, accepted(5.0e12), accepted(3.0e12))
        fwd, bwd = self._catalogue_with_reverse(table, system)
        event = _hop_event(system, 0, 0.5, 0.7, np.zeros(3))
        request = table._event_request(event, system.pbc, event_key=("existing", 7))
        # contracts 7f policy 6: the geometric gate proposes the catalogued
        # reverse whatever its prefactor status, with or without a request.
        assert table._admit_series(fwd, bwd).ok_value().reverse_idx_ref == 7
        assert (
            table._admit_series(fwd, bwd, request=request).ok_value().reverse_idx_ref
            == 7
        )
        prior = table.prefactor_service.compute([request])[request.event_key]
        assert prior.provenance.source.is_complete
        table._patch_row(
            7, prior.backward, calculation=prior.calculation("backward"), fresh=True
        )
        return table, fake, fwd, bwd, event, request

    def test_htst_links_to_the_matched_logical_id(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """A catalogued reverse links the lone forward row; its estimate stands."""
        # contracts 7f policy 6: one forward row, backward estimate archived
        table, fake, fwd, bwd, event, request = self._resolved_catalogue_with_reverse(
            htst_config, system_single_type_fcc
        )

        assert table.find_matching_event(bwd) == 7
        assert table.find_matching_event(fwd) is None
        admission = table._admit_series(fwd, bwd, request=request).ok_value()
        assert admission.reverse_idx_ref == 7
        assert admission.self_reverse_candidate is False
        assert admission.same_topology is False
        assert len(admission.frame) == 1

        table.add(admission.frame, reverse_idx_ref=admission.reverse_idx_ref)
        assert _links(table) == [(7, 7), (8, 7)]
        table._resolve_prefactors(
            [(8, None, admission, event)], system_single_type_fcc.pbc
        )
        assert _links(table) == [(7, 7), (8, 7)]
        assert [r.event_key for r in fake.prefactor_requests] == [
            ("existing", 7),
            (8, None),
        ]
        assert table.table.loc[table.table.idx_ref == 7, "nu0"].item() == 3.0e12
        assert table.table.loc[table.table.idx_ref == 8, "nu0"].item() == 5.0e12
        assert table.prefactor_archive.history[8][-1]["nu0"] == 3.0e12
        assert "reference 7" in table.prefactor_archive.history[8][-1]["reason"]

    def test_constant_self_links_as_the_base(
        self, constant_config: Any, system_single_type_fcc: Any
    ) -> None:
        """Constant mode keeps the base self-link for a lone forward row."""
        table = ReferenceEventTable(constant_config)
        fwd, bwd = self._catalogue_with_reverse(table, system_single_type_fcc)
        admission = table._admit_series(fwd, bwd).ok_value()
        assert admission.reverse_idx_ref is None
        table.add(admission.frame, reverse_idx_ref=admission.reverse_idx_ref)
        assert _links(table) == [(7, 7), (8, 8)]

    def test_remove_drops_rows_linked_to_a_removed_reverse(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """Removing the reverse also removes the forward row that links to it."""
        table, _, fwd, bwd, event, request = self._resolved_catalogue_with_reverse(
            htst_config, system_single_type_fcc
        )
        admission = table._admit_series(fwd, bwd, request=request).ok_value()
        table.add(admission.frame, reverse_idx_ref=admission.reverse_idx_ref)
        table._resolve_prefactors(
            [(8, None, admission, event)], system_single_type_fcc.pbc
        )
        assert _links(table) == [(7, 7), (8, 7)]

        table.remove([7])
        assert _links(table) == []

    def test_remove_forward_drops_its_reverse(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """Removing the forward row removes the reverse it links to (base rule)."""
        table, _, fwd, bwd, event, request = self._resolved_catalogue_with_reverse(
            htst_config, system_single_type_fcc
        )
        admission = table._admit_series(fwd, bwd, request=request).ok_value()
        table.add(admission.frame, reverse_idx_ref=admission.reverse_idx_ref)
        table._resolve_prefactors(
            [(8, None, admission, event)], system_single_type_fcc.pbc
        )
        table.remove([8])
        assert _links(table) == []


class TestRemoveClosure:
    """``remove`` never leaves a dangling reverse link in htst/rpa mode."""

    @staticmethod
    def _chain(table: ReferenceEventTable, system: Any) -> None:
        """X(3) <-> Y(4); F(7) -> X; W(11) -> Y; V(13) -> W; Z(9) self-linked."""
        row, _ = _series(table, system, 0, 0.5, 0.5)
        for idx_ref, idx_backward in (
            (3, 4),
            (4, 3),
            (7, 3),
            (11, 4),
            (13, 11),
            (9, 9),
        ):
            _insert(table, row, idx_ref=idx_ref, idx_backward=idx_backward)

    def test_htst_removal_is_closed_under_reverse_links(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """Two hops away (W -> Y, V -> W) go too; the unrelated row stays."""
        table = ReferenceEventTable(htst_config)
        self._chain(table, system_single_type_fcc)
        table.remove([7])
        # F(7) and its reverse X(3) by the base rule; Y(4) points at X, W(11)
        # points at Y and V(13) points at W: the closure drops all of them.
        assert _links(table) == [(9, 9)]
        surviving = {r for r, _ in _links(table)}
        assert all(b in surviving for _, b in _links(table))

    def test_htst_removal_of_a_leaf_keeps_the_rest(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """Removing V(13) drops W(11) (its reverse) and nothing points at them."""
        table = ReferenceEventTable(htst_config)
        self._chain(table, system_single_type_fcc)
        table.remove([13])
        assert _links(table) == [(3, 4), (4, 3), (7, 3), (9, 9)]

    def test_constant_removal_is_the_base_rule(
        self, constant_config: Any, system_single_type_fcc: Any
    ) -> None:
        """Constant mode removes the event and its backward only, as the base."""
        table = ReferenceEventTable(constant_config)
        self._chain(table, system_single_type_fcc)
        table.remove([7])
        assert _links(table) == [(4, 3), (11, 4), (13, 11), (9, 9)]


class TestSparseReorderedCatalogue:
    """Logical ids are returned even when rows are shuffled and sparse."""

    def test_find_matching_event_returns_logical_id(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """The match is reported by ``idx_ref``, never by row position."""
        table = ReferenceEventTable(htst_config)
        a, _ = _series(table, system_single_type_fcc, 0, 0.5, 0.5)
        b, _ = _series(table, system_single_type_fcc, 0, 1.0, 1.0)
        c, _ = _series(table, system_single_type_fcc, 0, 1.6, 1.6)
        _insert(table, a, idx_ref=12, idx_backward=12)
        _insert(table, b, idx_ref=3, idx_backward=3)
        _insert(table, c, idx_ref=7, idx_backward=7)

        probe, _ = _series(table, system_single_type_fcc, 0, 1.05, 1.05)
        assert table.find_matching_event(probe) == 3
        assert table.is_new_event(probe) is False
        assert table.max_idx_ref() == 13

    @pytest.mark.parametrize("coloring,expected", [("full", False), ("grey", True)])
    def test_species_swapped_saddle_crops_follow_the_colouring_mode(
        self, htst_config: Any, system_binary_fcc: Any, coloring: str, expected: bool
    ) -> None:
        """Ni<->Fe swapped directional crops are a candidate only in grey mode.

        ``_saddle_crops_match`` feeds IRA the same labels as
        ``find_matching_event``: the local element types in ``full``
        colouring, one grey label otherwise. Both modes retain a pair until a full physical map is proven.
        """
        env = htst_config.atomicenvironment.model_copy(
            update={"atom_coloring_mode": coloring}
        )
        config = htst_config.model_copy(update={"atomicenvironment": env})
        table = ReferenceEventTable(config)
        fwd, bwd = _series(table, system_binary_fcc, 0, 0.5, 0.5)
        assert set(fwd["types"]) == {"Ni", "Fe"}
        assert table._saddle_crops_match(fwd, bwd) is True  # identical crops
        bwd["types"] = ["Fe" if t == "Ni" else "Ni" for t in bwd["types"]]
        assert table._saddle_crops_match(fwd, bwd) is expected
        admission = table._admit_series(fwd, bwd).ok_value()
        assert admission.self_reverse_candidate is True
        assert admission.same_topology is True
        assert len(admission.frame) == 2


class TestConstantModeFixedSequence:
    """Constant-mode admission of a fixed sequence equals the base catalogue.

    The expectations were recorded by running the same sequence through the
    integration base (``3974ace``): the pickled tables are byte-identical.
    """

    @staticmethod
    def _events(system: Any) -> list[EventSearchOutput]:
        pos = np.asarray(system.positions, dtype=float)
        cell = system.cell
        types = list(system.types)

        def make(
            move: int, dE_f: float, dE_b: float, shift: list | None = None
        ) -> EventSearchOutput:
            min2 = pos.copy()
            saddle = pos.copy()
            if shift is not None:
                min2[move] += np.asarray(shift)
                saddle[move] += 0.5 * np.asarray(shift)
            return EventSearchOutput(
                central_atom_index=move,
                min1_positions=pos.copy(),
                saddle_positions=saddle,
                min2_positions=min2,
                dE_forward=dE_f,
                dE_backward=dE_b,
                move_atom_index=move,
                cell=cell,
                types=types,
            )

        hop = [1.2, 0.3, 0.0]
        return [
            make(0, 0.5, 0.5),
            make(0, 0.6, 0.6),  # duplicate of the first (same topology, 0.1 eV)
            make(0, 1.0, 1.0),  # same topology, distinct barrier
            make(0, 2.0, 1.5, hop),  # distinct endpoint topologies
            make(5, 2.0, 1.5, hop),
            make(5, 0.5, 0.5),
        ]

    def test_catalogue_matches_the_base(
        self, constant_config: Any, system_single_type_fcc: Any
    ) -> None:
        """Acceptance, ids and reciprocal links equal the base recording."""
        table = ReferenceEventTable(constant_config)
        results = table.add_events(self._events(system_single_type_fcc))

        assert [r.is_ok() for r in results] == [True, False, True, True, True, True]
        assert list(table.table.columns) == [
            "idx_ref",
            "event_id",
            "initial_positions",
            "saddle_positions",
            "final_positions",
            "types",
            "energy_barrier",
            "k",
            "id_saddle",
            "id_final",
            "move_atom_idx",
            "sym_matrix",
            "sym_perm",
            "idx_backward",
            "dra",
        ]
        assert _links(table) == [(0, 0), (1, 1), (2, 3), (3, 2), (4, 5), (5, 4), (6, 6)]
        assert list(table.table["energy_barrier"]) == [
            0.5,
            1.0,
            2.0,
            1.5,
            2.0,
            1.5,
            0.5,
        ]
        same = table.table["event_id"] == table.table["id_final"]
        assert list(same) == [True, True, False, False, False, False, True]

        table.remove([1])
        assert _links(table) == [(0, 0), (2, 3), (3, 2), (4, 5), (5, 4), (6, 6)]

    def test_htst_same_sequence_has_the_same_row_layout(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """HTST admits the constant-mode sequence; unproven candidates are one row.

        Admission is the constant-mode geometric gate (contracts 7f policy 6):
        the same five events are dispatched. The hop at atom 5 is a lattice
        translation of the hop at atom 0 that the saddle-crop IRA does not
        match, so both are admitted; after resolution the whole-event proof
        maps the new pair onto rows 4/5 with agreeing spectra and merges it
        (policy 4), so that search is reported as not new.
        """
        table, fake = _table_with_service(
            htst_config, accepted(5.0e12), accepted(4.0e12)
        )
        results = table.add_events(
            self._events(system_single_type_fcc), pbc=system_single_type_fcc.pbc
        )
        assert [r.is_ok() for r in results] == [True, False, True, True, False, True]
        assert results[4].err_value().type == ErrorType.EVENT_NOT_NEW
        assert _links(table) == [(0, 0), (2, 2), (4, 5), (5, 4), (8, 8)]
        assert len(fake.prefactor_requests) == 5
        assert [k for k in (r.event_key for r in fake.prefactor_requests)] == [
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (8, 9),
        ]
        assert list(table.table.energy_barrier) == [0.5, 1.0, 2.0, 1.5, 0.5]
        # Same-topology candidates keep the forward estimate, note the 20 % gap
        # and archive the backward value under the discarded id.
        for idx in (0, 2, 8):
            row = table.table[table.table["idx_ref"] == idx].iloc[0]
            assert row["nu0"] == 5.0e12
            assert row["nu0_reason"].startswith(
                "self-reverse unproven: backward nu0 = 4.0000e+12 Hz"
            )
            assert archived_frequency(table, idx + 1, 4.0e12)
        assert table.table[table.table["idx_ref"] == 4].iloc[0]["nu0"] == 5.0e12
        backward = table.table[table.table["idx_ref"] == 5].iloc[0]
        assert backward["nu0"] == 4.0e12 and backward["nu0_reason"] == ""
        for merged, survivor in ((6, 4), (7, 5)):
            assert any(
                entry["reason"] == f"whole-event equivalent to reference {survivor}"
                for entry in table.prefactor_archive.history[merged]
            )
        assert table.max_idx_ref() == 10


class TestSameTopologyBarriersDecide:
    """htst/rpa: same-topology admission is decided by the barriers (contracts 7d, F3)."""

    def test_unequal_barriers_admit_two_directional_rows(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """0.5 / 0.7 eV with matching topologies and crops: two rows, reciprocal links."""
        from pykmc.event_table import SELF_REVERSE_BARRIER_TOL

        assert SELF_REVERSE_BARRIER_TOL == 0.01
        table = ReferenceEventTable(htst_config)
        fwd, bwd = _series(table, system_single_type_fcc, 0, 0.5, 0.7)
        assert fwd["event_id"] == fwd["id_final"]
        assert table._saddle_crops_match(fwd, bwd)  # crops map: still two rows
        admission = table._admit_series(fwd, bwd).ok_value()
        assert admission.same_topology is False
        assert admission.self_reverse_candidate is False
        assert admission.reverse_idx_ref is None
        assert len(admission.frame) == 2
        table.add(admission.frame)
        assert _links(table) == [(0, 1), (1, 0)]
        assert list(table.table["energy_barrier"]) == [0.5, 0.7]
        assert list(table.table["event_id"]) == [fwd["event_id"]] * 2

    def test_gap_inside_the_tolerance_is_one_row_beyond_it_two(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """A 9 meV gap may be proved later; an 11 meV gap cannot collapse."""
        from pykmc.event_table import SELF_REVERSE_BARRIER_TOL

        table = ReferenceEventTable(htst_config)
        fwd, bwd = _series(table, system_single_type_fcc, 0, 0.5, 0.509)
        assert abs(0.509 - 0.5) < SELF_REVERSE_BARRIER_TOL
        one = table._admit_series(fwd, bwd).ok_value()
        assert (
            one.same_topology is True
            and one.self_reverse_candidate is True
            and len(one.frame) == 2
        )
        fwd, bwd = _series(table, system_single_type_fcc, 0, 0.5, 0.511)
        assert abs(0.511 - 0.5) > SELF_REVERSE_BARRIER_TOL
        two = table._admit_series(fwd, bwd).ok_value()
        assert (
            two.same_topology is False
            and two.self_reverse_candidate is False
            and len(two.frame) == 2
        )

    def test_unequal_barriers_get_separate_estimates_without_comparison(
        self, htst_config: Any, system_single_type_fcc: Any, htst_log_records: Any
    ) -> None:
        """Equal prefactors on both rows never collapse them; no self-reverse note."""
        table, fake = _table_with_service(
            htst_config, accepted(5.0e12), accepted(5.0e12)
        )
        pos = np.asarray(system_single_type_fcc.positions, dtype=float)
        event = EventSearchOutput(
            central_atom_index=0,
            min1_positions=pos.copy(),
            saddle_positions=pos.copy(),
            min2_positions=pos.copy(),
            dE_forward=0.5,
            dE_backward=0.7,
            move_atom_index=0,
            cell=np.asarray(system_single_type_fcc.cell, dtype=float),
            types=list(system_single_type_fcc.types),
        )
        results = table.add_events([event], pbc=system_single_type_fcc.pbc)
        assert results[0].is_ok()
        assert _links(table) == [(0, 1), (1, 0)]
        assert len(fake.prefactor_requests) == 1
        assert fake.prefactor_requests[0].event_key == (0, 1)
        T = htst_config.rateconstant.T
        forward = table.table[table.table["idx_ref"] == 0].iloc[0]
        backward = table.table[table.table["idx_ref"] == 1].iloc[0]
        for row, dE in ((forward, 0.5), (backward, 0.7)):
            assert row["nu0"] == 5.0e12 and row["nu0_status"] == "ok"
            assert row["nu0_reason"] == ""
            assert row["energy_barrier"] == dE
            assert row["k"] == rate_from_prefactor(5.0, dE, T)
        assert forward["k"] > backward["k"]
        assert table.prefactor_summary()["ok"] == 2
        assert not [r for r in htst_log_records if r.levelno >= logging.WARNING]
        assert not [r for r in htst_log_records if "self-reverse" in r.getMessage()]
        assert table.max_idx_ref() == 2

    def test_constant_mode_still_collapses_by_topology_alone(
        self, constant_config: Any, system_single_type_fcc: Any
    ) -> None:
        """The base rule is untouched: constant mode keeps one row for 0.5 / 0.7 eV."""
        table = ReferenceEventTable(constant_config)
        fwd, bwd = _series(table, system_single_type_fcc, 0, 0.5, 0.7)
        admission = table._admit_series(fwd, bwd).ok_value()
        assert admission.same_topology is True
        assert len(admission.frame) == 1
        table.add(admission.frame)
        assert _links(table) == [(0, 0)]


class TestRemoveReport:
    """``remove`` reports the complete removed set with its topologies (contracts 7d, F4)."""

    @staticmethod
    def _catalogue(table: ReferenceEventTable, system: Any) -> None:
        """Insert 0 <-> 1 (A, B), 2 -> 0 (C) and 9 <-> 9 (D)."""
        for idx_ref, idx_backward, topology in (
            (0, 1, "A"),
            (1, 0, "B"),
            (2, 0, "C"),
            (9, 9, "D"),
        ):
            row, _ = _series(table, system, 0, 0.5, 0.5)
            row["event_id"] = topology
            _insert(table, row, idx_ref=idx_ref, idx_backward=idx_backward)

    def test_htst_remove_reports_the_closure_and_its_topologies(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """Removing 2 removes 0 and 1 as well; the report is sorted and aligned."""
        from pykmc.event_table import RemovedReferences

        table = ReferenceEventTable(htst_config)
        self._catalogue(table, system_single_type_fcc)
        removed = table.remove([2])
        assert isinstance(removed, RemovedReferences)
        assert removed.idx_refs == (0, 1, 2)
        assert removed.event_ids == ("A", "B", "C")
        assert len(removed) == 3
        assert _links(table) == [(9, 9)]

    def test_remove_of_an_unknown_id_reports_nothing(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """An id without a row removes nothing and reports an empty set."""
        table = ReferenceEventTable(htst_config)
        self._catalogue(table, system_single_type_fcc)
        removed = table.remove([42])
        assert removed.idx_refs == () and removed.event_ids == ()
        assert len(removed) == 0
        assert len(table.table) == 4

    def test_constant_remove_reports_the_base_pair(
        self, constant_config: Any, system_single_type_fcc: Any
    ) -> None:
        """Constant mode reports the base pair rule (no closure)."""
        table = ReferenceEventTable(constant_config)
        self._catalogue(table, system_single_type_fcc)
        removed = table.remove([0])
        assert removed.idx_refs == (0, 1)
        assert removed.event_ids == ("A", "B")
        assert _links(table) == [(2, 0), (9, 9)]  # base rule leaves the alias
