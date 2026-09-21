"""Geometric admission gate and one-row self-reverse fallback (contracts 7f policy 6).

In htst/rpa mode ``find_matching_event`` (topology + 0.25 eV + IRA saddle-crop
match) is the admission duplicate gate exactly as in constant mode, so a
catalogued event stays a duplicate whatever its prefactor status. The
whole-event proof only decides whether a resolved pair may be collapsed; a
same-topology candidate whose merge is not proven (rejected or disagreeing
backward estimate) collapses back to one self-linked forward row, with the
backward estimate retained in the archive and the failure in ``nu0_reason``.
No LAMMPS, no MPI: the worker is the protocol double of the lifecycle suite.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest

from pykmc.result import ErrorType
from tests.lifecycle.conftest import accepted, rejected

from .protocol_producers import archived_frequency
from .test_identity_gate import (
    _hop_event,
    _links,
    _table_with_service,
    _trivial_event,
)

HOP = np.array([0.6, 0.0, 0.0])


def _add_hop(table: Any, system: Any) -> list[Any]:
    return table.add_events([_hop_event(system, 0, 0.5, 0.5, HOP)], pbc=system.pbc)


class TestGeometricAdmissionGate:
    """Duplicate rejection never depends on an accepted prefactor."""

    @pytest.mark.parametrize(
        "forward,backward",
        [
            (rejected("w"), rejected("w")),
            (accepted(5.0e12), rejected("w")),
            (accepted(5.0e12), accepted(5.0e12)),
        ],
        ids=["both-rejected", "backward-rejected", "both-accepted"],
    )
    def test_readmitting_the_same_hop_is_rejected_whatever_the_prefactors(
        self, htst_config: Any, system_single_type_fcc: Any, forward: Any, backward: Any
    ) -> None:
        """Four identical searches: two rows, one worker request, three EVENT_NOT_NEW."""
        table, fake = _table_with_service(htst_config, forward, backward)
        rows = []
        results = []
        for _ in range(4):
            results.append(_add_hop(table, system_single_type_fcc)[0])
            rows.append(len(table.table))
        assert rows == [2, 2, 2, 2]
        assert results[0].is_ok()
        for res in results[1:]:
            assert not res.is_ok()
            assert res.err_value().type == ErrorType.EVENT_NOT_NEW
            assert res.err_value().details == "Same topology"
        assert len(fake.prefactor_requests) == 1
        assert table.table.event_id.nunique() == 2
        assert _links(table) == [(0, 1), (1, 0)]

    def test_constant_and_htst_admit_the_same_sequence(
        self, constant_config: Any, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """The acceptance pattern of a mixed sequence is style-independent."""
        from pykmc.event_table import ReferenceEventTable

        system = system_single_type_fcc
        events = [
            _trivial_event(system, 0, 0.5),
            _trivial_event(system, 0, 0.6),  # same topology, 0.1 eV: duplicate
            _trivial_event(system, 0, 1.0),  # same topology, distinct barrier
            _hop_event(system, 0, 2.0, 1.5, HOP),
            _hop_event(system, 0, 2.0, 1.5, HOP),  # exact repeat
        ]
        constant = ReferenceEventTable(constant_config)
        constant_ok = [r.is_ok() for r in constant.add_events(events)]
        table, _ = _table_with_service(htst_config, rejected("w"), rejected("w"))
        htst_ok = [r.is_ok() for r in table.add_events(events, pbc=system.pbc)]
        assert constant_ok == [True, False, True, True, False]
        assert htst_ok == constant_ok


class TestUnprovenSelfReverseIsOneRow:
    """An unproven same-topology candidate never exposes two selectable rows."""

    def test_rejected_backward_collapses_to_one_self_linked_forward_row(
        self, htst_config: Any, system_single_type_fcc: Any, htst_log_records: Any
    ) -> None:
        """Forward accepted, backward rejected: one row, k = k_forward only."""
        table, fake = _table_with_service(
            htst_config, accepted(5.0e12), rejected("bwd window")
        )
        system = system_single_type_fcc
        results = table.add_events([_trivial_event(system)], pbc=system.pbc)
        assert results[0].is_ok() and len(results[0].ok_value()) == 1
        assert len(table.table) == 1
        assert _links(table) == [(0, 0)]
        assert len(fake.prefactor_requests) == 1
        row = table.table.iloc[0]
        assert row["nu0_status"] == "ok" and row["nu0"] == 5.0e12
        assert row["k_prefactor"] == 5.0
        assert "self-reverse unproven: backward prefactor rejected" in row["nu0_reason"]
        assert "out_of_window: bwd window" in row["nu0_reason"]
        # The site expansion sees exactly one channel for this environment.
        subset = table.has_id_subset_table([row["event_id"]])
        assert len(subset) == 1
        assert float(subset.k.sum()) == float(row["k"])
        # The discarded backward estimate and its id survive in the archive.
        history = table.prefactor_archive.history[1]
        assert any(entry["nu0_status"] == "rejected" for entry in history)
        assert table.max_idx_ref() == 2
        warnings = [r for r in htst_log_records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "one self-linked row" in warnings[0].getMessage()

    def test_disagreeing_backward_collapses_and_archives_its_frequency(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """Both accepted but 40 % apart: one row, the backward value is archived."""
        table, _ = _table_with_service(htst_config, accepted(5.0e12), accepted(3.0e12))
        system = system_single_type_fcc
        table.add_events([_trivial_event(system)], pbc=system.pbc)
        assert _links(table) == [(0, 0)]
        row = table.table.iloc[0]
        assert row["nu0"] == 5.0e12 and row["nu0_status"] == "ok"
        assert "differs by 40.0%" in row["nu0_reason"]
        assert archived_frequency(table, 1, 3.0e12)
        assert len(table.has_id_subset_table([row["event_id"]])) == 1

    def test_rejected_forward_keeps_k0_on_the_single_row(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """Forward rejected, backward accepted: one k0 row, backward archived."""
        table, _ = _table_with_service(
            htst_config, rejected("fwd window"), accepted(4.0e12)
        )
        system = system_single_type_fcc
        table.add_events([_trivial_event(system)], pbc=system.pbc)
        assert _links(table) == [(0, 0)]
        row = table.table.iloc[0]
        assert row["nu0_status"] == "rejected"
        assert row["k_prefactor"] == htst_config.rateconstant.k0
        assert row["nu0_reason"].startswith("out_of_window: fwd window; ")
        assert "backward nu0 = 4.0000e+12 Hz (forward rejected)" in row["nu0_reason"]
        assert archived_frequency(table, 1, 4.0e12)

    def test_collapsed_candidate_is_still_a_duplicate_afterwards(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """The surviving self-linked k0-carrying row still gates re-admission."""
        table, fake = _table_with_service(htst_config, rejected("w"), rejected("w"))
        system = system_single_type_fcc
        first = table.add_events([_trivial_event(system)], pbc=system.pbc)[0]
        second = table.add_events([_trivial_event(system)], pbc=system.pbc)[0]
        assert first.is_ok() and not second.is_ok()
        assert second.err_value().type == ErrorType.EVENT_NOT_NEW
        assert _links(table) == [(0, 0)]
        assert len(fake.prefactor_requests) == 1

    def test_proven_symmetric_control_still_collapses(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """Agreeing spectra with a common map: one row, empty reason (unchanged)."""
        table, _ = _table_with_service(htst_config, accepted(5.0e12), accepted(5.02e12))
        system = system_single_type_fcc
        table.add_events([_trivial_event(system)], pbc=system.pbc)
        assert _links(table) == [(0, 0)]
        row = table.table.iloc[0]
        assert row["nu0"] == 5.0e12 and row["nu0_reason"] == ""
