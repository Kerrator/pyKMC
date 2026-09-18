"""Directional identity gate of the reference catalogue (architecture rule S6).

``idx_ref`` is a logical event id and ``idx_backward`` a directional link; the
base admission collapses an event to one self-linked row whenever its endpoint
topologies match, and self-links a lone forward row even when its reverse is
already catalogued. In constant mode that representation is kept byte for
byte (the fixed-sequence test below pins it). In htst/rpa mode the collapse is
deferred until both directional prefactors are known and an already
catalogued reverse is linked by its logical id.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from pykmc.event_table import (
    SELF_REVERSE_NU0_RTOL,
    ReferenceEventTable,
    self_reverse_prefactors_agree,
)
from pykmc.result import EventSearchOutput
from tests.lifecycle.conftest import accepted, rejected


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


class TestPrefactorAgreement:
    """``self_reverse_prefactors_agree`` is the numerical half of the collapse."""

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
        ],
    )
    def test_rejected_direction_never_agrees(self, forward: Any, backward: Any) -> None:
        """Equal fallbacks do not demonstrate equal spectra."""
        assert not self_reverse_prefactors_agree(forward, backward)


class TestEqualTopologyUnequalSpectra:
    """Case (a): same endpoint topologies, different directional prefactors."""

    def test_htst_keeps_two_directional_rows(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """In htst both directions are admitted and kept when the spectra differ."""
        table = ReferenceEventTable(htst_config)
        fwd, bwd = _series(table, system_single_type_fcc, 0, 0.5, 0.5)
        assert fwd["event_id"] == fwd["id_final"]  # the base would collapse this

        admission = table._admit_series(fwd, bwd).ok_value()
        assert admission.self_reverse_candidate is True
        assert admission.reverse_idx_ref is None
        assert len(admission.frame) == 2

        table.add(admission.frame, reverse_idx_ref=admission.reverse_idx_ref)
        assert _links(table) == [(0, 1), (1, 0)]

        agree = self_reverse_prefactors_agree(accepted(5.0e12), accepted(3.0e12))
        table.finalize_self_reverse(0, 1, agree)
        assert _links(table) == [(0, 1), (1, 0)]

    def test_rejected_direction_keeps_two_rows(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """A rejected direction (mixed fallback) never collapses the pair."""
        table = ReferenceEventTable(htst_config)
        fwd, bwd = _series(table, system_single_type_fcc, 0, 0.5, 0.5)
        admission = table._admit_series(fwd, bwd).ok_value()
        table.add(admission.frame)
        table.finalize_self_reverse(
            0, 1, self_reverse_prefactors_agree(accepted(5.0e12), rejected())
        )
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


class TestTrueSelfReverse:
    """A genuinely self-reverse event collapses once its spectra agree."""

    def test_agreeing_prefactors_collapse_and_relink(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """Collapse drops the backward row and re-points links made to it."""
        table = ReferenceEventTable(htst_config)
        fwd, bwd = _series(table, system_single_type_fcc, 0, 0.5, 0.5)
        admission = table._admit_series(fwd, bwd).ok_value()
        table.add(admission.frame)  # ids 0 (forward) and 1 (backward)

        # A later event whose reverse matched the temporary backward row.
        other, _ = _series(table, system_single_type_fcc, 5, 1.0, 1.0)
        other["event_id"] = "OTHER"
        _insert(table, other, idx_ref=2, idx_backward=1)

        agree = self_reverse_prefactors_agree(accepted(5.0e12), accepted(5.01e12))
        assert agree
        table.finalize_self_reverse(0, 1, agree)

        assert _links(table) == [(0, 0), (2, 0)]
        assert table.max_idx_ref() == 3  # ids stay logical; the catalogue is sparse

    def test_finalize_unknown_pair_raises(self, htst_config: Any) -> None:
        """Finalising ids that are not in the table is a programming error."""
        table = ReferenceEventTable(htst_config)
        with pytest.raises(ValueError, match="not in the reference table"):
            table.finalize_self_reverse(0, 1, agree=True)


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

    def test_htst_links_to_the_matched_logical_id(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """The lone forward row links to the catalogued reverse, not to itself."""
        table = ReferenceEventTable(htst_config)
        fwd, bwd = self._catalogue_with_reverse(table, system_single_type_fcc)

        assert table.find_matching_event(bwd) == 7
        assert table.find_matching_event(fwd) is None
        admission = table._admit_series(fwd, bwd).ok_value()
        assert admission.reverse_idx_ref == 7
        assert admission.self_reverse_candidate is False
        assert len(admission.frame) == 1

        table.add(admission.frame, reverse_idx_ref=admission.reverse_idx_ref)
        assert _links(table) == [(7, 7), (8, 7)]

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
        table = ReferenceEventTable(htst_config)
        fwd, bwd = self._catalogue_with_reverse(table, system_single_type_fcc)
        admission = table._admit_series(fwd, bwd).ok_value()
        table.add(admission.frame, reverse_idx_ref=admission.reverse_idx_ref)
        assert _links(table) == [(7, 7), (8, 7)]

        table.remove([7])
        assert _links(table) == []

    def test_remove_forward_drops_its_reverse(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """Removing the forward row removes the reverse it links to (base rule)."""
        table = ReferenceEventTable(htst_config)
        fwd, bwd = self._catalogue_with_reverse(table, system_single_type_fcc)
        admission = table._admit_series(fwd, bwd).ok_value()
        table.add(admission.frame, reverse_idx_ref=admission.reverse_idx_ref)
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

    def test_mismatching_saddle_crops_are_not_a_candidate(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """Equal topologies with non-matching saddle crops keep two plain rows."""
        table = ReferenceEventTable(htst_config)
        fwd, bwd = _series(table, system_single_type_fcc, 0, 0.5, 0.5)
        rng = np.random.default_rng(7)
        bwd["saddle_positions"] = np.asarray(bwd["saddle_positions"]) + rng.uniform(
            -1.5, 1.5, size=np.asarray(bwd["saddle_positions"]).shape
        )
        admission = table._admit_series(fwd, bwd).ok_value()
        assert admission.self_reverse_candidate is False
        assert len(admission.frame) == 2

    @pytest.mark.parametrize("coloring,expected", [("full", False), ("grey", True)])
    def test_species_swapped_saddle_crops_follow_the_colouring_mode(
        self, htst_config: Any, system_binary_fcc: Any, coloring: str, expected: bool
    ) -> None:
        """Ni<->Fe swapped directional crops are a candidate only in grey mode.

        ``_saddle_crops_match`` feeds IRA the same labels as
        ``find_matching_event``: the local element types in ``full``
        colouring, one grey label otherwise.
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
        assert admission.self_reverse_candidate is expected
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
