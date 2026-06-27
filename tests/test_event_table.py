"""Unit tests for full-colour typing in the reference event table."""

import numpy as np
import pandas as pd

from pykmc import NeighborsList
from pykmc.event_table import ReferenceEventTable


def _build_trivial_series(config, system):
    """Run ``_build_event_series`` for a trivial (min1==saddle==min2) event.

    This exercises the graph/symmetry/type-storage path without going through
    barrier validation, which is unrelated to colouring.
    """
    table = ReferenceEventTable(config)
    pos = system.positions
    return table._build_event_series(
        min1_positions=pos,
        saddle_positions=pos,
        min2_positions=pos,
        index_move=0,
        dE_forward=0.5,
        dE_backward=0.5,
        cell=system.cell,
        types=list(system.types),
    )


class TestReferenceTableTypes:
    """The ``types`` column is always populated, so the schema is mode-independent."""

    def test_types_stored_in_both_modes(
        self, system_binary_fcc, config_system_single_type
    ):
        """Both grey and full store the real forward/backward neighbour element types."""
        config = config_system_single_type

        # Expected forward types = element of each atom in atom 0's rcut shell.
        nl = NeighborsList(
            system_binary_fcc,
            config.atomicenvironment.rnei,
            config.atomicenvironment.rcut,
        )
        forward_neighbors = nl.neighbors_list["rcut"][0]
        expected = list(np.array(system_binary_fcc.types)[forward_neighbors])

        for mode in ("grey", "full"):
            config.atomicenvironment.atom_coloring_mode = mode
            fwd, bwd = _build_trivial_series(config, system_binary_fcc)

            assert fwd["types"] is not None, mode
            assert list(fwd["types"]) == expected, mode
            # A genuine multi-element environment carries both species.
            assert set(fwd["types"]) == {"Ni", "Fe"}, mode
            assert bwd["types"] is not None, mode

    def test_event_id_differs_by_mode_same_types(
        self, system_binary_fcc, config_system_single_type
    ):
        """Full colouring changes the reference hash (event_id), but not the stored types."""
        config = config_system_single_type

        config.atomicenvironment.atom_coloring_mode = "grey"
        grey_fwd, _ = _build_trivial_series(config, system_binary_fcc)

        config.atomicenvironment.atom_coloring_mode = "full"
        full_fwd, _ = _build_trivial_series(config, system_binary_fcc)

        # Colouring changes the hash...
        assert grey_fwd["event_id"] != full_fwd["event_id"]
        # ...but the stored types are identical (storage is mode-independent).
        assert list(grey_fwd["types"]) == list(full_fwd["types"])

    def test_types_aligned_with_positions(
        self, system_binary_fcc, config_system_single_type
    ):
        """``types`` is element-wise aligned with the position slices (the ordering invariant).

        Equal length alone is tautological (everything is sliced by the same index
        array), so we tie both the type label *and* the position row of each entry
        back to the same global atom index. For this trivial event
        (min1==saddle==min2==system.positions) every position slice reduces to
        ``system.positions[neighbour_list]``.
        """
        config = config_system_single_type
        config.atomicenvironment.atom_coloring_mode = "full"

        nl = NeighborsList(
            system_binary_fcc,
            config.atomicenvironment.rnei,
            config.atomicenvironment.rcut,
        )
        forward_neighbors = nl.neighbors_list["rcut"][0]

        fwd, _ = _build_trivial_series(config, system_binary_fcc)

        n = len(forward_neighbors)
        assert len(fwd["types"]) == n
        # Each entry's type and its three position rows all map to the same atom.
        for k, atom in enumerate(forward_neighbors):
            assert fwd["types"][k] == system_binary_fcc.types[atom]
            for key in ("initial_positions", "saddle_positions", "final_positions"):
                assert np.array_equal(fwd[key][k], system_binary_fcc.positions[atom]), (
                    key,
                    k,
                )


class TestGreyDedupKnownLimitation:
    """Characterize the *deferred* grey-mode de-dup limitation in ``is_new_event``.

    These are KNOWN-LIMITATION characterization tests, not specification tests.
    ``is_new_event`` deliberately compares the REAL element types in IRA in BOTH
    coloring modes (grey gating is a documented deferred limitation -- see the
    ``atom_coloring_mode`` field description in ``config.py``). Consequently, in
    grey mode two events that share a (species-blind) ``event_id`` but carry
    swapped species are treated as DISTINCT. This pins that current behaviour so a
    future grey-gating fix (forcing types to ``'X'`` in this path) trips the test
    deliberately and prompts a conscious update.
    """

    def _insert_forward_row(self, table: ReferenceEventTable, fwd: pd.Series) -> None:
        """Append a single forward event ``Series`` as a row of ``table.table``.

        Parameters
        ----------
        table : ReferenceEventTable
            The reference table whose ``table`` DataFrame is appended to.
        fwd : pd.Series
            The forward event series to insert as a single row.

        """
        fwd = fwd.copy()
        fwd["idx_ref"] = 0
        fwd["idx_backward"] = 0
        table.table = pd.concat([table.table, fwd.to_frame().T], ignore_index=True)

    def test_grey_swapped_species_not_merged(
        self, system_binary_fcc: object, config_system_single_type: object
    ) -> None:
        """KNOWN LIMITATION: grey mode does NOT merge swapped-species duplicates.

        A trivial (min1==saddle==min2) event is stored, then an identical-geometry
        event with every Ni<->Fe swapped is queried. The grey ``event_id`` hash is
        species-blind so both share an id and reach the IRA de-dup path; but the
        path feeds the real (swapped) element types to IRA, so the match is
        rejected and ``is_new_event`` reports the swapped event as new.

        Parameters
        ----------
        system_binary_fcc : object
            Binary Ni/Fe FCC ``System`` fixture (alternating species).
        config_system_single_type : object
            Loaded ``Config`` fixture; its coloring mode is set to ``grey`` here.

        """
        config = config_system_single_type
        config.atomicenvironment.atom_coloring_mode = "grey"

        pos = system_binary_fcc.positions
        cell = system_binary_fcc.cell
        types = list(system_binary_fcc.types)
        swapped = ["Fe" if t == "Ni" else "Ni" for t in types]

        table = ReferenceEventTable(config)
        fwd, _ = table._build_event_series(
            min1_positions=pos,
            saddle_positions=pos,
            min2_positions=pos,
            index_move=0,
            dE_forward=0.5,
            dE_backward=0.5,
            cell=cell,
            types=types,
        )
        self._insert_forward_row(table, fwd)

        fwd_swapped, _ = table._build_event_series(
            min1_positions=pos,
            saddle_positions=pos,
            min2_positions=pos,
            index_move=0,
            dE_forward=0.5,
            dE_backward=0.5,
            cell=cell,
            types=swapped,
        )

        # Grey hash is species-blind: identical geometry -> identical event_id,
        # so the swapped event reaches the type-aware IRA de-dup path.
        assert fwd_swapped["event_id"] == fwd["event_id"]
        # ...but the stored/queried element types genuinely differ (storage is
        # mode-independent; nothing is greyed to 'X' here).
        assert list(fwd_swapped["types"]) != list(fwd["types"])

        # KNOWN LIMITATION: species-aware de-dup -> swapped event is NOT merged.
        # If a future change grey-gates this path, IRA would match and this flips
        # to False -- intentionally tripping the test.
        assert table.is_new_event(fwd_swapped) is True

        # Sanity control: the byte-identical event IS recognised as a duplicate.
        assert table.is_new_event(fwd) is False
