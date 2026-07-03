"""Unit tests for full-colour typing in the reference event table."""

import numpy as np
import pandas as pd
import pytest

from pykmc import NeighborsList
from pykmc.event_table import ReferenceEventTable
from pykmc.environments.graph_nauty import combine_ids
import pykmc.point_set_registration as psr_module
from pykmc.point_set_registration import PointSetRegistration


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

    def test_legacy_pickle_is_normalized_on_load(
        self, tmp_path, config_system_single_type
    ):
        config = config_system_single_type
        legacy_table = pd.DataFrame(
            {
                "event_id": [b"ini0", b"ini1"],
                "initial_positions": [
                    np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                    np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
                ],
                "saddle_positions": [
                    np.array([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                    np.array([[0.0, 0.5, 1.0], [1.0, 0.0, 1.0]]),
                ],
                "final_positions": [
                    np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                    np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]),
                ],
                "energy_barrier": [0.3, 0.4],
                "k": [1.0, 2.0],
                "id_saddle": [b"sad0", b"sad1"],
                "id_final": [b"fin0", b"fin1"],
                "move_atom_idx": [0, 0],
                "sym_matrix": [np.array([np.eye(3)]), np.array([np.eye(3)])],
                "sym_perm": [np.array([[0, 1]]), np.array([[0, 1]])],
                "idx_backward": [1, 0],
            }
        )
        pickle_path = tmp_path / "legacy_reference_table.pickle"
        legacy_table.to_pickle(pickle_path)
        config.control.reference_table = str(pickle_path)

        reference_table = ReferenceEventTable(config)
        table = reference_table.table

        assert {
            "idx_ref",
            "id_initial",
            "generic_id_initial",
            "dE_forward",
            "dE_backward",
            "types",
            "dra",
        } <= set(table.columns)
        assert table.loc[0, "idx_ref"] == 0
        assert isinstance(table.loc[0, "id_initial"], str)
        assert table.loc[0, "dE_forward"] == pytest.approx(0.3)
        assert table.loc[0, "dE_backward"] == pytest.approx(0.4)
        assert table.loc[0, "types"] == ["X", "X"]
        assert table.loc[0, "dra"] == pytest.approx(0.5)
        assert table.loc[0, "event_id"] == combine_ids(
            table.loc[0, "id_initial"],
            table.loc[0, "id_saddle"],
            table.loc[0, "id_final"],
        )
        assert isinstance(reference_table.is_new_event(table.loc[0].copy()), bool)
        subset = reference_table.has_id_subset_table([table.loc[0, "id_initial"]])
        assert subset["idx_ref"].tolist() == [0]

    def test_legacy_grey_row_remains_compatible_in_full_mode(
        self, tmp_path, system_binary_fcc, config_system_single_type
    ):
        config = config_system_single_type
        config.atomicenvironment.atom_coloring_mode = "grey"
        legacy_fwd, _ = _build_trivial_series(config, system_binary_fcc)
        legacy_row = legacy_fwd.drop(
            labels=["types", "legacy_untyped"], errors="ignore"
        ).copy()
        legacy_row["idx_ref"] = 0
        legacy_row["idx_backward"] = 0
        legacy_table = pd.DataFrame([legacy_row])
        pickle_path = tmp_path / "legacy_binary_reference_table.pickle"
        legacy_table.to_pickle(pickle_path)

        config.atomicenvironment.atom_coloring_mode = "full"
        config.control.reference_table = str(pickle_path)
        reference_table = ReferenceEventTable(config)

        full_fwd, _ = _build_trivial_series(config, system_binary_fcc)
        assert bool(reference_table.table.loc[0, "legacy_untyped"]) is True

        subset = reference_table.has_id_subset_table(
            [full_fwd["id_initial"]],
            generic_ids=[full_fwd["generic_id_initial"]],
        )
        assert subset["idx_ref"].tolist() == [0]
        assert reference_table.is_new_event(full_fwd) is False

    def test_legacy_full_mode_psr_uses_current_neighbor_types(
        self, tmp_path, system_binary_fcc, config_system_single_type, monkeypatch
    ):
        config = config_system_single_type
        config.atomicenvironment.atom_coloring_mode = "grey"
        legacy_fwd, _ = _build_trivial_series(config, system_binary_fcc)
        legacy_row = legacy_fwd.drop(
            labels=["types", "legacy_untyped"], errors="ignore"
        ).copy()
        legacy_table = pd.DataFrame([legacy_row])
        pickle_path = tmp_path / "legacy_psr_reference_table.pickle"
        legacy_table.to_pickle(pickle_path)

        config.atomicenvironment.atom_coloring_mode = "full"
        config.control.reference_table = str(pickle_path)
        reference_table = ReferenceEventTable(config)
        nl = NeighborsList(
            system_binary_fcc,
            config.atomicenvironment.rnei,
            config.atomicenvironment.rcut,
        )
        captured = {}

        class DummyIRA:
            def match(self, nat1, typ1, coords1, nat2, typ2, coords2, kmax_factor):
                captured["typ1"] = list(typ1)
                captured["typ2"] = list(typ2)
                return np.eye(3), np.zeros(3), np.arange(nat2), 0.0

        monkeypatch.setattr(psr_module.ira_mod, "IRA", lambda: DummyIRA())
        result = PointSetRegistration(
            config,
            system_binary_fcc,
            reference_table.table.loc[0],
            nl,
            0,
        ).match()

        assert result.is_ok()
        assert captured["typ2"] == captured["typ1"]


class TestGreyDedupSpeciesGating:
    """Specify the coloring-mode gating of de-dup in ``is_new_event``.

    ``is_new_event`` mirrors the PSR / classification paths: in "full" mode it
    feeds the REAL element types to IRA, so geometrically-identical but
    species-swapped saddles stay DISTINCT; in "grey" mode every atom is greyed to a
    single dummy label (``'X'``), so the same swapped saddles de-duplicate as one
    event (grey-alloy approximation). These tests pin both sides of that gate.
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

    def test_grey_swapped_species_merged(
        self, system_binary_fcc: object, config_system_single_type: object
    ) -> None:
        """Grey mode merges swapped-species, geometrically-identical duplicates.

        A trivial (min1==saddle==min2) event is stored, then an identical-geometry
        event with every Ni<->Fe swapped is queried. The grey ``event_id`` hash is
        species-blind so both share an id and reach the IRA de-dup path; that path
        greys every atom to ``'X'``, so the geometries match and ``is_new_event``
        reports the swapped event as a duplicate.

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
        # so the swapped event reaches the (grey-gated) IRA de-dup path.
        assert fwd_swapped["event_id"] == fwd["event_id"]
        # ...the stored/queried element types still genuinely differ (storage is
        # mode-independent; greying to 'X' happens inside is_new_event).
        assert list(fwd_swapped["types"]) != list(fwd["types"])

        # Grey-gated de-dup: types are greyed to 'X', so the swapped event MERGES.
        assert table.is_new_event(fwd_swapped) is False

        # Sanity control: the byte-identical event IS recognised as a duplicate.
        assert table.is_new_event(fwd) is False

    def test_full_swapped_species_not_merged(
        self, system_binary_fcc: object, config_system_single_type: object
    ) -> None:
        """Full mode keeps swapped-species saddles distinct (species-aware de-dup).

        Same geometry as the grey case, but in "full" mode ``is_new_event`` feeds
        the real (swapped) element types to IRA, so the match is rejected and the
        swapped event is reported as new.

        Parameters
        ----------
        system_binary_fcc : object
            Binary Ni/Fe FCC ``System`` fixture (alternating species).
        config_system_single_type : object
            Loaded ``Config`` fixture; its coloring mode is set to ``full`` here.

        """
        config = config_system_single_type
        config.atomicenvironment.atom_coloring_mode = "full"

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

        # Full mode compares real element types (species-aware event_id and/or
        # type-aware IRA), so the swapped saddle is not merged -> reported as new.
        assert table.is_new_event(fwd_swapped) is True

        # Sanity control: the byte-identical event IS recognised as a duplicate.
        assert table.is_new_event(fwd) is False
