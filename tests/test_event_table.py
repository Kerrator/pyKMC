"""Unit tests for full-colour typing in the reference event table."""

import numpy as np
import pandas as pd
import pytest

from pykmc import NeighborsList
from pykmc.atomic_environment import AtomicEnvironment
from pykmc.config import RateConstantConfig
from pykmc.event_table import ActiveEventTable, ReferenceEventTable
from pykmc.result import EventSearchOutput

S0_REFERENCE_COLUMNS = [
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
"""Constant-mode reference schema recorded at S0 (contracts section 7)."""

S0_ACTIVE_COLUMNS = [
    "atom_index",
    "saddle_positions",
    "final_positions",
    "energy_barrier",
    "k",
    "num_reference_event",
    "refined",
]
"""Constant-mode active schema recorded at S0 (contracts section 7)."""


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


class TestReferenceTablePeriodicity:
    """The catalogue builds its scratch systems on the real periodic axes.

    Regression for review cluster C2 (R05 ``f7bc418``): an unspecified
    ``System`` is non-periodic, so a bare scratch system truncated every
    catalogued environment at the cell faces while the runtime ids stayed
    periodic. Atom 0 of the 4x4x4 fixture sits at the cell corner, so its
    rcut shell wraps across all three faces.
    """

    @staticmethod
    def _trivial_event(system):
        pos = np.asarray(system.positions, dtype=float)
        return EventSearchOutput(
            central_atom_index=0,
            min1_positions=pos.copy(),
            saddle_positions=pos.copy(),
            min2_positions=pos.copy(),
            dE_forward=0.5,
            dE_backward=0.5,
            move_atom_index=0,
            cell=np.asarray(system.cell, dtype=float),
            types=list(system.types),
        )

    @pytest.mark.parametrize("mode", ["grey", "full"])
    def test_catalogue_event_id_equals_runtime_environment_id(
        self, system_binary_fcc, config_system_single_type, mode
    ):
        """``add_events`` threads the system axes: catalogue id == runtime id."""
        config = config_system_single_type
        config.atomicenvironment.atom_coloring_mode = mode
        system = system_binary_fcc
        assert list(system.pbc) == [True, True, True]
        nl = NeighborsList(
            system, config.atomicenvironment.rnei, config.atomicenvironment.rcut
        )
        runtime = AtomicEnvironment(
            "graph",
            nl.neighbors_list["rnei"],
            nl.neighbors_list["rcut"],
            types=system.types,
            coloring_mode=mode,
        )
        runtime_id = runtime.atomic_environment_list[0]
        runtime_shell = list(nl.neighbors_list["rcut"][0])
        # The corner atom really crosses the boundary: an open shell is smaller.
        open_system = type(system)(
            positions=system.positions, cell=system.cell, pbc=[False, False, False]
        )
        open_shell = NeighborsList(
            open_system,
            config.atomicenvironment.rnei,
            config.atomicenvironment.rcut,
        ).neighbors_list["rcut"][0]
        assert len(open_shell) < len(runtime_shell)

        table = ReferenceEventTable(config)
        results = table.add_events([self._trivial_event(system)], pbc=system.pbc)
        assert results[0].is_ok()
        row = table.table.iloc[0]
        assert row["event_id"] == runtime_id
        assert row["id_final"] == runtime_id
        assert len(row["initial_positions"]) == len(runtime_shell)
        assert list(row["types"]) == list(np.asarray(system.types)[runtime_shell])

    def test_open_axes_are_honoured(self, system_binary_fcc, config_system_single_type):
        """Explicit open axes give the open-boundary crop, so ``pbc`` is consumed."""
        config = config_system_single_type
        config.atomicenvironment.atom_coloring_mode = "grey"
        system = system_binary_fcc
        table = ReferenceEventTable(config)
        pos = system.positions
        kwargs = dict(
            min1_positions=pos,
            saddle_positions=pos,
            min2_positions=pos,
            index_move=0,
            dE_forward=0.5,
            dE_backward=0.5,
            cell=system.cell,
            types=list(system.types),
        )
        periodic, _ = table._build_event_series(pbc=system.pbc, **kwargs)
        open_, _ = table._build_event_series(pbc=[False, False, False], **kwargs)
        default, _ = table._build_event_series(**kwargs)
        assert len(open_["initial_positions"]) < len(periodic["initial_positions"])
        assert open_["event_id"] != periodic["event_id"]
        # An unspecified ``pbc`` keeps the catalogue's all-periodic convention.
        assert default["event_id"] == periodic["event_id"]
        assert len(default["initial_positions"]) == len(periodic["initial_positions"])


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


class TestSchemaPins:
    """Serialized schemas: constant mode equals S0; htst/rpa append their columns."""

    def test_constant_schemas_equal_s0(self, config_system_single_type: object) -> None:
        """The constant tables expose exactly the S0 columns, in order."""
        config = config_system_single_type
        assert config.rateconstant.style == "constant"
        assert list(ReferenceEventTable(config).table.columns) == S0_REFERENCE_COLUMNS
        assert list(ActiveEventTable(config).table.columns) == S0_ACTIVE_COLUMNS

    @pytest.mark.parametrize("style", ["htst", "rpa"])
    def test_prefactor_schemas_append_columns(
        self, style: str, config_system_single_type: object
    ) -> None:
        """htst/rpa add four reference columns and six active columns after S0."""
        config = config_system_single_type.model_copy(
            update={"rateconstant": RateConstantConfig(style=style, k0=1.0, T=300.0)}
        )
        assert list(ReferenceEventTable(config).table.columns) == (
            S0_REFERENCE_COLUMNS + ["k_prefactor", "nu0", "nu0_status", "nu0_reason"]
        )
        assert list(ActiveEventTable(config).table.columns) == (
            S0_ACTIVE_COLUMNS
            + [
                "k_prefactor",
                "nu0",
                "nu0_status",
                "nu0_reason",
                "nu0_source",
                "nu0_site_attempted",
            ]
        )
