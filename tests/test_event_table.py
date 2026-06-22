"""Unit tests for full-colour typing in the reference event table."""

import numpy as np

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


class TestReferenceTableInitialTypes:
    """The ``initial_types`` column is populated only in full coloring mode."""

    def test_full_mode_stores_initial_types(
        self, system_binary_fcc, config_system_single_type
    ):
        """Full mode records the forward/backward neighbour element types."""
        config = config_system_single_type
        config.atomicenvironment.atom_coloring_mode = "full"

        fwd, bwd = _build_trivial_series(config, system_binary_fcc)

        # Expected forward types = element of each atom in atom 0's rcut shell.
        nl = NeighborsList(
            system_binary_fcc,
            config.atomicenvironment.rnei,
            config.atomicenvironment.rcut,
        )
        forward_neighbors = nl.neighbors_list["rcut"][0]
        expected = list(np.array(system_binary_fcc.types)[forward_neighbors])

        assert fwd["initial_types"] is not None
        assert list(fwd["initial_types"]) == expected
        # A genuine multi-element environment carries both species.
        assert set(fwd["initial_types"]) == {"Ni", "Fe"}
        assert bwd["initial_types"] is not None

    def test_grey_mode_stores_no_initial_types(
        self, system_binary_fcc, config_system_single_type
    ):
        """Grey mode leaves ``initial_types`` as None even when types are supplied."""
        config = config_system_single_type
        config.atomicenvironment.atom_coloring_mode = "grey"

        fwd, bwd = _build_trivial_series(config, system_binary_fcc)

        assert fwd["initial_types"] is None
        assert bwd["initial_types"] is None
