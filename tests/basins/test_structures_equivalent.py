"""Unit tests for colour-aware basin state de-duplication.

``BasinsGenericEvents.are_structures_equivalent`` decides whether two explored
states are the same. In ``full`` coloring mode a species swap (same geometry,
different element arrangement) must be treated as a *distinct* state; in ``grey``
mode the two must merge. These run without an MPI session pool.
"""

import numpy as np

from pykmc import System
from pykmc.basins import BasinsGenericEvents, StateData


def _make_basin(config):
    """Build a BasinsGenericEvents that only needs ``self.config`` (no MPI)."""
    return BasinsGenericEvents(
        config=config,
        reference_table=None,
        known_environments=set(),
        manager=None,
    )


def _make_system(types):
    """Two-atom System with a diagonal box, parametrised by species."""
    system = System()
    system.positions = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
    system.cell = np.diag([10.0, 10.0, 10.0])
    system.types = np.array(types)
    return system


class TestStructuresEquivalentColoring:
    """full splits the Fe/Ni swap case; grey merges it."""

    def test_swap_merges_in_grey_splits_in_full(self, config_system_single_type):
        """Same positions, swapped species: equivalent in grey, distinct in full."""
        config = config_system_single_type
        cell = np.diag([10.0, 10.0, 10.0])
        pos = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
        types_a = np.array(["Ni", "Fe"])
        types_b = np.array(["Fe", "Ni"])

        basin = _make_basin(config)

        config.atomicenvironment.atom_coloring_mode = "grey"
        assert basin.are_structures_equivalent(
            pos, pos, cell, types1=types_a, types2=types_b
        )

        config.atomicenvironment.atom_coloring_mode = "full"
        assert not basin.are_structures_equivalent(
            pos, pos, cell, types1=types_a, types2=types_b
        )

    def test_identical_types_equivalent_in_full(self, config_system_single_type):
        """In full mode, identical geometry *and* identical species still merge."""
        config = config_system_single_type
        config.atomicenvironment.atom_coloring_mode = "full"
        cell = np.diag([10.0, 10.0, 10.0])
        pos = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
        types = np.array(["Ni", "Fe"])

        basin = _make_basin(config)
        assert basin.are_structures_equivalent(
            pos, typ1=types, pos, typ2=types, cell
        )


class TestIsNewStateColoring:
    """is_new_state — the production dedup path — must thread types into the check.

    Guards against a regression that drops ``types1``/``types2`` at the call site:
    with the defaults (None, None) the full-mode guard never fires and swapped
    states would silently merge again.
    """

    def test_swap_state_new_in_full_merges_in_grey(self, config_system_single_type):
        """Same geometry, swapped species: a new state in full, an existing one in grey."""
        config = config_system_single_type
        basin = _make_basin(config)

        existing = _make_system(["Ni", "Fe"])
        basin.states = {
            0: StateData(system=existing, environment=None, neighbors_list=None)
        }

        query = _make_system(["Fe", "Ni"])  # identical positions, swapped species

        config.atomicenvironment.atom_coloring_mode = "grey"
        assert basin.is_new_state(query) == 0  # merges with the existing state

        config.atomicenvironment.atom_coloring_mode = "full"
        assert basin.is_new_state(query) == -1  # distinct -> reported as new
