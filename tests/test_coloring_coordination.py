"""Integration test: atom colouring must reach the 'coordination/graph' style.

The 'coordination/graph' style classifies atoms by nearest-neighbour count, then
computes a nauty graph ID for every non-crystal atom. The combined branch wires
PR #90's colouring into that graph call, so with ``coloring_mode='full'`` the graph
IDs of a multi-species environment must be coloured by element type. This proves the
colour/coordination integration: full colouring yields strictly more distinct graph
IDs than the grey (species-blind) baseline.
"""

from pykmc import AtomicEnvironment, Config, NeighborsList


def test_full_colouring_reaches_coordination_graph(system_binary_fcc, config_system_single_type):
    """Full colouring distinguishes Ni/Fe environments under 'coordination/graph'."""
    config = config_system_single_type
    nl = NeighborsList(
        system_binary_fcc,
        config.atomicenvironment.rnei,
        config.atomicenvironment.rcut,
    )
    types = list(system_binary_fcc.types)

    # Perfect FCC is 12-coordinated everywhere; threshold=13 makes every atom
    # 'noncrystal', so every atom receives a graph ID (otherwise grey vs full
    # would be a trivial no-op with no non-crystal atoms to colour).
    grey = AtomicEnvironment(
        "coordination/graph",
        nl.neighbors_list["rnei"],
        nl.neighbors_list["rcut"],
        coordination_threshold=13,
        types=types,
        coloring_mode="grey",
    )
    full = AtomicEnvironment(
        "coordination/graph",
        nl.neighbors_list["rnei"],
        nl.neighbors_list["rcut"],
        coordination_threshold=13,
        types=types,
        coloring_mode="full",
    )

    grey_ids = grey.atomic_environment_list
    full_ids = full.atomic_environment_list

    # Sanity: threshold=13 left no crystal atoms, so every atom carries a graph ID.
    assert "crystal" not in grey_ids
    assert "crystal" not in full_ids

    # Colouring reaches coordination/graph: at least one atom's graph ID changes
    # between grey and full, and full produces strictly more distinct IDs.
    assert any(g != f for g, f in zip(grey_ids, full_ids))
    assert len(set(full_ids)) > len(set(grey_ids))
