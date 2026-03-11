import numpy as np

from pykmc import NeighborsList, System
from pykmc.basins import BasinsGenericEvents


def _periodic_two_atom_system(positions: np.ndarray) -> System:
    return System(
        positions=np.array(positions, dtype=float),
        types=np.array(["Ni", "Ni"]),
        cell=np.diag([10.0, 10.0, 10.0]),
        pbc=np.array([True, True, True]),
        index=np.array([0, 1]),
    )


def test_neighbors_list_accepts_out_of_box_periodic_positions() -> None:
    system = _periodic_two_atom_system([[0.0, 0.0, 0.0], [10.2, 0.0, 0.0]])
    nl = NeighborsList(system=system, rnei=0.5, rcut=1.0)

    assert 1 in nl.get_neighbors("rnei", 0)
    assert 0 in nl.get_neighbors("rnei", 1)


def test_are_structures_equivalent_wraps_periodic_coordinates() -> None:
    basin = BasinsGenericEvents(
        config=None,
        reference_table=None,
        known_environments=set(),
        manager=None,
    )

    cell = np.diag([10.0, 10.0, 10.0])
    pos1 = np.array([[9.9, 1.0, 1.0], [0.2, 0.2, 0.2]])
    pos2 = np.array([[-0.1, 1.0, 1.0], [10.2, 0.2, 0.2]])

    assert basin.are_structures_equivalent(
        pos1=pos1,
        pos2=pos2,
        cell=cell,
        pbc=np.array([True, True, True]),
        tol=1e-8,
    )
