from pykmc import System, Config, NeighborsList
import pytest
from pytest_lazy_fixtures import lf
import numpy as np

expected_neighbors = {
    "system_single_type_fcc": 12, 
}


class TestNeighborsList : 


    @pytest.mark.parametrize("system_name, system, config", [("system_single_type_fcc", lf("system_single_type_fcc"), lf("config_system_single_type"))])
    def test_get_neighbors_list(self, system_name: str, system: System, config: Config) :
        nl = NeighborsList(system, config.atomicenvironment.rnei, config.atomicenvironment.rcut)

        expected_nb_neighbors = expected_neighbors[system_name]

        for i in range(len(system.types)) : 
            neighbors = nl.get_neighbors('rnei', i)
            assert len(neighbors) == expected_nb_neighbors

    def test_mixed_pbc_keeps_self_only_in_rcut(self):
        system = System(
            positions=np.array([[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 1.0, -1.0]], dtype=float),
            types=np.array(["Ni", "Ni", "Ni"]),
            cell=np.diag([10.0, 10.0, 20.0]),
            pbc=np.array([True, True, False]),
            index=np.array([0, 1, 2]),
        )

        nl = NeighborsList(system, rnei=1.5, rcut=2.5)

        rnei_neighbors = nl.get_neighbors("rnei", 0)
        rcut_neighbors = nl.get_neighbors("rcut", 0)

        assert 0 not in rnei_neighbors
        assert rcut_neighbors.count(0) == 1
        assert np.where(np.array(rcut_neighbors) == 0)[0][0] >= 0

