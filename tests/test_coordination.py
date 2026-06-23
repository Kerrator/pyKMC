from collections import Counter

import pytest
from pytest_lazy_fixtures import lf

from pykmc import AtomicEnvironment, NeighborsList


class TestCoordination:
    @pytest.mark.parametrize(
        "system, config", [(lf("system_single_type_fcc"), lf("config_system_single_type"))]
    )
    def test_coordination_bulk_fcc(self, system, config):
        """All atoms in a perfect FCC bulk should be crystal (12 neighbours)."""
        nl = NeighborsList(system, config.atomicenvironment.rnei, config.atomicenvironment.rcut)
        ae = AtomicEnvironment("coordination", nl.neighbors_list["rnei"], coordination_threshold=12)
        assert all(e == "crystal" for e in ae.atomic_environment_list)

    @pytest.mark.parametrize(
        "system, config", [(lf("system_single_type_fcc_vacancy"), lf("config_system_single_type"))]
    )
    def test_coordination_vacancy(self, system, config):
        """FCC with a vacancy: the 12 vacancy neighbours become noncrystal (<12 neighbours)."""
        nl = NeighborsList(system, config.atomicenvironment.rnei, config.atomicenvironment.rcut)
        ae = AtomicEnvironment("coordination", nl.neighbors_list["rnei"], coordination_threshold=12)
        hash_count = Counter(ae.atomic_environment_list)
        assert "noncrystal" in hash_count
        assert hash_count["noncrystal"] == 12

    @pytest.mark.parametrize(
        "system, config", [(lf("system_single_type_fcc_vacancy"), lf("config_system_single_type"))]
    )
    def test_coordination_graph_hashes_only_noncrystal(self, system, config):
        """coordination/graph: bulk atoms stay 'crystal'; the 12 under-coordinated atoms get graph IDs."""
        nl = NeighborsList(system, config.atomicenvironment.rnei, config.atomicenvironment.rcut)
        ae = AtomicEnvironment(
            "coordination/graph",
            nl.neighbors_list["rnei"],
            nl.neighbors_list["rcut"],
            coordination_threshold=12,
        )
        ids = ae.atomic_environment_list
        # Exactly the 12 vacancy neighbours are replaced by a graph hash (not the literal "crystal").
        assert Counter(ids)["crystal"] == len(ids) - 12
        non_crystal = [e for e in ids if e != "crystal"]
        assert len(non_crystal) == 12
        assert all(e != "noncrystal" for e in non_crystal)  # replaced by graph hashes, not left as "noncrystal"
