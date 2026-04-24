from pykmc import System, Config, NeighborsList
from pykmc.config import AtomicEnvironmentConfig
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


class TestGraphCutoff:

    def test_no_graph_cutoff_unchanged(self, system_binary_fcc):
        """Without graph_cutoff, all atoms have 12 neighbors (FCC)."""
        nl = NeighborsList(system_binary_fcc, rnei=3.0, graph_cutoff=None)
        for i in range(len(system_binary_fcc.types)):
            assert len(nl.get_neighbors("rnei", i)) == 12

    def test_pair_cutoff_prunes_cross_element_bonds(self, system_binary_fcc):
        """A very short Ni-Fe cutoff prunes cross-element bonds."""
        # FCC NN distance is a/sqrt(2) = 3.52/1.414 ≈ 2.49 Å
        # Set Ni-Fe cutoff below NN distance to prune all Ni-Fe bonds
        graph_cutoff = {"Fe-Ni": 2.0}  # below 2.49, so no Ni-Fe neighbors
        nl = NeighborsList(system_binary_fcc, rnei=3.0, graph_cutoff=graph_cutoff)

        types = list(system_binary_fcc.types)
        for i in range(len(types)):
            neighbors = nl.get_neighbors("rnei", i)
            for j in neighbors:
                # No cross-element bonds should remain
                pair = sorted([types[i], types[j]])
                assert pair != ["Fe", "Ni"], (
                    f"Atom {i} ({types[i]}) should not have atom {j} ({types[j]}) as neighbor"
                )

    def test_pair_cutoff_fallback_to_rnei(self, system_binary_fcc):
        """Pairs not listed in graph_cutoff use rnei as fallback."""
        # Only specify Ni-Ni cutoff, Fe-Fe and Ni-Fe should use rnei=3.0
        graph_cutoff = {"Ni-Ni": 3.0}
        nl_with = NeighborsList(system_binary_fcc, rnei=3.0, graph_cutoff=graph_cutoff)
        nl_without = NeighborsList(system_binary_fcc, rnei=3.0, graph_cutoff=None)

        types = list(system_binary_fcc.types)
        for i in range(len(types)):
            # Fe atoms have no Ni-Ni neighbors, so their neighbor lists should be identical
            if types[i] == "Fe":
                assert set(nl_with.get_neighbors("rnei", i)) == set(nl_without.get_neighbors("rnei", i))

    def test_rcut_unaffected_by_graph_cutoff(self, system_binary_fcc):
        """rcut neighbors are not affected by graph_cutoff."""
        graph_cutoff = {"Fe-Ni": 2.0}  # prunes rnei Ni-Fe bonds
        nl = NeighborsList(system_binary_fcc, rnei=3.0, rcut=6.5, graph_cutoff=graph_cutoff)
        nl_ref = NeighborsList(system_binary_fcc, rnei=3.0, rcut=6.5, graph_cutoff=None)

        for i in range(len(system_binary_fcc.types)):
            assert set(nl.get_neighbors("rcut", i)) == set(nl_ref.get_neighbors("rcut", i))

    def test_mixed_pbc_with_graph_cutoff(self):
        """Pair-specific cutoffs work with mixed PBC (ghost images)."""
        system = System(
            positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
            types=np.array(["Ni", "Fe"]),
            cell=np.diag([10.0, 10.0, 20.0]),
            pbc=np.array([True, True, False]),
            index=np.array([0, 1]),
        )
        # Distance is 1.0 Å. Cutoff 0.5 should prune, 1.5 should keep.
        nl_pruned = NeighborsList(system, rnei=1.5, graph_cutoff={"Fe-Ni": 0.5})
        nl_kept = NeighborsList(system, rnei=1.5, graph_cutoff={"Fe-Ni": 1.5})

        assert len(nl_pruned.get_neighbors("rnei", 0)) == 0
        assert len(nl_kept.get_neighbors("rnei", 0)) == 1


class TestGraphCutoffConfig:

    def test_parse_string(self):
        """INI-style string is parsed into normalized dict."""
        config = AtomicEnvironmentConfig(
            style="graph", rnei=3.0, rcut=6.5,
            atom_coloring_mode="full",
            graph_cutoff="Ni-Ni:3.0, Ni-Fe:2.7, Fe-Fe:2.8",
        )
        assert config.graph_cutoff == {"Ni-Ni": 3.0, "Fe-Ni": 2.7, "Fe-Fe": 2.8}

    def test_key_normalization(self):
        """Keys are sorted alphabetically (Fe-Ni, not Ni-Fe)."""
        config = AtomicEnvironmentConfig(
            style="graph", rnei=3.0, rcut=6.5,
            atom_coloring_mode="full",
            graph_cutoff="Ni-Fe:2.7",
        )
        assert "Fe-Ni" in config.graph_cutoff
        assert "Ni-Fe" not in config.graph_cutoff

    def test_requires_full_coloring_mode(self):
        """graph_cutoff raises error when atom_coloring_mode is grey."""
        with pytest.raises(Exception):
            AtomicEnvironmentConfig(
                style="graph", rnei=3.0, rcut=6.5,
                atom_coloring_mode="grey",
                graph_cutoff="Ni-Ni:3.0",
            )

    def test_none_by_default(self):
        """graph_cutoff defaults to None."""
        config = AtomicEnvironmentConfig(style="graph", rnei=3.0)
        assert config.graph_cutoff is None

    def test_malformed_string_raises(self):
        """Malformed graph_cutoff string raises ValueError."""
        with pytest.raises(Exception):
            AtomicEnvironmentConfig(
                style="graph", rnei=3.0, rcut=6.5,
                atom_coloring_mode="full",
                graph_cutoff="Ni-Fe",  # missing :value
            )

