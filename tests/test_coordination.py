from collections import Counter

import numpy as np
import pytest
from pytest_lazy_fixtures import lf

from pykmc import AtomicEnvironment, NeighborsList


class TestCoordination:
    @pytest.mark.parametrize(
        "system, config",
        [(lf("system_single_type_fcc"), lf("config_system_single_type"))],
    )
    def test_coordination_bulk_fcc(self, system, config):
        """All atoms in a perfect FCC bulk should be crystal (12 neighbours)."""
        nl = NeighborsList(
            system, config.atomicenvironment.rnei, config.atomicenvironment.rcut
        )
        ae = AtomicEnvironment(
            "coordination", nl.neighbors_list["rnei"], coordination_threshold=12
        )
        assert all(e == "crystal" for e in ae.atomic_environment_list)

    @pytest.mark.parametrize(
        "system, config",
        [(lf("system_single_type_fcc_vacancy"), lf("config_system_single_type"))],
    )
    def test_coordination_vacancy(self, system, config):
        """FCC with a vacancy: the 12 vacancy neighbours become noncrystal (<12 neighbours)."""
        nl = NeighborsList(
            system, config.atomicenvironment.rnei, config.atomicenvironment.rcut
        )
        ae = AtomicEnvironment(
            "coordination", nl.neighbors_list["rnei"], coordination_threshold=12
        )
        hash_count = Counter(ae.atomic_environment_list)
        assert "noncrystal" in hash_count
        assert hash_count["noncrystal"] == 12

    @pytest.mark.parametrize(
        "system, config",
        [(lf("system_single_type_fcc_vacancy"), lf("config_system_single_type"))],
    )
    def test_coordination_graph_hashes_only_noncrystal(self, system, config):
        """coordination/graph: bulk atoms stay 'crystal'; the 12 under-coordinated atoms get graph IDs."""
        nl = NeighborsList(
            system, config.atomicenvironment.rnei, config.atomicenvironment.rcut
        )
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
        assert all(
            e != "noncrystal" for e in non_crystal
        )  # replaced by graph hashes, not left as "noncrystal"

    @pytest.mark.parametrize(
        "system, config",
        [(lf("system_single_type_fcc_vacancy"), lf("config_system_single_type"))],
    )
    def test_coordination_threshold_threaded_from_config(self, system, config):
        """Honour coordination_threshold when args are built from the config object.

        Mirrors the run-site construction path, unlike test_coordination_vacancy.
        """
        config.atomicenvironment.style = "coordination"
        config.atomicenvironment.coordination_threshold = 12
        nl = NeighborsList(
            system, config.atomicenvironment.rnei, config.atomicenvironment.rcut
        )
        ae = AtomicEnvironment(
            config.atomicenvironment.style,
            nl.neighbors_list["rnei"],
            nl.neighbors_list["rcut"],
            config.atomicenvironment.neighbors_add,
            coordination_threshold=config.atomicenvironment.coordination_threshold,
        )
        assert ae.coordination_threshold == 12
        assert sum(e == "noncrystal" for e in ae.atomic_environment_list) == 12

    @pytest.mark.parametrize(
        "system, config",
        [(lf("system_binary_fcc"), lf("config_system_single_type"))],
    )
    def test_coordination_graph_honours_full_coloring(self, system, config):
        """Binary vacancy graphs differ between grey and full coloring in coordination/graph."""
        vacancy_system = type(system)()
        vacancy_system.cell = np.array(system.cell, copy=True)
        vacancy_system.pbc = np.array(system.pbc, copy=True)
        vacancy_system.positions = np.delete(
            np.array(system.positions, copy=True), 0, axis=0
        )
        vacancy_system.types = np.delete(np.array(system.types, copy=True), 0, axis=0)
        vacancy_system.index = np.arange(len(vacancy_system.positions))

        nl = NeighborsList(
            vacancy_system,
            config.atomicenvironment.rnei,
            config.atomicenvironment.rcut,
        )
        kwargs = {
            "style": "coordination/graph",
            "neighbors_list": nl.neighbors_list["rnei"],
            "environment_list": nl.neighbors_list["rcut"],
            "coordination_threshold": 12,
            "types": vacancy_system.types,
        }

        grey = AtomicEnvironment(coloring_mode="grey", **kwargs).atomic_environment_list
        full = AtomicEnvironment(coloring_mode="full", **kwargs).atomic_environment_list

        grey_non_crystal = [env for env in grey if env != "crystal"]
        full_non_crystal = [env for env in full if env != "crystal"]

        assert len(grey_non_crystal) == len(full_non_crystal) == 12
        assert grey_non_crystal != full_non_crystal
