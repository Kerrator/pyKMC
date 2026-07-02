"""Coordination-environment classification and multi-element (full colour) support.

Ported from the multi-element colour feature (commit 76ba649) and extended with
focused unit coverage for the colour graph hashing and the config surface.
"""

from collections import Counter

import pytest
from pytest_lazy_fixtures import lf

from pykmc import AtomicEnvironment, NeighborsList
from pykmc.config import AtomicEnvironmentConfig
from pykmc.environments import graph


class TestCoordination:
    @pytest.mark.parametrize(
        "system, config",
        [(lf("system_single_type_fcc"), lf("config_system_single_type"))],
    )
    def test_coordination_bulk_fcc(self, system, config):
        """All atoms in a perfect FCC bulk are 'crystal' (12 neighbors)."""
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
        """A monovacancy leaves exactly its 12 first-neighbors under-coordinated."""
        nl = NeighborsList(
            system, config.atomicenvironment.rnei, config.atomicenvironment.rcut
        )
        ae = AtomicEnvironment(
            "coordination", nl.neighbors_list["rnei"], coordination_threshold=12
        )
        hash_count = Counter(ae.atomic_environment_list)
        assert "noncrystal" in hash_count
        assert hash_count["noncrystal"] == 12

    def test_coordination_requires_threshold(self):
        """coordination() without a threshold fails loudly rather than silently."""
        with pytest.raises(AssertionError):
            AtomicEnvironment("coordination", [[1], [0]], coordination_threshold=None)


class TestColourGraph:
    """`atom_coloring_mode='full'` must make the graph hash element-sensitive."""

    # An asymmetric environment (a 3-path 0-1-2) so colour can break symmetry.
    NL = [[1], [0, 2], [1]]
    ENV = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]

    def test_grey_hash_ignores_elements(self):
        """Without types, the hash is colour-blind (same for any labeling)."""
        h_a = graph(self.NL, self.ENV, [0], types=None)
        h_b = graph(self.NL, self.ENV, [0])  # default types=None
        assert h_a == h_b

    def test_full_colour_distinguishes_elements(self):
        """A Cr at the path centre vs. at an end gives distinct coloured hashes."""
        h_grey = graph(self.NL, self.ENV, [0])
        h_centre = graph(self.NL, self.ENV, [0], types=["Ni", "Cr", "Ni"])
        h_end = graph(self.NL, self.ENV, [0], types=["Cr", "Ni", "Ni"])
        assert h_grey != h_centre
        assert h_centre != h_end

    def test_same_colouring_is_stable(self):
        """Identical geometry + identical labeling -> identical hash."""
        types = ["Ni", "Cr", "Ni"]
        assert graph(self.NL, self.ENV, [0], types=types) == graph(
            self.NL, self.ENV, [0], types=types
        )


class TestAtomColoringConfig:
    def test_default_is_full(self):
        cfg = AtomicEnvironmentConfig(style="cna/graph", rnei=3.0, rcut=6.5)
        assert cfg.atom_coloring_mode == "full"
        assert cfg.coordination_threshold is None

    def test_accepts_full_colour(self):
        cfg = AtomicEnvironmentConfig(
            style="coordination/graph",
            rnei=3.0,
            rcut=6.5,
            coordination_threshold=12,
            atom_coloring_mode="full",
        )
        assert cfg.atom_coloring_mode == "full"
        assert cfg.coordination_threshold == 12

    def test_coordination_threshold_required_for_coordination_styles(self):
        for style in ("coordination", "coordination/graph"):
            with pytest.raises(ValueError):
                AtomicEnvironmentConfig(style=style, rnei=3.0, rcut=6.5)

    def test_rejects_unknown_colour_mode(self):
        with pytest.raises(ValueError):
            AtomicEnvironmentConfig(
                style="cna/graph", rnei=3.0, rcut=6.5, atom_coloring_mode="rainbow"
            )
