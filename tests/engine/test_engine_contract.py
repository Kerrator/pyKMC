from pykmc.engine import Engine, EngineExtension
import numpy as np
import pytest

class EngineContractTests:
    """
    Suite de tests définissant le contrat que toute implémentation
    d'Engine doit respecter. Ne pas instancier directement.

    Les sous-classes doivent implémenter `make_engine()`, qui retourne
    une instance configurée mais non démarrée de l'engine.
    """

    def make_engine(self) -> Engine:
        raise NotImplementedError

    @property
    def is_rank0(self) -> bool:
        """True si le rank courant doit valider les assertions."""
        return True

    # ── start / close ─────────────────────────────────────────────────────────

    def test_start_does_not_raise(self):
        """Test open and close engine."""
        engine = self.make_engine()
        engine.start()
        engine.close()

    def initialize(self, engine):
        """Initialization parameters and system convenience method."""
        engine.initialize_parameters()
        engine.initialize_system(
            types=self.system.types,
            positions=self.system.positions,
            cell=self.system.cell,
            pbc=self.system.pbc,
        )
        engine.initialize_potential()

    def test_initialize(self):
        """Test initialization parameters and system."""
        engine = self.make_engine()
        engine.start()
        self.initialize(engine)
        engine.close()

    # ── positions ─────────────────────────────────────────────────────────────

    def test_set_get_positions(self):
        """set_positions() puis get_positions() retourne les mêmes positions."""
        engine = self.make_engine()
        engine.start()
        self.initialize(engine)
        positions = self.system.positions.copy()
        positions[0, 0] += 0.2
        engine.set_positions(positions)
        result = engine.get_positions()
        if self.is_rank0:
            np.testing.assert_allclose(result, positions, atol=1e-10)
        engine.close()

    # ── énergie ───────────────────────────────────────────────────────────────

    def test_get_potential_energy(self):
        """get_potential_energy() retourne un float."""
        engine = self.make_engine()
        engine.start()
        self.initialize(engine)
        pe = engine.get_potential_energy()
        if self.is_rank0:
            assert isinstance(pe, float)
        engine.close()

    def test_get_total_energy(self):
        """get_total_energy() retourne un float."""
        engine = self.make_engine()
        engine.start()
        self.initialize(engine)
        tot_e = engine.get_total_energy()
        if self.is_rank0:
            assert isinstance(tot_e, float)
        engine.close()

    # ── minimisation ──────────────────────────────────────────────────────────

    def test_minimize(self):
        """minimize() réduit l'énergie d'une configuration perturbée."""
        engine = self.make_engine()
        engine.start()
        self.initialize(engine)
        rng = np.random.default_rng(seed=0)
        perturbed = self.system.positions.copy() + rng.uniform(-0.05, 0.05, size=self.system.positions.shape)
        e_before = engine.get_potential_energy(positions=perturbed)
        engine.minimize()
        e_after = engine.get_potential_energy()
        if self.is_rank0:
            assert e_after < e_before
        engine.close()

    def test_minimize_with_results(self):
        """minimize_with_results() réduit l'énergie et retourne positions + énergie."""
        engine = self.make_engine()
        engine.start()
        self.initialize(engine)
        rng = np.random.default_rng(seed=42)
        perturbed = self.system.positions.copy() + rng.uniform(-0.1, 0.1, size=self.system.positions.shape)
        e_perturbed = engine.get_potential_energy(positions=perturbed)
        min_positions, e_min = engine.minimize_with_results(positions=perturbed)
        if self.is_rank0:
            assert min_positions.shape == self.system.positions.shape
            assert e_min < e_perturbed
        engine.close()

    # ── extensions ────────────────────────────────────────────────────────────

    def make_test_extension(self, engine) -> EngineExtension:
        """Retourne une extension concrète compatible avec cet engine."""
        raise NotImplementedError

    def make_conflicting_extension(self, engine) -> EngineExtension:
        """Retourne une extension dont les méthodes entrent en conflit avec make_test_extension()."""
        raise NotImplementedError

    def test_extension_registers_and_delegates(self):
        """Une extension enregistrée expose ses méthodes publiques via l'engine."""
        engine = self.make_engine()
        engine.start()
        self.initialize(engine)
        ext = self.make_test_extension(engine)
        public = [m for m in dir(ext) if not m.startswith("_")]
        assert all(hasattr(engine, m) for m in public)
        engine.close()

    def test_extension_conflict_raises(self):
        """Enregistrer deux extensions avec un nom de méthode identique lève ValueError."""
        engine = self.make_engine()
        engine.start()
        self.initialize(engine)
        self.make_test_extension(engine)
        with pytest.raises(ValueError):
            self.make_conflicting_extension(engine)
        engine.close()
