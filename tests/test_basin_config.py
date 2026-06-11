"""Validation tests for the ``[BASIN]`` configuration model."""

import pytest
from pydantic import ValidationError

from pykmc.config import BasinConfig


class TestBasinConfig:
    """Cover defaults, accepted values, and rejected values of ``BasinConfig``."""

    def test_defaults(self) -> None:
        """A bare ``BasinConfig`` keeps the basin serial and the solver automatic."""
        cfg = BasinConfig()
        assert cfg.energy_thr == 0.0
        assert cfg.strategy == "serial"
        assert cfg.n_workers == 4
        assert cfg.max_states is None
        assert cfg.max_total_states is None
        assert cfg.max_basin_walltime_s is None
        assert cfg.max_frontier_size is None
        assert cfg.max_failed_fraction == 0.2
        assert cfg.fingerprint_mode == "auto"
        assert cfg.fingerprint_coordination_thr is None
        assert cfg.fingerprint_tolerance is None
        assert cfg.solver == "auto"

    def test_fingerprint_mode_values(self) -> None:
        """All four fingerprint modes are accepted; anything else is rejected."""
        for mode in ("auto", "com", "atoms_of_interest", "off"):
            assert BasinConfig(fingerprint_mode=mode).fingerprint_mode == mode
        with pytest.raises(ValidationError):
            BasinConfig(fingerprint_mode="none")

    def test_accepts_full_parallel_config(self) -> None:
        """All parallel/fingerprint/solver fields accept their intended values."""
        cfg = BasinConfig(
            strategy="wavefront",
            n_workers=8,
            max_states=2000,
            fingerprint_coordination_thr=9,
            fingerprint_tolerance=1.0,
            solver="qsd",
        )
        assert cfg.strategy == "wavefront"
        assert cfg.n_workers == 8
        assert cfg.max_states == 2000
        assert cfg.fingerprint_coordination_thr == 9
        assert cfg.fingerprint_tolerance == 1.0
        assert cfg.solver == "qsd"

    def test_accepts_budget_config(self) -> None:
        """The exploration-budget knobs accept their intended values."""
        cfg = BasinConfig(
            max_total_states=10000,
            max_basin_walltime_s=7200.0,
            max_frontier_size=128,
            max_failed_fraction=0.5,
        )
        assert cfg.max_total_states == 10000
        assert cfg.max_basin_walltime_s == 7200.0
        assert cfg.max_frontier_size == 128
        assert cfg.max_failed_fraction == 0.5
        # boundary values of the failure budget
        assert BasinConfig(max_failed_fraction=0.0).max_failed_fraction == 0.0
        assert BasinConfig(max_failed_fraction=1.0).max_failed_fraction == 1.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"strategy": "parallel_reconstruct"},  # a removed dead alias
            {"strategy": "batch_dedup"},
            {"solver": "spectral"},
            {"n_workers": 0},
            {"max_states": 0},
            {"max_total_states": 0},
            {"max_basin_walltime_s": 0},
            {"max_frontier_size": 0},
            {"max_failed_fraction": -0.1},
            {"max_failed_fraction": 1.1},
        ],
    )
    def test_rejects_invalid_values(self, kwargs: dict) -> None:
        """Bad strategy/solver enums and non-positive counts raise ValidationError."""
        with pytest.raises(ValidationError):
            BasinConfig(**kwargs)
