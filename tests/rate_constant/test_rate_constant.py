"""RateConstant facade, unit oracle and constant-mode parity with the base."""

from __future__ import annotations

import dataclasses
import itertools
import math as m
from pathlib import Path

import pytest

from pykmc.config import Config, PhysicalConstants, RateConstantConfig
from pykmc.rate_constant import (
    RateComponents,
    RateConstant,
    compute_rate_Eyring,
    create_rate_constant,
    rate_from_prefactor,
)
from pykmc.rate_constant.backends.constant import ConstantBackend
from pykmc.rate_constant.backends.htst import HtstBackend

ROOT = Path(__file__).resolve().parents[2]
INPUT_IN = ROOT / "tests" / "data" / "input.in"

DE_GRID = (0.0, 0.05, 0.1, 0.5, 0.597, 1.2, 3.0, 10.0, 50.0)
T_GRID = (100.0, 300.0, 500.0, 1000.0, 1500.0)
K0_GRID = (1.0, 5.0, 10.0, 1e12)


def _base_expression(k0: float, dE: float, T: float) -> float:
    """Evaluate the literal expression of the flat pykmc.rate_constant module."""
    p = PhysicalConstants()
    return k0 * m.exp(-dE / (p.kb * T))


def _config_with(rate_cfg: RateConstantConfig) -> Config:
    """Return the committed test Config with its rateconstant section replaced."""
    config = Config.from_ini_file(str(INPUT_IN))
    return config.model_copy(update={"rateconstant": rate_cfg})


class TestUnitOracle:
    """constant k0=5.0 and htst nu0_hz=5e12 describe the same 5 THz prefactor."""

    @pytest.mark.parametrize("dE,T", list(itertools.product(DE_GRID, T_GRID)))
    def test_constant_equals_htst_at_equal_physical_prefactor(
        self, dE: float, T: float
    ) -> None:
        """Assert constant (k0=5.0) and htst (5e12 Hz) rates agree to 1e-12."""
        const = create_rate_constant(RateConstantConfig(style="constant", k0=5.0, T=T))
        htst = create_rate_constant(RateConstantConfig(style="htst", k0=1.0, T=T))
        k_const = const.compute_rate(dE)
        k_htst = htst.compute_rate(dE, nu0_hz=5e12)
        assert k_const.prefactor == 5.0
        assert k_htst.prefactor == 5.0
        assert math_isclose_or_both_zero(k_const.rate, k_htst.rate, 1e-12)

    def test_1e13_hz_resolves_to_10_per_ps(self) -> None:
        """Assert 1e13 Hz resolves to exactly 10.0 ps^-1."""
        rc = create_rate_constant(RateConstantConfig(style="htst", k0=1.0, T=300.0))
        out = rc.compute_rate(0.5, nu0_hz=1e13)
        assert out.prefactor == 10.0
        assert out.rate == rate_from_prefactor(10.0, 0.5, 300.0)

    def test_residence_time_in_seconds_is_physical(self) -> None:
        """Drive the KMC-clock arithmetic: 1/k (ps) * 1e-12 equals 1/(nu0 exp) in s."""
        T, dE, nu0_hz = 500.0, 0.597, 5.0e12
        rc = create_rate_constant(RateConstantConfig(style="htst", k0=1.0, T=T))
        k_per_ps = rc.compute_rate(dE, nu0_hz=nu0_hz).rate
        elapsed_s = (1.0 / k_per_ps) * 1e-12
        expected_s = 1.0 / (nu0_hz * m.exp(-dE / (PhysicalConstants().kb * T)))
        assert m.isclose(elapsed_s, expected_s, rel_tol=1e-9)


def math_isclose_or_both_zero(a: float, b: float, rel_tol: float) -> bool:
    """Return True when a and b agree to rel_tol or both underflowed to zero."""
    if a == 0.0 or b == 0.0:
        return a == b
    return m.isclose(a, b, rel_tol=rel_tol)


class TestRateFromPrefactor:
    """rate_from_prefactor is the Arrhenius arithmetic on a ps^-1 prefactor."""

    @pytest.mark.parametrize(
        "prefactor,dE,T", list(itertools.product((1.0, 5.0, 1e12), DE_GRID, T_GRID))
    )
    def test_matches_literal_expression(
        self, prefactor: float, dE: float, T: float
    ) -> None:
        """Assert the result equals prefactor * exp(-dE/(kb T)) bit for bit."""
        assert rate_from_prefactor(prefactor, dE, T) == _base_expression(
            prefactor, dE, T
        )

    def test_zero_barrier_returns_prefactor(self) -> None:
        """Assert a zero barrier gives the prefactor itself."""
        assert rate_from_prefactor(7.5, 0.0, 300.0) == 7.5


class TestRateConstant:
    """Facade behaviour: RateComponents and backend delegation."""

    def test_rate_components_is_frozen(self) -> None:
        """Assert RateComponents cannot be mutated after construction."""
        rc = RateComponents(prefactor=1.0, rate=0.5)
        assert (rc.prefactor, rc.rate) == (1.0, 0.5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            rc.prefactor = 2.0  # type: ignore[misc]

    def test_constant_prefactor_is_k0(self) -> None:
        """Assert the constant backend yields prefactor == k0 and ignores nu0_hz."""
        cfg = RateConstantConfig(style="constant", k0=10.0, T=100.0)
        rc = create_rate_constant(cfg)
        assert isinstance(rc.backend, ConstantBackend)
        assert rc.T == 100.0
        out = rc.compute_rate(0.5, nu0_hz=5e12)
        assert out == RateComponents(
            prefactor=10.0, rate=_base_expression(10.0, 0.5, 100.0)
        )

    def test_htst_uses_nu0_else_k0(self) -> None:
        """Assert htst uses the converted nu0 when present and k0 otherwise."""
        cfg = RateConstantConfig(style="htst", k0=10.0, T=100.0)
        rc = create_rate_constant(cfg)
        assert isinstance(rc.backend, HtstBackend)
        assert rc.compute_rate(0.5, nu0_hz=5e12).prefactor == 5.0
        assert rc.compute_rate(0.5).prefactor == 10.0
        assert rc.compute_rate(0.5, nu0_hz=None).prefactor == 10.0

    def test_direct_construction(self) -> None:
        """Assert RateConstant(T, backend) works without the factory."""
        backend = ConstantBackend(RateConstantConfig(style="constant", k0=2.0))
        rc = RateConstant(T=250.0, backend=backend)
        assert rc.compute_rate(0.3).rate == _base_expression(2.0, 0.3, 250.0)

    def test_unknown_style_cannot_reach_factory(self) -> None:
        """Assert the config Literal rejects an unregistered style before create."""
        with pytest.raises(ValueError):
            RateConstantConfig(style="nope")  # type: ignore[arg-type]


class TestComputeRateEyringParity:
    """compute_rate_Eyring reproduces the base bit for bit in constant mode."""

    @pytest.mark.parametrize(
        "k0,dE,T", list(itertools.product(K0_GRID, DE_GRID, T_GRID))
    )
    def test_constant_parity_exact(self, k0: float, dE: float, T: float) -> None:
        """Assert equality (==, not approx) with the base literal expression."""
        config = _config_with(RateConstantConfig(style="constant", k0=k0, T=T))
        result = compute_rate_Eyring(dE, config)
        assert isinstance(result, float)
        assert result == _base_expression(
            config.rateconstant.k0, dE, config.rateconstant.T
        )

    def test_parity_with_committed_input_file(self) -> None:
        """Assert the committed tests/data/input.in (k0=1e12) reproduces the base."""
        config = Config.from_ini_file(str(INPUT_IN))
        assert config.rateconstant.style == "constant"
        assert config.rateconstant.k0 == 1e12
        for dE in DE_GRID:
            assert compute_rate_Eyring(dE, config) == _base_expression(
                config.rateconstant.k0, dE, config.rateconstant.T
            )

    def test_large_barrier_underflows_to_zero_in_both(self) -> None:
        """Assert an Arrhenius underflow gives 0.0 on both sides, not an error."""
        config = _config_with(RateConstantConfig(style="constant", k0=1.0, T=100.0))
        assert (
            compute_rate_Eyring(100.0, config)
            == 0.0
            == _base_expression(1.0, 100.0, 100.0)
        )

    @pytest.mark.parametrize("style", ["htst", "rpa"])
    def test_htst_config_without_nu0_uses_k0_path(self, style: str) -> None:
        """Assert compute_rate_Eyring on an htst/rpa Config falls back to k0."""
        config = _config_with(RateConstantConfig(style=style, k0=3.0, T=400.0))
        for dE in (0.1, 0.5, 1.2):
            assert compute_rate_Eyring(dE, config) == _base_expression(3.0, dE, 400.0)

    def test_matches_facade_definition(self) -> None:
        """Assert the wrapper equals create_rate_constant(...).compute_rate(dE).rate."""
        config = _config_with(RateConstantConfig(style="constant", k0=4.0, T=350.0))
        facade = create_rate_constant(config.rateconstant).compute_rate(0.7).rate
        assert compute_rate_Eyring(0.7, config) == facade
