"""Tests for the htst rate constant strategy."""

import math

from pykmc.config import PhysicalConstants, RateConstantConfig
from pykmc.rate_constant import RateConstant, RateConstantStrategy


def test_htst_strategy_uses_nu0_in_thz() -> None:
    config = RateConstantConfig(style="constant", k0=10.0, T=500.0)
    rate_constant = RateConstant.create("htst", config=config)

    for dE, nu0 in ((0.0, 10.0), (0.5, 22.0), (1.2, 7.0)):
        expected = nu0 * math.exp(-dE / (PhysicalConstants.kb * config.T))
        assert rate_constant.compute_rate(dE, nu0=nu0) == expected


def test_htst_prefactor_is_nu0_or_k0() -> None:
    config = RateConstantConfig(style="constant", k0=10.0, T=500.0)
    strategy = RateConstantStrategy.create("htst", config=config)

    assert strategy.prefactor(22.0) == 22.0
    for nu0 in (None, math.nan, math.inf, -math.inf):
        assert strategy.prefactor(nu0) == config.k0


def test_htst_strategy_without_nu0_equals_constant() -> None:
    config = RateConstantConfig(style="constant", k0=10.0, T=500.0)
    htst = RateConstant.create("htst", config=config)
    constant = RateConstant.create("constant", config=config)

    for dE in (0.0, 0.5, 1.2):
        for nu0 in (None, math.nan, math.inf):
            assert htst.compute_rate(dE, nu0=nu0) == constant.compute_rate(dE)
