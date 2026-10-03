"""Tests for the htst rate constant strategy."""

import math

from pykmc.config import PhysicalConstants, RateConstantConfig
from pykmc.rate_constant import RateConstant


def test_htst_strategy_uses_nu0() -> None:
    config = RateConstantConfig(style="constant", k0=10.0, T=500.0)
    rate_constant = RateConstant.create("htst", config=config)

    for dE, nu0 in ((0.0, 1.0e13), (0.5, 2.2e13), (1.2, 7.0e12)):
        expected = nu0 * 1e-12 * math.exp(-dE / (PhysicalConstants.kb * config.T))
        assert rate_constant.compute_rate(dE, nu0=nu0) == expected


def test_htst_strategy_without_nu0_equals_constant() -> None:
    config = RateConstantConfig(style="constant", k0=10.0, T=500.0)
    htst = RateConstant.create("htst", config=config)
    constant = RateConstant.create("constant", config=config)

    for dE in (0.0, 0.5, 1.2):
        for nu0 in (None, math.nan, math.inf):
            assert htst.compute_rate(dE, nu0=nu0) == constant.compute_rate(dE)
