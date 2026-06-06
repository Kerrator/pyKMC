"""Backend dispatch via create_rate_constant (constant / htst / rpa)."""

import math

import pytest

from pykmc.config import PhysicalConstants, RateConstantConfig
from pykmc.rate_constant import create_rate_constant

KB = PhysicalConstants().kb


def _rate(
    style: str, dE: float, nu0: float | None = None, k0: float = 10.0, T: float = 300.0
) -> float:
    rcfg = RateConstantConfig(style=style, k0=k0, T=T)
    rc = create_rate_constant(T=T, prefactor_backend_name=style, config=rcfg)
    return rc.compute_rate(dE, nu0=nu0).rate


def test_constant_style_uses_k0() -> None:
    """Constant style multiplies k0 by the Eyring exponential."""
    assert math.isclose(_rate("constant", 0.5), 10.0 * math.exp(-0.5 / (KB * 300.0)))


def test_htst_uses_nu0_when_present() -> None:
    """HTST style uses the supplied finite nu0 as the prefactor."""
    assert math.isclose(_rate("htst", 0.5, nu0=5e12), 5e12 * math.exp(-0.5 / (KB * 300.0)))


def test_htst_falls_back_to_k0_when_nu0_none() -> None:
    """HTST style falls back to k0 when nu0 is None."""
    assert math.isclose(_rate("htst", 0.5, nu0=None), 10.0 * math.exp(-0.5 / (KB * 300.0)))


def test_htst_falls_back_when_nu0_not_finite() -> None:
    """HTST style falls back to k0 when nu0 is non-finite (inf/nan)."""
    assert math.isclose(
        _rate("htst", 0.5, nu0=float("inf")), 10.0 * math.exp(-0.5 / (KB * 300.0))
    )


def test_rpa_uses_nu0_like_htst() -> None:
    """RPA style is bare-Vineyard for now: uses nu0 when present."""
    assert math.isclose(_rate("rpa", 0.5, nu0=7e12), 7e12 * math.exp(-0.5 / (KB * 300.0)))


def test_unknown_backend_raises() -> None:
    """An unregistered backend name raises ValueError (no silent fallback)."""
    rcfg = RateConstantConfig(style="constant", k0=1.0, T=300.0)
    with pytest.raises(ValueError):
        create_rate_constant(T=300.0, prefactor_backend_name="bogus", config=rcfg)
