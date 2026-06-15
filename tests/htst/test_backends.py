"""Prefactor backends + factory registry + RateConstant (pykmc.rate_constant)."""

import math

import pytest

from pykmc.config import PhysicalConstants, RateConstantConfig
from pykmc.rate_constant import RateConstant, create_rate_constant, rate_from_prefactor
from pykmc.rate_constant.backends.base import PrefactorBackend
from pykmc.rate_constant.backends.constant import ConstantBackend
from pykmc.rate_constant.backends.htst import HtstBackend
from pykmc.rate_constant.backends.rpa import RpaBackend
from pykmc.rate_constant.factory import _get_registry

KB = PhysicalConstants().kb


def _cfg(k0: float = 10.0, T: float = 100.0) -> RateConstantConfig:
    return RateConstantConfig(style="constant", k0=k0, T=T)


def test_registry_contains_all_backends() -> None:
    """The auto-import registry discovers constant, htst and rpa."""
    assert {"constant", "htst", "rpa"} <= set(_get_registry())


def test_constant_backend_ignores_nu0() -> None:
    """ConstantBackend always returns k0, even when handed an nu0."""
    backend = ConstantBackend(_cfg(k0=10.0))
    assert backend.compute() == 10.0
    assert backend.compute(nu0=5e12) == 10.0


def test_htst_backend_uses_nu0_else_k0() -> None:
    """HtstBackend returns nu0 when finite, else the k0 fallback."""
    backend = HtstBackend(_cfg(k0=10.0))
    assert backend.compute(nu0=5e12) == 5e12 * 1e-12  # nu0 Hz -> ps^-1
    assert backend.compute() == 10.0
    assert backend.compute(nu0=None) == 10.0
    assert backend.compute(nu0=float("nan")) == 10.0


def test_rpa_backend_is_direct_subclass_and_bare_vineyard() -> None:
    """RpaBackend must subclass PrefactorBackend DIRECTLY (non-transitive registry)."""
    assert RpaBackend in PrefactorBackend.__subclasses__()
    assert RpaBackend(_cfg(k0=10.0)).compute(nu0=7e12) == 7e12 * 1e-12  # nu0 Hz -> ps^-1


def test_factory_dispatch_and_unknown_raises() -> None:
    """create_rate_constant builds a RateConstant; an unknown name raises ValueError."""
    rc = create_rate_constant(T=100.0, prefactor_backend_name="htst", config=_cfg())
    assert isinstance(rc, RateConstant)
    with pytest.raises(ValueError):
        create_rate_constant(T=100.0, prefactor_backend_name="nope", config=_cfg())


def test_rate_constant_compute_rate() -> None:
    """compute_rate threads nu0 to the backend and applies the Eyring exponential."""
    rc = create_rate_constant(T=100.0, prefactor_backend_name="htst", config=_cfg(k0=10.0))
    out = rc.compute_rate(0.5, nu0=5e12)
    assert out.prefactor == 5e12 * 1e-12  # nu0 Hz -> ps^-1
    assert math.isclose(out.rate, 5e12 * 1e-12 * math.exp(-0.5 / (KB * 100.0)))
    assert rc.compute_rate(0.5).prefactor == 10.0  # fallback to k0 (already ps^-1)


def test_rate_from_prefactor_helper() -> None:
    """rate_from_prefactor is the shared Eyring combine reused by active events."""
    assert math.isclose(
        rate_from_prefactor(5e12, 0.5, 100.0), 5e12 * math.exp(-0.5 / (KB * 100.0))
    )
