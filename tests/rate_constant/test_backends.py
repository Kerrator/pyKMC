"""Registry and prefactor resolution of the PrefactorBackend hierarchy."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pykmc.config import RateConstantConfig
from pykmc.rate_constant import PrefactorBackend, create_rate_constant
from pykmc.rate_constant.backends.constant import ConstantBackend
from pykmc.rate_constant.backends.htst import HtstBackend
from pykmc.rate_constant.backends.rpa import RpaBackend

K0 = 10.0


def _cfg(style: str = "constant", k0: float = K0) -> RateConstantConfig:
    return RateConstantConfig(style=style, k0=k0, T=300.0)


class TestRegistry:
    """Backends are registered through the shared Registrable/autodiscover."""

    def test_registered_names(self) -> None:
        """Assert the registry holds exactly constant, htst and rpa."""
        assert set(PrefactorBackend._registry) == {"constant", "htst", "rpa"}
        assert PrefactorBackend._registry["constant"] is ConstantBackend
        assert PrefactorBackend._registry["htst"] is HtstBackend
        assert PrefactorBackend._registry["rpa"] is RpaBackend

    def test_no_import_errors_recorded(self) -> None:
        """Assert every backend module imported cleanly in this environment."""
        assert PrefactorBackend._import_errors == {}

    def test_unknown_name_lists_available(self) -> None:
        """Assert create('nope') raises ValueError naming the registered backends."""
        with pytest.raises(ValueError, match="nope") as excinfo:
            PrefactorBackend.create("nope", config=_cfg())
        message = str(excinfo.value)
        for name in ("constant", "htst", "rpa"):
            assert name in message

    def test_every_backend_declares_contract(self) -> None:
        """Assert each registered backend has a bool flag and a module-basename name."""
        for name, cls in PrefactorBackend._registry.items():
            assert issubclass(cls, PrefactorBackend)
            assert cls.name == name
            assert cls.__module__.rsplit(".", 1)[-1] == name
            flag = cls.__dict__.get("requires_event_prefactors", None)
            if flag is None:
                flag = cls.requires_event_prefactors
            assert isinstance(flag, bool)
            backend = cls(_cfg(style=name, k0=K0))
            assert backend.resolve_prefactor(None) == K0

    def test_requires_event_prefactors_flags(self) -> None:
        """Assert constant does not need per-event prefactors while htst and rpa do."""
        assert ConstantBackend.requires_event_prefactors is False
        assert HtstBackend.requires_event_prefactors is True
        assert RpaBackend.requires_event_prefactors is True

    def test_rpa_is_an_htst_subclass(self) -> None:
        """Assert rpa subclasses the htst backend and registers under its own name."""
        assert issubclass(RpaBackend, HtstBackend)
        assert RpaBackend.name == "rpa"
        assert "no recrossing correction" in RpaBackend.__doc__.lower()

    def test_root_is_abstract(self) -> None:
        """Assert the registry root cannot be instantiated."""
        with pytest.raises(TypeError):
            PrefactorBackend(_cfg())

    def test_create_binds_config(self) -> None:
        """Assert create() forwards the config to the backend instance."""
        cfg = _cfg(style="htst", k0=3.0)
        backend = PrefactorBackend.create("htst", config=cfg)
        assert isinstance(backend, HtstBackend)
        assert backend.config is cfg

    def test_unavailable_htst_keeps_constant_working(self) -> None:
        """Simulate an htst import failure: constant works, htst raises ImportError."""
        registry = PrefactorBackend._registry
        errors = PrefactorBackend._import_errors
        saved_registry = dict(registry)
        saved_errors = dict(errors)
        cause = OSError("dlopen failed: simulated native failure")
        recorded = ImportError("pykmc.rate_constant.backends.htst failed to import")
        recorded.__cause__ = cause
        try:
            del registry["htst"]
            errors["htst"] = recorded
            rc = create_rate_constant(_cfg(style="constant", k0=K0))
            assert rc.compute_rate(0.5).prefactor == K0
            with pytest.raises(ImportError, match="htst") as excinfo:
                create_rate_constant(_cfg(style="htst", k0=1.0))
            assert excinfo.value.__cause__ is recorded
            assert recorded.__cause__ is cause
        finally:
            registry.clear()
            registry.update(saved_registry)
            errors.clear()
            errors.update(saved_errors)
        assert "htst" in PrefactorBackend._registry


class TestConstantBackend:
    """The constant backend always returns k0."""

    @pytest.mark.parametrize("nu0_hz", [None, 5e12, float("nan"), 0.0])
    def test_ignores_nu0(self, nu0_hz: float | None) -> None:
        """Assert the resolved prefactor is k0 whatever nu0_hz is."""
        assert ConstantBackend(_cfg(k0=K0)).resolve_prefactor(nu0_hz) == K0


UNAVAILABLE = [None, float("nan"), float("inf"), float("-inf"), np.float64("nan")]
BAD_TYPES = [True, False, "5e12", b"5e12", [5e12], (5e12,), {"nu0": 5e12}, 5e12 + 0j]
NON_POSITIVE = [0.0, 0, -1.0, -5e12, np.float64(0.0), np.float64(-3e12)]


@pytest.mark.parametrize("backend_cls", [HtstBackend, RpaBackend])
class TestVineyardBackends:
    """htst and rpa share the resolve_prefactor encodings."""

    @pytest.mark.parametrize("nu0_hz", UNAVAILABLE)
    def test_unavailable_encodings_fall_back_to_k0(
        self, backend_cls: type[HtstBackend], nu0_hz: float | None
    ) -> None:
        """Assert None, NaN and +-inf resolve to k0 (unconverted, ps^-1)."""
        backend = backend_cls(_cfg(style=backend_cls.name, k0=K0))
        assert backend.resolve_prefactor(nu0_hz) == K0

    @pytest.mark.parametrize("nu0_hz", BAD_TYPES)
    def test_non_real_values_raise_type_error(
        self, backend_cls: type[HtstBackend], nu0_hz: object
    ) -> None:
        """Assert bool, str and other non-real values are programming errors."""
        backend = backend_cls(_cfg(style=backend_cls.name, k0=K0))
        with pytest.raises(TypeError):
            backend.resolve_prefactor(nu0_hz)

    @pytest.mark.parametrize("nu0_hz", NON_POSITIVE)
    def test_non_positive_values_raise_value_error(
        self, backend_cls: type[HtstBackend], nu0_hz: float
    ) -> None:
        """Assert zero and negative finite frequencies are rejected, not k0."""
        backend = backend_cls(_cfg(style=backend_cls.name, k0=K0))
        with pytest.raises(ValueError, match="positive"):
            backend.resolve_prefactor(nu0_hz)

    def test_finite_positive_is_converted_once(
        self, backend_cls: type[HtstBackend]
    ) -> None:
        """Assert a Hz value is converted to ps^-1 exactly once."""
        backend = backend_cls(_cfg(style=backend_cls.name, k0=K0))
        assert backend.resolve_prefactor(1e13) == 10.0
        assert backend.resolve_prefactor(5e12) == 5.0
        assert math.isclose(backend.resolve_prefactor(3.7e12), 3.7, rel_tol=1e-12)

    def test_numpy_and_int_scalars_accepted(
        self, backend_cls: type[HtstBackend]
    ) -> None:
        """Assert numpy float64 and Python int are accepted as real numbers."""
        backend = backend_cls(_cfg(style=backend_cls.name, k0=K0))
        out = backend.resolve_prefactor(np.float64(1e13))
        assert out == 10.0
        assert isinstance(out, float)
        assert backend.resolve_prefactor(10_000_000_000_000) == 10.0
        assert backend.resolve_prefactor(np.float32(1e12)) == pytest.approx(1.0)


def test_rpa_matches_htst_on_a_grid() -> None:
    """Assert rpa resolves the same prefactor as htst for every input."""
    htst = HtstBackend(_cfg(style="htst", k0=K0))
    rpa = RpaBackend(_cfg(style="rpa", k0=K0))
    for nu0_hz in (None, float("nan"), 1e12, 5e12, 1e13, 9.87e13):
        assert rpa.resolve_prefactor(nu0_hz) == htst.resolve_prefactor(nu0_hz)
