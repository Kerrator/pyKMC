"""PrefactorService: request building, batching and strict result mapping.

No LAMMPS, no MPI: a fake manager resolves ``compute_event_prefactors``
Futures, including out of submission order and by raising.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from ase.data import atomic_masses, atomic_numbers

from pykmc.config import Config, RateConstantConfig
from pykmc.htst.request import HTSTEventRequest, HTSTRequestError
from pykmc.htst.settings import HTSTSettings
from pykmc.rate_constant import create_rate_constant, thz_to_hz
from pykmc.rate_constant.prefactors import (
    PREFACTOR_OPERATION,
    PrefactorService,
    settings_from_config,
)
from tests.lifecycle.conftest import (
    DATA_INPUT,
    FakeManager,
    accepted,
    event_prefactors,
    rejected,
)


def _config(style: str, **rate_kwargs: Any) -> Config:
    """Return the committed test input with its rateconstant section replaced."""
    config = Config.from_ini_file(DATA_INPUT)
    rate = RateConstantConfig(style=style, **rate_kwargs)
    return config.model_copy(update={"rateconstant": rate})


def _service(config: Config, manager: FakeManager) -> PrefactorService:
    return PrefactorService(config, manager, create_rate_constant(config.rateconstant))


def _geometry(n: int = 8) -> dict[str, Any]:
    """Small full-system geometry with two species and a cubic cell."""
    rng = np.random.default_rng(3)
    pos = rng.uniform(0.0, 8.0, size=(n, 3))
    return {
        "min1_positions": pos,
        "saddle_positions": pos + 0.1,
        "min2_positions": pos + 0.2,
        "types": ["Ni", "Fe", "Ni", "Ni", "Fe", "Ni", "Ni", "Ni"][:n],
        "cell": np.eye(3) * 8.0,
        "pbc": np.array([True, True, False]),
        "center_index": 2,
    }


class TestSettings:
    """The THz window is converted to Hz exactly once, at the service boundary."""

    def test_settings_from_config_converts_window_once(self) -> None:
        """nu0_min/max_THz become nu0_min/max_hz through thz_to_hz."""
        config = _config(
            "htst",
            k0=1.0,
            T=300.0,
            nu0_min_THz=2.5,
            nu0_max_THz=80.0,
            free_radius=4.5,
            fd_step=0.02,
            zone_radius=9.0,
            premin=True,
        )
        settings = settings_from_config(config.rateconstant)
        assert isinstance(settings, HTSTSettings)
        assert settings.nu0_min_hz == thz_to_hz(2.5) == 2.5e12
        assert settings.nu0_max_hz == thz_to_hz(80.0) == 8.0e13
        assert settings.free_radius == 4.5
        assert settings.fd_step == 0.02
        assert settings.zone_radius == 9.0
        assert settings.premin is True

    def test_service_holds_one_settings_object(self) -> None:
        """Every request shares the service's settings instance."""
        config = _config("htst", k0=1.0, T=300.0)
        service = _service(config, FakeManager())
        req = service.build_request(event_key=("a",), **_geometry())
        assert req.settings is service.settings
        assert req.settings.nu0_min_hz == 1.0e12
        assert req.settings.nu0_max_hz == 1.0e14


class TestConstruction:
    """The service exists only for backends that need per-event prefactors."""

    def test_constant_style_cannot_build_a_service(self) -> None:
        """The constant backend never gets a service (nothing to batch)."""
        config = _config("constant", k0=1.0, T=300.0)
        with pytest.raises(ValueError, match="requires per-event prefactors"):
            _service(config, FakeManager())

    def test_manager_required(self) -> None:
        """A None manager is an ordering error, not a silent no-op."""
        config = _config("htst", k0=1.0, T=300.0)
        with pytest.raises(RuntimeError, match="manager"):
            PrefactorService(config, None, create_rate_constant(config.rateconstant))

    def test_rpa_style_builds_a_service(self) -> None:
        """The rpa style shares the htst backend behaviour."""
        config = _config("rpa", k0=1.0, T=300.0)
        assert _service(config, FakeManager()).settings.nu0_min_hz == 1.0e12


class TestBuildRequest:
    """Requests carry the one species rule, the real pbc and copied geometry."""

    def test_species_and_masses_follow_the_one_species_rule(self) -> None:
        """Species are sorted(set(types)); masses are ASE masses in that order."""
        config = _config("htst", k0=1.0, T=300.0)
        service = _service(config, FakeManager())
        geometry = _geometry()
        req = service.build_request(event_key=("evt", 1), **geometry)

        assert isinstance(req, HTSTEventRequest)
        assert req.species == ("Fe", "Ni")
        expected = tuple(float(atomic_masses[atomic_numbers[s]]) for s in ("Fe", "Ni"))
        assert req.masses == expected
        assert req.types == tuple(geometry["types"])
        assert req.pbc == (True, True, False)
        assert all(isinstance(p, bool) for p in req.pbc)
        assert req.center_index == 2
        assert req.event_key == ("evt", 1)

    def test_geometry_is_copied(self) -> None:
        """Mutating the caller's arrays after building does not alter the request."""
        config = _config("htst", k0=1.0, T=300.0)
        service = _service(config, FakeManager())
        geometry = _geometry()
        req = service.build_request(event_key=("evt",), **geometry)
        before = req.min1_positions.copy()
        geometry["min1_positions"][0] += 100.0
        assert np.array_equal(req.min1_positions, before)

    def test_invalid_request_is_rejected_at_build(self) -> None:
        """A non-orthorhombic cell fails validation before any submission."""
        config = _config("htst", k0=1.0, T=300.0)
        service = _service(config, FakeManager())
        geometry = _geometry()
        geometry["cell"] = np.array([[8.0, 1.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]])
        with pytest.raises(HTSTRequestError):
            service.build_request(event_key=("evt",), **geometry)


class TestCompute:
    """Batch submission with results mapped by key, whatever the completion order."""

    @staticmethod
    def _responder(req: HTSTEventRequest) -> Any:
        scale = float(req.event_key[1])
        return event_prefactors(
            req.event_key, accepted(1.0e12 * scale), accepted(2.0e12 * scale)
        )

    def test_out_of_order_completion_maps_back_by_key(self) -> None:
        """Futures resolved last-to-first still land on their own event."""
        config = _config("htst", k0=1.0, T=300.0)
        fake = FakeManager(self._responder, completion="reverse", expected=3)
        service = _service(config, fake)
        keys = [("evt", 1), ("evt", 2), ("evt", 3)]
        requests = [service.build_request(event_key=k, **_geometry()) for k in keys]

        results = service.compute(requests)

        assert fake.completion_order == list(reversed(keys))
        assert set(results) == set(keys)
        for key in keys:
            assert results[key].event_key == key
            assert results[key].forward.nu0_hz == 1.0e12 * key[1]
            assert results[key].backward.nu0_hz == 2.0e12 * key[1]
        assert [op for op, _ in fake.submitted] == [PREFACTOR_OPERATION] * 3
        assert service.n_submitted == 3

    def test_duplicate_keys_are_rejected_before_submission(self) -> None:
        """Two requests with one key would be unmappable: nothing is submitted."""
        config = _config("htst", k0=1.0, T=300.0)
        fake = FakeManager(self._responder)
        service = _service(config, fake)
        requests = [
            service.build_request(event_key=("evt", 1), **_geometry()),
            service.build_request(event_key=("evt", 1), **_geometry()),
        ]
        with pytest.raises(ValueError, match="distinct"):
            service.compute(requests)
        assert fake.submitted == []

    def test_wrong_echoed_key_is_a_contract_failure(self) -> None:
        """A worker echoing another event's key is reported, never mapped."""
        config = _config("htst", k0=1.0, T=300.0)
        fake = FakeManager(
            lambda req: event_prefactors(("other",), accepted(1e12), accepted(1e12))
        )
        service = _service(config, fake)
        with pytest.raises(RuntimeError, match="echoed event_key"):
            service.compute(
                [service.build_request(event_key=("evt", 1), **_geometry())]
            )

    def test_non_result_value_is_a_contract_failure(self) -> None:
        """A None result (e.g. a non-root reply leaking through) is an error."""
        config = _config("htst", k0=1.0, T=300.0)
        service = _service(config, FakeManager(lambda req: None))
        with pytest.raises(RuntimeError, match="expected EventPrefactors"):
            service.compute(
                [service.build_request(event_key=("evt", 1), **_geometry())]
            )

    def test_worker_exception_propagates(self) -> None:
        """A raising worker operation surfaces through the Future, no fallback."""

        def boom(req: HTSTEventRequest) -> Any:
            raise OSError("scratch LAMMPS instance failed")

        config = _config("htst", k0=1.0, T=300.0)
        service = _service(config, FakeManager(boom))
        with pytest.raises(OSError, match="scratch LAMMPS"):
            service.compute(
                [service.build_request(event_key=("evt", 1), **_geometry())]
            )

    def test_rejections_are_data_not_errors(self) -> None:
        """Scientific rejections come back as data for the tables to apply."""
        config = _config("htst", k0=1.0, T=300.0)
        fake = FakeManager(
            lambda req: event_prefactors(
                req.event_key, accepted(3e12), rejected("nope")
            )
        )
        service = _service(config, fake)
        res = service.compute(
            [service.build_request(event_key=("evt", 1), **_geometry())]
        )
        pre = res[("evt", 1)]
        assert pre.forward.ok and pre.forward.nu0_hz == 3e12
        assert not pre.backward.ok and pre.backward.reason == "nope"

    def test_empty_batch_submits_nothing(self) -> None:
        """No requests, no jobs."""
        config = _config("htst", k0=1.0, T=300.0)
        fake = FakeManager(self._responder)
        assert _service(config, fake).compute([]) == {}
        assert fake.submitted == []
