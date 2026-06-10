"""Standalone tests for the batch prefactor API on the rate-constant backends.

The rate_constant module is standalone: backends optionally hold an injected
``manager`` (never the pydantic config) and expose
``compute_prefactors_batch(payloads, config) -> list[Future]``. htst/rpa fan
out one nu0 job per payload via ``manager.compute_event_prefactors``; the
constant backend (ABC default) short-circuits with immediately-resolved
futures so callers read a uniform ``EventPrefactors`` contract. No kmc import,
no MPI, no LAMMPS.
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import Any

import pytest

from pykmc.rate_constant import create_rate_constant
from pykmc.rate_constant.backends.constant import ConstantBackend
from pykmc.rate_constant.backends.htst import HtstBackend
from pykmc.rate_constant.backends.rpa import RpaBackend
from pykmc.rate_constant.prefactor import EventPrefactors


class _RC:
    """Minimal rateconstant sub-config (what backends store as self.config)."""

    k0 = 10.0


class _FullConfig:
    """Stand-in for the FULL pykmc Config forwarded to the manager fan-out."""


def _prefactors(nu0_f: float | None, nu0_b: float | None) -> EventPrefactors:
    return EventPrefactors(
        nu0_forward=nu0_f,
        nu0_backward=nu0_b,
        n_free=5,
        n_neg_saddle=1,
        ok_forward=nu0_f is not None,
        ok_backward=nu0_b is not None,
        reason="",
    )


class FakeManager:
    """Record fan-out calls; return pre-resolved futures (one per payload)."""

    def __init__(self, results: list[EventPrefactors]) -> None:
        self.calls: list[tuple[Any, list[dict[str, Any]]]] = []
        self._results = results

    def compute_event_prefactors(
        self, config: Any, events: list[dict[str, Any]]
    ) -> list[Future]:
        """Record the call and return one resolved future per event."""
        self.calls.append((config, list(events)))
        futures: list[Future] = []
        for pre in self._results[: len(events)]:
            f: Future = Future()
            f.set_result(pre)
            futures.append(f)
        return futures


_PAYLOADS = [
    {"central_atom_idx": 1, "min1_positions": "m1", "saddle_positions": "s",
     "min2_positions": "m2", "types": ["Ni"], "cell": "c"},
    {"central_atom_idx": 2, "min1_positions": "m1b", "saddle_positions": "sb",
     "min2_positions": "m2b", "types": ["Ni"], "cell": "c"},
]


def test_htst_batch_fans_out_via_manager() -> None:
    """Htst forwards the payloads + full config to the manager fan-out."""
    fake = FakeManager([_prefactors(5e12, 3e12), _prefactors(4e12, None)])
    backend = HtstBackend(_RC(), manager=fake)
    full = _FullConfig()

    futures = backend.compute_prefactors_batch(_PAYLOADS, full)

    assert fake.calls == [(full, _PAYLOADS)]
    assert len(futures) == 2
    assert futures[0].result().nu0_forward == 5e12
    assert futures[1].result().nu0_backward is None


def test_rpa_batch_fans_out_via_manager() -> None:
    """Rpa shares the fan-out behavior (direct PrefactorBackend subclass)."""
    fake = FakeManager([_prefactors(7e12, 6e12)])
    backend = RpaBackend(_RC(), manager=fake)

    futures = backend.compute_prefactors_batch(_PAYLOADS[:1], _FullConfig())

    assert len(fake.calls) == 1
    assert futures[0].result().nu0_forward == 7e12


def test_constant_batch_short_circuits_without_manager() -> None:
    """Constant style resolves immediately with nu0=None (keep-k0 contract)."""
    backend = ConstantBackend(_RC())

    futures = backend.compute_prefactors_batch(_PAYLOADS, _FullConfig())

    assert len(futures) == 2
    for f in futures:
        pre = f.result(timeout=0)
        assert pre.nu0_forward is None
        assert pre.nu0_backward is None


def test_htst_batch_raises_without_manager() -> None:
    """Htst without an injected manager fails fast with a clear message."""
    backend = HtstBackend(_RC(), manager=None)
    with pytest.raises(RuntimeError, match="manager"):
        backend.compute_prefactors_batch(_PAYLOADS, _FullConfig())


def test_facade_passthrough() -> None:
    """create_rate_constant(..., manager=) reaches the backend; facade delegates."""
    fake = FakeManager([_prefactors(5e12, None)])
    rc = create_rate_constant(
        T=300.0, prefactor_backend_name="htst", config=_RC(), manager=fake
    )

    futures = rc.compute_prefactors_batch(_PAYLOADS[:1], _FullConfig())

    assert len(fake.calls) == 1
    assert futures[0].result().nu0_forward == 5e12


def test_compute_unchanged_with_manager_param() -> None:
    """The sync compute() contract is untouched by the new ctor param."""
    backend = HtstBackend(_RC(), manager=None)
    assert backend.compute(nu0=2.0e12) == 2.0e12
    assert backend.compute(nu0=None) == 10.0
    assert ConstantBackend(_RC()).compute() == 10.0
