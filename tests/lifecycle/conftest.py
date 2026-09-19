"""Shared fixtures for the catalogue-lifecycle tests (no LAMMPS, no MPI).

The doubles here stand in for the frozen cross-slice contracts:

* :func:`accepted` / :func:`rejected` build ``DirectionalPrefactor`` values
  exactly as ``pykmc.htst`` produces them;
* :func:`event_prefactors` wraps a forward/backward pair into the
  ``EventPrefactors`` a ``compute_event_prefactors`` worker operation returns;
* :class:`FakeManager` mimics the slice of ``pykmc.manager.Manager`` the
  lifecycle code uses (``submit`` returning a ``Future`` and ``broadcast``),
  resolving each request through a caller-supplied responder.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future
from typing import Any, Callable

import pytest

from pykmc.config import Config, RateConstantConfig
from pykmc.htst.result import (
    DirectionalPrefactor,
    EventPrefactors,
    PrefactorRejection,
)
from pykmc.htst.settings import HTSTSettings

DATA_INPUT = "./tests/data/input.in"


def accepted(nu0_hz: float) -> DirectionalPrefactor:
    """Return an accepted directional estimate carrying ``nu0_hz``."""
    return DirectionalPrefactor.accepted(
        nu0_hz, n_free=5, n_positive_min=15, n_negative_saddle=1
    )


def rejected(reason: str = "test rejection") -> DirectionalPrefactor:
    """Return a rejected directional estimate with an ``OUT_OF_WINDOW`` code."""
    return DirectionalPrefactor.rejected(
        PrefactorRejection.OUT_OF_WINDOW,
        reason,
        n_free=5,
        n_positive_min=15,
        n_negative_saddle=1,
    )


def skipped() -> DirectionalPrefactor:
    """Return the ``skipped`` backward estimate of a forward-only request."""
    return DirectionalPrefactor.not_requested(n_free=5, n_negative_saddle=1)


def event_prefactors(
    event_key: tuple,
    forward: DirectionalPrefactor,
    backward: DirectionalPrefactor,
    settings: HTSTSettings | None = None,
) -> EventPrefactors:
    """Wrap two directional estimates into the worker's ``EventPrefactors``."""
    return EventPrefactors(
        event_key=event_key,
        forward=forward,
        backward=backward,
        method="fd",
        n_free=5,
        settings=settings if settings is not None else HTSTSettings(),
    )


class FakeManager:
    """Manager double: ``submit`` returns a Future, ``broadcast`` records calls.

    Parameters
    ----------
    responder : Callable[[Any], Any] or None
        Called with the submitted request; its return value resolves the
        Future. Raising inside it sets the exception on the Future, which is
        what a failing worker operation looks like through the real manager.
    completion : {"immediate", "reverse"}
        ``"immediate"`` resolves each Future when submitted; ``"reverse"``
        withholds every result until ``expected`` requests are queued, then
        resolves them from the last submitted to the first on a helper thread,
        so a caller reading results in submission order observes out-of-order
        completion.
    expected : int
        Number of submissions the ``"reverse"`` mode waits for.

    """

    def __init__(
        self,
        responder: Callable[[Any], Any] | None = None,
        completion: str = "immediate",
        expected: int = 0,
    ) -> None:
        self.responder = responder
        self.completion = completion
        self.expected = expected
        self.submitted: list[tuple[str, dict[str, Any]]] = []
        self.broadcasts: list[tuple[str, dict[str, Any]]] = []
        self.group_calls: list[tuple[str, dict[str, Any]]] = []
        self.completion_order: list[Any] = []
        self.shutdowns = 0
        self._pending: list[tuple[Future, Any]] = []
        self._lock = threading.Lock()

    def _resolve(self, future: Future, request: Any) -> None:
        """Resolve one Future through the responder and record the order."""
        self.completion_order.append(request.event_key)
        try:
            value = self.responder(request) if self.responder else None
        except Exception as exc:  # noqa: BLE001 - mirrors the worker boundary
            future.set_exception(exc)
        else:
            future.set_result(value)

    def submit(self, op_name: str, **kwargs: Any) -> Future:
        """Queue ``op_name`` and return its Future."""
        self.submitted.append((op_name, kwargs))
        future: Future = Future()
        request = kwargs.get("request")
        if self.completion == "immediate":
            self._resolve(future, request)
            return future
        with self._lock:
            self._pending.append((future, request))
            ready = len(self._pending) >= self.expected
        if ready:
            pending = list(reversed(self._pending))
            threading.Thread(
                target=lambda: [self._resolve(f, r) for f, r in pending],
                daemon=True,
            ).start()
        return future

    def broadcast(self, op_name: str, **kwargs: Any) -> None:
        """Record a broadcast."""
        self.broadcasts.append((op_name, kwargs))

    def submit_group(self, op_name: str, **kwargs: Any) -> None:
        """Record a synchronous group operation."""
        self.group_calls.append((op_name, kwargs))

    def group_minimize_with_results(self, **kwargs: Any) -> tuple[Any, float]:
        """Pretend to minimise: return the given positions and a fixed energy."""
        return kwargs.get("positions"), -1.0

    def group_get_potential_energy(self, **kwargs: Any) -> float:
        """Pretend to evaluate the potential energy."""
        return -1.0

    def shutdown(self) -> None:
        """Pretend to shut the pool down (counted)."""
        self.shutdowns += 1

    @property
    def prefactor_requests(self) -> list[Any]:
        """Return the submitted ``compute_event_prefactors`` requests in order."""
        return [
            kw["request"]
            for op, kw in self.submitted
            if op == "compute_event_prefactors"
        ]

    @property
    def prefactor_backward_flags(self) -> list[bool]:
        """Return the ``compute_backward`` flag of every prefactor submission."""
        return [
            kw["compute_backward"]
            for op, kw in self.submitted
            if op == "compute_event_prefactors"
        ]


def _with_rate_style(config: Config, style: str, k0: float = 1.0) -> Config:
    """Return ``config`` with its rateconstant section replaced."""
    rate = RateConstantConfig(style=style, k0=k0, T=config.rateconstant.T)
    return config.model_copy(update={"rateconstant": rate})


@pytest.fixture
def constant_config() -> Config:
    """Return the committed test input (constant style, k0 as written)."""
    return Config.from_ini_file(DATA_INPUT)


@pytest.fixture
def htst_config() -> Config:
    """Return the committed test input switched to htst with ``k0 = 1.0``."""
    return _with_rate_style(Config.from_ini_file(DATA_INPUT), "htst", k0=1.0)


@pytest.fixture
def rpa_config() -> Config:
    """Return the committed test input switched to rpa with ``k0 = 1.0``."""
    return _with_rate_style(Config.from_ini_file(DATA_INPUT), "rpa", k0=1.0)
