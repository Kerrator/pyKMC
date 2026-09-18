"""Typed directional results of an HTST prefactor calculation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from .settings import HTSTSettings


class PrefactorRejection(str, Enum):
    """Scientific reasons for not producing a Vineyard prefactor.

    These are diagnosed physical conditions that map to a per-direction fallback at
    the consumer boundary. Programming and contract errors are never encoded here;
    they raise.
    """

    EMPTY_FREE_REGION = "empty_free_region"
    UNSTABLE_MINIMUM = "unstable_minimum"
    SADDLE_NOT_FIRST_ORDER = "saddle_not_first_order"
    MODE_COUNT_MISMATCH = "mode_count_mismatch"
    NONFINITE_HESSIAN = "nonfinite_hessian"
    NONFINITE_PREFACTOR = "nonfinite_prefactor"
    OUT_OF_WINDOW = "out_of_window"


class PrefactorRejected(Exception):
    """A kernel declined to produce a prefactor for a scientific reason.

    Parameters
    ----------
    reason_code : PrefactorRejection
        The rejection category.
    detail : str or None
        Human-readable explanation with the numbers that triggered it. ``None`` is
        stored as ``""`` so consumers can fall back to ``reason_code.value``.

    """

    def __init__(self, reason_code: PrefactorRejection, detail: str | None) -> None:
        if not isinstance(reason_code, PrefactorRejection):
            raise TypeError(
                f"reason_code must be a PrefactorRejection, got {reason_code!r}"
            )
        text = "" if detail is None else str(detail)
        super().__init__(reason_code, text)
        self.reason_code = reason_code
        self.detail = text

    def __str__(self) -> str:
        """Return ``"<code>: <detail>"``."""
        return f"{self.reason_code.value}: {self.detail}"


def _is_finite_positive_float(value: object) -> bool:
    """Return True when ``value`` is a real (non-bool) finite number > 0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and value > 0.0


def _check_optional_count(name: str, value: int | None) -> None:
    """Raise ``ValueError`` unless ``value`` is ``None`` or a non-negative int."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be None or a non-negative int, got {value!r}")


@dataclass(frozen=True)
class DirectionalPrefactor:
    """One direction's Vineyard estimate or its diagnosed rejection.

    Attributes
    ----------
    nu0_hz : float or None
        Linear Vineyard frequency in Hz; a finite float > 0 iff ``status == "ok"``.
    status : {"ok", "rejected"}
        Whether ``nu0_hz`` was produced (enforced at construction).
    reason_code : PrefactorRejection or None
        Set iff rejected.
    reason : str or None
        Human-readable detail; ``None`` when ok.
    n_free : int or None
        Number of free atoms in the partial Hessian, when known.
    n_positive_min : int or None
        Stable modes at this direction's minimum, when its spectrum was computed.
    n_negative_saddle : int or None
        Unstable modes at the saddle, when its spectrum was computed; ``None`` when
        not determined, never a sentinel.

    """

    nu0_hz: float | None
    status: Literal["ok", "rejected"]
    reason_code: PrefactorRejection | None
    reason: str | None
    n_free: int | None
    n_positive_min: int | None
    n_negative_saddle: int | None

    def __post_init__(self) -> None:
        """Enforce the ok/rejected invariants."""
        if self.status not in ("ok", "rejected"):
            raise ValueError(f"status must be 'ok' or 'rejected', got {self.status!r}")
        if self.status == "ok":
            if not _is_finite_positive_float(self.nu0_hz):
                raise ValueError(
                    f"status 'ok' requires a finite nu0_hz > 0, got {self.nu0_hz!r}"
                )
            if self.reason_code is not None:
                raise ValueError("status 'ok' must not carry a reason_code")
        else:
            if self.nu0_hz is not None:
                raise ValueError("status 'rejected' must carry nu0_hz=None")
            if not isinstance(self.reason_code, PrefactorRejection):
                raise ValueError(
                    "status 'rejected' requires a PrefactorRejection reason_code, "
                    f"got {self.reason_code!r}"
                )
            if not isinstance(self.reason, str) or not self.reason:
                raise ValueError("status 'rejected' requires a non-empty reason")
        _check_optional_count("n_free", self.n_free)
        _check_optional_count("n_positive_min", self.n_positive_min)
        _check_optional_count("n_negative_saddle", self.n_negative_saddle)

    @property
    def ok(self) -> bool:
        """Return True when this direction carries an accepted prefactor."""
        return self.status == "ok"

    @classmethod
    def accepted(
        cls,
        nu0_hz: float,
        *,
        n_free: int | None,
        n_positive_min: int | None,
        n_negative_saddle: int | None,
    ) -> DirectionalPrefactor:
        """Build an ``"ok"`` result.

        Parameters
        ----------
        nu0_hz : float
            Accepted linear frequency in Hz.
        n_free, n_positive_min, n_negative_saddle : int or None
            Diagnostics, see the class attributes.

        Returns
        -------
        DirectionalPrefactor
            The accepted result.

        """
        return cls(
            nu0_hz=float(nu0_hz),
            status="ok",
            reason_code=None,
            reason=None,
            n_free=n_free,
            n_positive_min=n_positive_min,
            n_negative_saddle=n_negative_saddle,
        )

    @classmethod
    def rejected(
        cls,
        reason_code: PrefactorRejection,
        reason: str,
        *,
        n_free: int | None,
        n_positive_min: int | None,
        n_negative_saddle: int | None,
    ) -> DirectionalPrefactor:
        """Build a ``"rejected"`` result.

        Parameters
        ----------
        reason_code : PrefactorRejection
            The rejection category.
        reason : str
            Human-readable detail.
        n_free, n_positive_min, n_negative_saddle : int or None
            Diagnostics known at the point of rejection.

        Returns
        -------
        DirectionalPrefactor
            The rejected result.

        """
        return cls(
            nu0_hz=None,
            status="rejected",
            reason_code=reason_code,
            reason=reason,
            n_free=n_free,
            n_positive_min=n_positive_min,
            n_negative_saddle=n_negative_saddle,
        )


@dataclass(frozen=True)
class EventPrefactors:
    """Forward and backward Vineyard estimates of one event sharing one saddle.

    Attributes
    ----------
    event_key : tuple
        The request's key, echoed unchanged.
    forward : DirectionalPrefactor
        ``min1 -> saddle``.
    backward : DirectionalPrefactor
        ``min2 -> saddle``.
    method : str
        How the Hessians were obtained, e.g. ``"fd"`` or ``"lammps_eskm"``.
    n_free : int
        Number of free atoms shared by every Hessian of this event.
    settings : HTSTSettings
        The settings the calculation used.

    """

    event_key: tuple
    forward: DirectionalPrefactor
    backward: DirectionalPrefactor
    method: str
    n_free: int
    settings: HTSTSettings

    def __post_init__(self) -> None:
        """Type-check the composite fields."""
        if not isinstance(self.event_key, tuple):
            raise ValueError("event_key must be a tuple")
        if not isinstance(self.forward, DirectionalPrefactor) or not isinstance(
            self.backward, DirectionalPrefactor
        ):
            raise ValueError("forward and backward must be DirectionalPrefactor")
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("method must be a non-empty str")
        if isinstance(self.n_free, bool) or not isinstance(self.n_free, int):
            raise ValueError("n_free must be an int")
        if self.n_free < 0:
            raise ValueError("n_free must be >= 0")
        if not isinstance(self.settings, HTSTSettings):
            raise ValueError("settings must be an HTSTSettings")


__all__ = [
    "DirectionalPrefactor",
    "EventPrefactors",
    "PrefactorRejected",
    "PrefactorRejection",
]
