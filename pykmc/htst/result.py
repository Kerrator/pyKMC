"""Typed directional results of an HTST prefactor calculation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Literal, cast

from .settings import HTSTSettings
from .provenance import CalculationProvenance
from ..physics import CalculationIdentity, _digest


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
    NONSTATIONARY_GEOMETRY = "nonstationary_geometry"
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
    status : {"ok", "rejected", "skipped"}
        Whether ``nu0_hz`` was produced (enforced at construction). ``"skipped"``
        means the direction was not requested (``compute_backward=False``): no
        Hessian was computed for it, it carries no estimate and no rejection
        code, and every consumer treats it as "no estimate". Tables never
        store it.
    reason_code : PrefactorRejection or None
        Set iff rejected.
    reason : str or None
        Human-readable detail; ``None`` when ok, ``"not requested"`` when
        skipped.
    n_free : int or None
        Number of free atoms in the partial Hessian, when known.
    n_positive_min : int or None
        Stable modes at this direction's minimum, when its spectrum was computed.
    n_negative_saddle : int or None
        Unstable modes at the saddle, when its spectrum was computed; ``None`` when
        not determined, never a sentinel.

    """

    nu0_hz: float | None
    status: Literal["ok", "rejected", "skipped"]
    reason_code: PrefactorRejection | None
    reason: str | None
    n_free: int | None
    n_positive_min: int | None
    n_negative_saddle: int | None

    def __post_init__(self) -> None:
        """Enforce the ok/rejected/skipped invariants."""
        if self.status not in ("ok", "rejected", "skipped"):
            raise ValueError(
                f"status must be 'ok', 'rejected' or 'skipped', got {self.status!r}"
            )
        if self.status == "ok":
            if not _is_finite_positive_float(self.nu0_hz):
                raise ValueError(
                    f"status 'ok' requires a finite nu0_hz > 0, got {self.nu0_hz!r}"
                )
            if self.reason_code is not None:
                raise ValueError("status 'ok' must not carry a reason_code")
        elif self.status == "skipped":
            if self.nu0_hz is not None:
                raise ValueError("status 'skipped' must carry nu0_hz=None")
            if self.reason_code is not None:
                raise ValueError("status 'skipped' must not carry a reason_code")
            if not isinstance(self.reason, str) or not self.reason:
                raise ValueError("status 'skipped' requires a non-empty reason")
            if self.n_positive_min is not None:
                raise ValueError("status 'skipped' never computed a minimum spectrum")
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

    @property
    def skipped(self) -> bool:
        """Return True when this direction was not requested (no estimate)."""
        return self.status == "skipped"

    @classmethod
    def not_requested(
        cls, *, n_free: int | None, n_negative_saddle: int | None
    ) -> DirectionalPrefactor:
        """Build a ``"skipped"`` result for a direction that was not computed.

        Parameters
        ----------
        n_free, n_negative_saddle : int or None
            Diagnostics shared with the computed direction (the free set and
            the saddle spectrum are common to both directions).

        Returns
        -------
        DirectionalPrefactor
            ``status="skipped"``, ``reason="not requested"``, no estimate.

        """
        return cls(
            nu0_hz=None,
            status="skipped",
            reason_code=None,
            reason="not requested",
            n_free=n_free,
            n_positive_min=None,
            n_negative_saddle=n_negative_saddle,
        )

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
class DirectionalCalculation:
    """An actual directional result bound to its immutable producing inputs."""

    direction: Literal["forward", "backward"]
    provenance: CalculationProvenance
    estimate: DirectionalPrefactor

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.direction not in ("forward", "backward"):
            raise ValueError("direction must be forward or backward")
        if not isinstance(self.provenance, CalculationProvenance):
            raise ValueError("calculation needs producing provenance")
        self.provenance.validate()
        if not isinstance(self.estimate, DirectionalPrefactor) or self.estimate.skipped:
            raise ValueError("calculation needs an actual directional estimate")
        self.estimate.__post_init__()
        if self.estimate.n_free != len(self.provenance.free_indices):
            raise ValueError("estimate free count disagrees with producing free set")

    @property
    def identity(self) -> CalculationIdentity:
        return self.provenance.directional_identity(self.direction)

    @property
    def descriptor_id(self) -> str | None:
        descriptor = self.provenance.produced.descriptor
        return None if descriptor is None else descriptor.descriptor_id

    @property
    def calculation_id(self) -> str:
        return _digest(
            (self.provenance.provenance_id, self.direction, asdict(self.estimate))
        )

    @property
    def reusable(self) -> bool:
        return self.provenance.reusable and self.estimate.ok


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
    provenance: CalculationProvenance | None = field(default=None, compare=False)

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
        if self.provenance is not None:
            if not isinstance(self.provenance, CalculationProvenance):
                raise ValueError("provenance must be CalculationProvenance or None")
            self.provenance.validate()
            if self.provenance.method != self.method:
                raise ValueError("result method disagrees with producing method")
            if len(self.provenance.free_indices) != self.n_free:
                raise ValueError("result free count disagrees with producing free set")
            if self.provenance.produced.settings != self.settings:
                raise ValueError("result settings disagree with producing settings")

    def calculation(self, direction: str) -> DirectionalCalculation | None:
        """Return truthful producing evidence, never inferred service context."""
        if direction not in ("forward", "backward"):
            raise ValueError("direction must be forward or backward")
        estimate = getattr(self, direction)
        if self.provenance is None or estimate.skipped:
            return None
        return DirectionalCalculation(
            cast(Literal["forward", "backward"], direction), self.provenance, estimate
        )


__all__ = [
    "DirectionalPrefactor",
    "DirectionalCalculation",
    "EventPrefactors",
    "PrefactorRejected",
    "PrefactorRejection",
]
