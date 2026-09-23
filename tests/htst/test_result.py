"""Tests for the typed HTST result types and their invariants."""

from __future__ import annotations

import pickle
from typing import Any

import pytest

from pykmc.htst import (
    DirectionalPrefactor,
    EventPrefactors,
    HTSTSettings,
    PrefactorRejected,
    PrefactorRejection,
)


def test_rejection_enum_has_exactly_the_contract_members() -> None:
    """The str-Enum carries the scientific rejection codes and nothing else."""
    assert {m.value for m in PrefactorRejection} == {
        "empty_free_region",
        "unstable_minimum",
        "saddle_not_first_order",
        "mode_count_mismatch",
        "nonfinite_hessian",
        "nonstationary_geometry",
        "nonfinite_prefactor",
        "out_of_window",
    }
    assert isinstance(PrefactorRejection.OUT_OF_WINDOW, str)
    assert PrefactorRejection("out_of_window") is PrefactorRejection.OUT_OF_WINDOW


def test_prefactor_rejected_carries_code_and_detail() -> None:
    """The exception exposes reason_code/detail and a readable str."""
    exc = PrefactorRejected(PrefactorRejection.UNSTABLE_MINIMUM, "one negative mode")
    assert exc.reason_code is PrefactorRejection.UNSTABLE_MINIMUM
    assert exc.detail == "one negative mode"
    assert str(exc) == "unstable_minimum: one negative mode"
    with pytest.raises(TypeError):
        PrefactorRejected("unstable_minimum", "not an enum")  # type: ignore[arg-type]


def test_accepted_direction_requires_finite_positive_nu0() -> None:
    """status='ok' iff nu0_hz is a finite float > 0."""
    ok = DirectionalPrefactor.accepted(
        5.0e12, n_free=4, n_positive_min=12, n_negative_saddle=1
    )
    assert ok.ok and ok.status == "ok" and ok.reason_code is None
    assert ok.nu0_hz == 5.0e12
    for bad in (0.0, -1.0, float("nan"), float("inf"), None):
        with pytest.raises(ValueError):
            DirectionalPrefactor(
                nu0_hz=bad,
                status="ok",
                reason_code=None,
                reason=None,
                n_free=4,
                n_positive_min=12,
                n_negative_saddle=1,
            )


def test_rejected_direction_invariants() -> None:
    """status='rejected' needs nu0 None, a code and a non-empty reason."""
    rej = DirectionalPrefactor.rejected(
        PrefactorRejection.OUT_OF_WINDOW,
        "nu0 too large",
        n_free=4,
        n_positive_min=12,
        n_negative_saddle=1,
    )
    assert not rej.ok and rej.nu0_hz is None
    assert rej.reason_code is PrefactorRejection.OUT_OF_WINDOW
    bad_cases: list[dict[str, Any]] = [
        {"nu0_hz": 1.0},
        {"reason_code": None},
        {"reason": ""},
        {"reason": None},
        {"status": "failed"},
        {"n_negative_saddle": -1},
    ]
    for override in bad_cases:
        fields: dict[str, Any] = {
            "nu0_hz": None,
            "status": "rejected",
            "reason_code": PrefactorRejection.OUT_OF_WINDOW,
            "reason": "x",
            "n_free": 4,
            "n_positive_min": None,
            "n_negative_saddle": None,
        }
        fields.update(override)
        with pytest.raises(ValueError):
            DirectionalPrefactor(**fields)


def test_unknown_negative_mode_count_is_none_not_sentinel() -> None:
    """``n_negative_saddle`` is None when unknown; a -1 sentinel is refused."""
    d = DirectionalPrefactor.accepted(
        1.0e13, n_free=1, n_positive_min=3, n_negative_saddle=None
    )
    assert d.n_negative_saddle is None
    with pytest.raises(ValueError):
        DirectionalPrefactor.accepted(
            1.0e13, n_free=1, n_positive_min=3, n_negative_saddle=-1
        )


def test_event_prefactors_pickle_round_trip_and_equality() -> None:
    """Results and settings pickle round-trip with value equality."""
    settings = HTSTSettings(free_radius=4.0)
    fwd = DirectionalPrefactor.accepted(
        7.0e12, n_free=3, n_positive_min=9, n_negative_saddle=1
    )
    bwd = DirectionalPrefactor.rejected(
        PrefactorRejection.UNSTABLE_MINIMUM,
        "min2 has a negative mode",
        n_free=3,
        n_positive_min=8,
        n_negative_saddle=1,
    )
    res = EventPrefactors(
        event_key=("ref", 3),
        forward=fwd,
        backward=bwd,
        method="fd",
        n_free=3,
        settings=settings,
    )
    back = pickle.loads(pickle.dumps(res))
    assert back == res
    assert back.forward == fwd and back.backward == bwd
    assert back.settings == settings
    assert back.backward.reason_code is PrefactorRejection.UNSTABLE_MINIMUM


def test_event_prefactors_type_checks() -> None:
    """Composite fields are type-checked at construction."""
    fwd = DirectionalPrefactor.accepted(
        7.0e12, n_free=3, n_positive_min=9, n_negative_saddle=1
    )
    with pytest.raises(ValueError):
        EventPrefactors(
            event_key=["ref", 3],  # type: ignore[arg-type]
            forward=fwd,
            backward=fwd,
            method="fd",
            n_free=3,
            settings=HTSTSettings(),
        )
    with pytest.raises(ValueError):
        EventPrefactors(
            event_key=("ref", 3),
            forward=fwd,
            backward=fwd,
            method="",
            n_free=3,
            settings=HTSTSettings(),
        )
    with pytest.raises(ValueError):
        EventPrefactors(
            event_key=("ref", 3),
            forward=fwd,
            backward=fwd,
            method="fd",
            n_free=-1,
            settings=HTSTSettings(),
        )


def test_prefactor_rejected_none_detail_is_stored_as_empty_string() -> None:
    """A ``None`` detail never becomes the truthy literal ``'None'``."""
    exc = PrefactorRejected(PrefactorRejection.NONFINITE_HESSIAN, None)
    assert exc.detail == ""
    assert str(exc) == "nonfinite_hessian: "
    assert (exc.detail or exc.reason_code.value) == "nonfinite_hessian"


def test_skipped_direction_invariants() -> None:
    """status='skipped' carries no estimate, no code and a non-empty reason."""
    skipped = DirectionalPrefactor.not_requested(n_free=4, n_negative_saddle=1)
    assert skipped.status == "skipped"
    assert skipped.skipped and not skipped.ok
    assert skipped.nu0_hz is None and skipped.reason_code is None
    assert skipped.reason == "not requested"
    assert skipped.n_free == 4 and skipped.n_negative_saddle == 1
    assert skipped.n_positive_min is None
    assert pickle.loads(pickle.dumps(skipped)) == skipped
    fields: dict[str, Any] = {
        "nu0_hz": None,
        "status": "skipped",
        "reason_code": None,
        "reason": "not requested",
        "n_free": 4,
        "n_positive_min": None,
        "n_negative_saddle": 1,
    }
    assert DirectionalPrefactor(**fields) == skipped
    for override in (
        {"nu0_hz": 1.0e12},
        {"reason_code": PrefactorRejection.OUT_OF_WINDOW},
        {"reason": ""},
        {"reason": None},
        {"n_positive_min": 3},
    ):
        with pytest.raises(ValueError):
            DirectionalPrefactor(**{**fields, **override})
    # accepted and rejected results never report skipped
    assert not DirectionalPrefactor.accepted(
        1.0e13, n_free=1, n_positive_min=3, n_negative_saddle=1
    ).skipped
