"""Tests for ``HTSTEventRequest`` validation and mass mapping."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from pykmc.htst import (
    HTSTEventRequest,
    HTSTGeometryError,
    HTSTRequestError,
    HTSTSettings,
    compute_event_prefactors,
    default_masses_for_species,
    require_orthorhombic,
)

_POS = np.array([[1.0, 1.0, 1.0], [2.5, 1.0, 1.0], [1.0, 2.5, 1.0]])


def make_request(**overrides: Any) -> HTSTEventRequest:
    """Build a valid three-atom request, overriding any field."""
    fields: dict[str, Any] = {
        "event_key": ("ref", 7, "forward"),
        "min1_positions": _POS.copy(),
        "saddle_positions": _POS + 0.1,
        "min2_positions": _POS + 0.2,
        "types": ("Ni", "Cr", "Ni"),
        "species": ("Ni", "Fe", "Cr"),
        "masses": (58.6934, 55.845, 51.9961),
        "cell": np.diag([10.0, 12.0, 14.0]),
        "pbc": (True, True, False),
        "center_index": 1,
        "settings": HTSTSettings(),
    }
    fields.update(overrides)
    return HTSTEventRequest(**fields)


def test_valid_request_validates_and_is_frozen() -> None:
    """A well-formed request passes and cannot be mutated."""
    req = make_request()
    req.validate()
    with pytest.raises(dataclasses.FrozenInstanceError):
        req.center_index = 0  # type: ignore[misc]


def test_masses_per_atom_follow_species_order_with_absent_species() -> None:
    """Masses map through the full species order even when Fe is absent from types."""
    req = make_request()
    np.testing.assert_allclose(req.masses_per_atom(), [58.6934, 51.9961, 58.6934])


def test_masses_per_atom_rejects_unknown_type() -> None:
    """A type missing from species raises HTSTRequestError from masses_per_atom."""
    req = make_request(types=("Ni", "Al", "Ni"))
    with pytest.raises(HTSTRequestError, match="Al"):
        req.masses_per_atom()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min1_positions", np.zeros((2, 3))),
        ("saddle_positions", np.zeros((3, 2))),
        ("min2_positions", np.zeros(9)),
        ("min1_positions", np.array([[np.nan, 0, 0], [0, 0, 0], [0, 0, 0]])),
        ("saddle_positions", np.array([[np.inf, 0, 0], [0, 0, 0], [0, 0, 0]])),
        ("types", ("Ni", "Cr")),
        ("types", ("Ni", "Al", "Ni")),
        ("types", ["Ni", "Cr", "Ni"]),
        ("species", ()),
        ("species", ("Ni", "Ni", "Cr")),
        ("masses", (58.6934, 55.845)),
        ("masses", (58.6934, 0.0, 51.9961)),
        ("masses", (58.6934, -1.0, 51.9961)),
        ("masses", (58.6934, float("nan"), 51.9961)),
        ("masses", [58.6934, 55.845, 51.9961]),
        ("masses", (58.6934, np.bool_(True), 51.9961)),
        ("masses", (58.6934, True, 51.9961)),
        ("cell", np.diag([10.0, 12.0])),
        ("cell", np.diag([10.0, 12.0, np.nan])),
        ("cell", np.diag([10.0, 0.0, 14.0])),
        ("pbc", (True, True)),
        ("pbc", (1, 1, 0)),
        ("pbc", ("True", "True", "False")),
        ("pbc", [True, True, False]),
        ("center_index", -1),
        ("center_index", 3),
        ("center_index", True),
        ("center_index", 1.0),
        ("event_key", "ref-7"),
        ("settings", None),
    ],
)
def test_invalid_request_raises_request_error(field: str, value: Any) -> None:
    """Shape, index, finiteness, species and mass violations raise HTSTRequestError."""
    req = make_request(**{field: value})
    with pytest.raises(HTSTRequestError):
        req.validate()


_UNHASHABLE_KEYS = {
    "list": ([],),
    "dict": ({},),
    "nested-list": ("ref", 7, [1]),
    "nested-dict": (("ref", {"a": 1}),),
    "ndarray": (np.zeros(3),),
}
_HASHABLE_KEYS = {
    "str-int-str": ("ref", 7, "forward"),
    "ints": (1, 2, 3),
    "float-str-negint": (1.5, "x", -2),
    "nested-tuples": (("ref", 7), ("site", (3.0, "b"))),
    "singleton": ("only",),
}


@pytest.mark.parametrize(
    "event_key", list(_UNHASHABLE_KEYS.values()), ids=list(_UNHASHABLE_KEYS)
)
def test_unhashable_event_key_is_rejected_at_validate(event_key: tuple) -> None:
    """An unhashable tuple fails ``validate()`` before any kernel or set/dict use."""
    req = make_request(event_key=event_key)  # construction itself does not validate
    with pytest.raises(HTSTRequestError, match="hashable"):
        req.validate()

    def never_called(positions: np.ndarray, free_indices: np.ndarray) -> np.ndarray:
        raise AssertionError("hessian_fn must not run for an invalid request")

    with pytest.raises(HTSTRequestError, match="hashable"):
        compute_event_prefactors(req, never_called)


@pytest.mark.parametrize(
    "event_key", list(_HASHABLE_KEYS.values()), ids=list(_HASHABLE_KEYS)
)
def test_hashable_event_keys_validate_and_hash(event_key: tuple) -> None:
    """Tuples of str/int/float/tuples validate, hash, and are echoed unchanged."""
    req = make_request(event_key=event_key)
    req.validate()
    assert req.event_key is event_key
    assert hash(req.event_key) == hash(event_key)
    assert {req.event_key: "v"}[event_key] == "v"
    assert len({req.event_key, event_key}) == 1


def test_numpy_bools_and_numpy_scalar_masses_are_accepted() -> None:
    """``tuple(atoms.pbc)`` (np.bool_) and np.float32/np.int64 masses validate."""
    req = make_request(
        pbc=tuple(np.array([True, False, True])),
        masses=(np.float64(58.6934), np.float32(55.845), np.int64(52)),
    )
    req.validate()
    np.testing.assert_allclose(req.masses_per_atom(), [58.6934, 52.0, 58.6934])


def test_triclinic_cell_raises_geometry_error() -> None:
    """A sheared cell raises HTSTGeometryError, a ValueError and HTSTRequestError."""
    cell = np.array([[10.0, 0.0, 0.0], [2.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    req = make_request(cell=cell)
    with pytest.raises(HTSTGeometryError):
        req.validate()
    assert issubclass(HTSTGeometryError, HTSTRequestError)
    assert issubclass(HTSTRequestError, ValueError)


def test_require_orthorhombic_tolerates_rounding_noise() -> None:
    """Off-diagonal noise below the tolerance is accepted."""
    cell = np.diag([10.0, 10.0, 10.0])
    cell[0, 1] = 1.0e-12
    out = require_orthorhombic(cell)
    assert out.shape == (3, 3)


def test_default_masses_for_species_uses_ase_lazily() -> None:
    """The optional helper looks up standard masses without importing ASE at module load."""
    pytest.importorskip("ase")
    masses = default_masses_for_species(("Ni", "Fe"))
    assert masses[0] == pytest.approx(58.6934, rel=1.0e-4)
    assert masses[1] == pytest.approx(55.845, rel=1.0e-4)
