"""Immutable inputs and actual producing context for a prefactor calculation.

The source snapshot precedes preprocessing; the produced snapshot is the full
geometry after preprocessing, before any Hessian crop. Neither a local crop nor
a transport key can stand in for the full source needed for recomputation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
import math
from typing import Any

import numpy as np

from .request import HTSTEventRequest
from .settings import HTSTSettings
from ..physics import (
    CalculationIdentity,
    PhysicalDescriptor,
    ResolvedConstraints,
    _digest,
    _indices,
)


def _coordinates(value: Any) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(x) for x in row) for row in np.asarray(value))


def _require_immutable(value: Any) -> None:
    """Reject mutable leaves, including scalar arrays hidden inside tuples."""
    if value is None or isinstance(
        value, (str, int, float, bool, np.integer, np.floating, np.bool_)
    ):
        return
    if isinstance(value, tuple):
        for item in value:
            _require_immutable(item)
        return
    if is_dataclass(value) and value.__dataclass_params__.frozen:
        for item in fields(value):
            _require_immutable(getattr(value, item.name))
        return
    raise ValueError("snapshot context must contain only immutable scalar values")


@dataclass(frozen=True)
class RequestSnapshot:
    """A detached immutable request, with no batch/event transport identity."""

    min1_positions: tuple[tuple[float, ...], ...]
    saddle_positions: tuple[tuple[float, ...], ...]
    min2_positions: tuple[tuple[float, ...], ...]
    types: tuple[str, ...]
    species: tuple[str, ...]
    masses: tuple[float, ...]
    cell: tuple[tuple[float, ...], ...]
    pbc: tuple[bool, bool, bool]
    center_index: int
    settings: HTSTSettings
    descriptor: PhysicalDescriptor | None
    constraints: ResolvedConstraints | None
    user_constraints: ResolvedConstraints | None

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def capture(cls, request: HTSTEventRequest) -> RequestSnapshot:
        """Copy validated geometry into immutable scalar tuples."""
        request.validate()
        return cls(
            _coordinates(request.min1_positions),
            _coordinates(request.saddle_positions),
            _coordinates(request.min2_positions),
            tuple(request.types),
            tuple(request.species),
            tuple(float(m) for m in request.masses),
            _coordinates(request.cell),
            tuple(bool(p) for p in request.pbc),
            int(request.center_index),
            request.settings,
            request.descriptor,
            request.constraints,
            request.user_constraints,
        )

    def to_request(self, *, event_key: tuple = ()) -> HTSTEventRequest:
        """Return fresh arrays; editing them cannot alter the stored snapshot."""
        return HTSTEventRequest(
            event_key=event_key,
            min1_positions=np.array(self.min1_positions, dtype=float),
            saddle_positions=np.array(self.saddle_positions, dtype=float),
            min2_positions=np.array(self.min2_positions, dtype=float),
            types=self.types,
            species=self.species,
            masses=self.masses,
            cell=np.array(self.cell, dtype=float),
            pbc=self.pbc,
            center_index=self.center_index,
            settings=self.settings,
            descriptor=self.descriptor,
            constraints=self.constraints,
            user_constraints=self.user_constraints,
        )

    def validate(self) -> None:
        """Validate immutable storage, also when reading a persisted object."""
        _require_immutable(self)
        for name in ("min1_positions", "saddle_positions", "min2_positions", "cell"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(
                not isinstance(row, tuple) for row in values
            ):
                raise ValueError("snapshot coordinates must be immutable tuples")
        for name in ("types", "species", "masses", "pbc"):
            if not isinstance(getattr(self, name), tuple):
                raise ValueError(f"snapshot {name} must be an immutable tuple")
        self.to_request().validate()

    @property
    def is_complete(self) -> bool:
        """Whether the geometry includes every declared source identity."""
        return self.constraints is None or (
            set(self.constraints.atom_ids) == set(self.constraints.source_ids)
        )

    @property
    def snapshot_id(self) -> str:
        """Content identity including original authority and numerical inputs."""
        request = self.to_request()
        identity = request.calculation_identity()
        return _digest(
            (
                identity.geometry_digest,
                identity.atom_ids,
                identity.center_id,
                identity.constraint_id,
                identity.descriptor_id,
                None
                if self.user_constraints is None
                else self.user_constraints.constraint_id,
                asdict(self.settings),
            )
        )


@dataclass(frozen=True)
class CalculationProvenance:
    """Full source, actual prepared geometry and actual Hessian correspondence."""

    source: RequestSnapshot
    produced: RequestSnapshot
    method: str
    free_indices: tuple[int, ...]
    zone_indices: tuple[int, ...]
    energies: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def capture(
        cls,
        source: HTSTEventRequest,
        produced: HTSTEventRequest,
        *,
        method: str,
        free_indices: Any,
        zone_indices: Any = None,
        energies: Any = None,
    ) -> CalculationProvenance:
        """Capture actual selections; do not infer them from a free-atom count."""
        count = len(source.types)
        free = _indices(free_indices, upper=count)
        zone = (
            tuple(range(count))
            if zone_indices is None
            else _indices(zone_indices, upper=count)
        )
        if energies is not None:
            energies = tuple(energies)
            if any(isinstance(e, (bool, np.bool_)) for e in energies):
                raise ValueError("producing energies must be finite real values")
            energies = tuple(float(e) for e in energies)
        return cls(
            RequestSnapshot.capture(source),
            RequestSnapshot.capture(produced),
            method,
            free,
            zone,
            energies,
        )

    def validate(self) -> None:
        """Reject mismatched source authority, mappings or nonfinite energies."""
        if not isinstance(self.source, RequestSnapshot) or not isinstance(
            self.produced, RequestSnapshot
        ):
            raise ValueError("provenance needs source and produced snapshots")
        self.source.validate()
        self.produced.validate()
        for name in (
            "types",
            "species",
            "masses",
            "cell",
            "pbc",
            "center_index",
            "settings",
            "descriptor",
            "constraints",
            "user_constraints",
        ):
            if getattr(self.source, name) != getattr(self.produced, name):
                raise ValueError(f"producing context changed {name}")
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("producing method must be a nonempty string")
        count = len(self.source.types)
        for name in ("free_indices", "zone_indices"):
            value = getattr(self, name)
            if (
                not isinstance(value, tuple)
                or any(type(i) is not int for i in value)
                or _indices(value, upper=count) != value
            ):
                raise ValueError(f"{name} must be immutable unique source rows")
        if not set(self.free_indices).issubset(self.zone_indices):
            raise ValueError("producing crop must contain every free row")
        fixed = (
            ()
            if self.source.constraints is None
            else self.source.constraints.local_fixed_indices
        )
        if set(self.free_indices).intersection(fixed):
            raise ValueError("producing free set contains fixed source rows")
        if self.energies is not None and (
            not isinstance(self.energies, tuple)
            or len(self.energies) != 3
            or any(
                isinstance(e, (bool, np.bool_))
                or not isinstance(e, (float, int))
                or not math.isfinite(e)
                for e in self.energies
            )
        ):
            raise ValueError("producing energies must be three finite eV values")

    @property
    def reusable(self) -> bool:
        """Only known complete physics can support subsequent cache reuse."""
        descriptor = self.produced.descriptor
        return (
            self.source.is_complete
            and self.produced.is_complete
            and descriptor is not None
            and descriptor.reusable
        )

    @property
    def provenance_id(self) -> str:
        return _digest(
            (
                self.source.snapshot_id,
                self.produced.snapshot_id,
                self.method,
                self.free_indices,
                self.zone_indices,
                self.energies,
            )
        )

    def directional_identity(self, direction: str) -> CalculationIdentity:
        """Bind the full produced geometry to actual free global identities."""
        return self.produced.to_request().calculation_identity(
            direction, self.free_indices
        )
