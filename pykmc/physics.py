"""Immutable physical identity and resolved atom-constraint transport.

These contracts describe a calculation; they do not turn an unknown historical
estimate into a current one. Cache consumers must retain the producing descriptor.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import shlex
from typing import Any

import numpy as np


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _indices(values: Any, *, upper: int | None = None) -> tuple[int, ...]:
    raw = tuple(values)
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) for i in raw):
        raise ValueError("atom identities and indices must be integers")
    result = tuple(int(i) for i in raw)
    if len(set(result)) != len(result) or any(
        i < 0 or (upper is not None and i >= upper) for i in result
    ):
        raise ValueError("atom identities/indices must be unique and in range")
    return result


@dataclass(frozen=True)
class ForceModel:
    """Content-addressed LAMMPS force definition in metal/atomic units.

    Known one-file styles and numeric LJ/zero definitions are fingerprinted.
    Other styles remain usable for fresh calculations but explicitly non-reusable:
    their commands may hide dependencies this parser cannot establish.
    """

    style: tuple[str, ...]
    coefficients: tuple[str, ...]
    file_digests: tuple[tuple[int, str], ...] = ()
    limitation: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not all(
            isinstance(seq, tuple) and all(isinstance(s, str) for s in seq)
            for seq in (self.style, self.coefficients)
        ):
            raise ValueError("force-model commands must be immutable string tuples")
        if not isinstance(self.file_digests, tuple) or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], int)
            or isinstance(item[0], bool)
            or not isinstance(item[1], str)
            for item in self.file_digests
        ):
            raise ValueError("file digests must be immutable index/digest pairs")
        if self.limitation is not None and not isinstance(self.limitation, str):
            raise ValueError("force-model limitation must be text")

    @classmethod
    def capture(cls, config: Any) -> ForceModel:
        style = tuple(shlex.split(str(config.pair_style)))
        coeff = list(shlex.split(str(config.pair_coeff)))
        hashes: list[tuple[int, str]] = []
        limitation = None
        file_styles = {"sw", "tersoff", "eam", "eam/alloy", "eam/fs"}
        numeric_styles = {"lj/cut", "zero"}
        name = style[0] if style else ""
        if any(
            x in str(config.pair_style) + str(config.pair_coeff)
            for x in ("$", "\n", "&")
        ):
            limitation = "dynamic or multi-command force definition"
        elif name in file_styles and len(coeff) >= 3:
            path = Path(coeff[2])
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                limitation = "force-model file is unavailable"
            else:
                hashes.append((2, digest))
                coeff[2] = "sha256:" + digest
        elif name in numeric_styles:
            try:
                numbers = [float(v) for v in (*style[1:], *coeff[2:])]
                if not all(math.isfinite(v) for v in numbers):
                    raise ValueError
            except ValueError:
                limitation = "unresolved numeric force definition"
        else:
            limitation = "force-model dependencies are not fingerprintable"
        return cls(style, tuple(coeff), tuple(hashes), limitation)

    @property
    def reusable(self) -> bool:
        return self.limitation is None

    @property
    def identity(self) -> tuple:
        return (
            self.schema_version,
            "metal",
            "atomic",
            self.style,
            self.coefficients,
            self.file_digests,
            self.limitation,
        )


@dataclass(frozen=True)
class EnginePhysics:
    """Authoritative full potential type map captured after potential initialization."""

    species: tuple[str, ...]
    masses: tuple[float, ...]
    force_model: ForceModel
    premin_solver: tuple[str, str]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.species, tuple)
            or not self.species
            or not all(isinstance(s, str) and s for s in self.species)
            or len(set(self.species)) != len(self.species)
        ):
            raise ValueError("species must be a nonempty unique tuple of symbols")
        if (
            not isinstance(self.masses, tuple)
            or len(self.masses) != len(self.species)
            or any(
                isinstance(m, (bool, np.bool_))
                or not isinstance(m, Real)
                or not math.isfinite(m)
                or m <= 0
                for m in self.masses
            )
        ):
            raise ValueError(
                "masses must be a tuple of positive finite numbers in species order"
            )
        object.__setattr__(self, "masses", tuple(float(m) for m in self.masses))
        if (
            not isinstance(self.force_model, ForceModel)
            or not isinstance(self.premin_solver, tuple)
            or len(self.premin_solver) != 2
            or not all(isinstance(s, str) for s in self.premin_solver)
        ):
            raise ValueError("invalid force model or premin solver")

    @classmethod
    def capture(cls, config: Any, species: Any, masses: Any) -> EnginePhysics:
        return cls(
            tuple(species),
            tuple(masses),
            ForceModel.capture(config),
            (str(config.min_style), str(config.frz_min)),
        )


@dataclass(frozen=True)
class DescriptorComparison:
    """Comparison only; the caller owns invalidation/recompute/fallback policy."""

    status: str
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class PhysicalDescriptor:
    """Versioned producing physics, excluding rate and acceptance-window policy."""

    engine: EnginePhysics
    numerical: tuple[tuple[str, Any], ...]
    constraint_policy: str
    premin: tuple[str, str] | None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not isinstance(self.engine, EnginePhysics):
            raise ValueError("unsupported physical descriptor")
        if not isinstance(self.numerical, tuple) or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], (str, float, int, bool, type(None)))
            for item in self.numerical
        ):
            raise ValueError("numerical settings must be immutable scalar pairs")
        required = {
            "free_radius",
            "free_region_center",
            "fd_step",
            "zone_radius",
            "premin",
            "zero_mode_tol",
        }
        if (
            len(self.numerical) != len(required)
            or set(dict(self.numerical)) != required
        ):
            raise ValueError("physical descriptor needs all numerical settings")
        if not isinstance(self.constraint_policy, str):
            raise ValueError("constraint policy must be immutable JSON text")
        json.loads(self.constraint_policy)
        if self.premin is not None and (
            not isinstance(self.premin, tuple)
            or len(self.premin) != 2
            or not all(isinstance(s, str) for s in self.premin)
        ):
            raise ValueError("premin inputs must be an immutable solver tuple")
        if bool(dict(self.numerical)["premin"]) != (self.premin is not None):
            raise ValueError("premin inputs are required exactly when enabled")
        _digest(self.numerical)

    @classmethod
    def from_config(
        cls, config: Any, engine: EnginePhysics, settings: Any
    ) -> PhysicalDescriptor:
        numerical = tuple(
            (key, getattr(settings, key))
            for key in (
                "free_radius",
                "free_region_center",
                "fd_step",
                "zone_radius",
                "premin",
                "zero_mode_tol",
            )
        )
        region = getattr(config, "frozen_atoms", None)
        policy = json.dumps(
            None if region is None else region.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return cls(
            engine, numerical, policy, engine.premin_solver if settings.premin else None
        )

    @property
    def descriptor_id(self) -> str:
        return _digest(
            (
                self.schema_version,
                self.engine.species,
                self.engine.masses,
                self.engine.force_model.identity,
                self.numerical,
                self.constraint_policy,
                self.premin,
            )
        )

    @property
    def reusable(self) -> bool:
        return self.engine.force_model.reusable

    def numerical_settings(self) -> dict[str, Any]:
        return dict(self.numerical)

    def resolve_constraints(
        self, positions: Any, types: Any, atom_ids: Any = None
    ) -> ResolvedConstraints:
        """Resolve the captured policy, immune to later config-object mutation."""
        from .config import RegionConfig

        policy = json.loads(self.constraint_policy)
        region = None if policy is None else RegionConfig.model_validate(policy)
        return ResolvedConstraints.resolve(positions, types, region, atom_ids)

    def compare(self, other: PhysicalDescriptor | None) -> DescriptorComparison:
        if other is None:
            return DescriptorComparison("unknown", ("missing producing descriptor",))
        if not isinstance(other, PhysicalDescriptor):
            return DescriptorComparison(
                "unknown", ("unsupported producing descriptor",)
            )
        if not self.reusable or not other.reusable:
            return DescriptorComparison(
                "unknown", ("unfingerprintable force-model dependencies",)
            )
        if self.descriptor_id == other.descriptor_id:
            return DescriptorComparison("compatible")
        return DescriptorComparison(
            "incompatible", ("producing physical descriptor differs",)
        )


@dataclass(frozen=True)
class ResolvedConstraints:
    """Global source identities, crop correspondence and immutable fixed coordinates.

    ``source_ids`` names the full ordering; ``atom_ids`` names local rows, including
    after a crop. Frozen coordinates stay in the source frame. This endpoint
    restriction is separate from the Hessian's common free-atom set.
    """

    source_ids: tuple[int, ...]
    atom_ids: tuple[int, ...]
    fixed_ids: tuple[int, ...]
    fixed_positions: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        self.validate(len(self.atom_ids))

    @classmethod
    def resolve(
        cls, positions: Any, types: Any, region: Any = None, atom_ids: Any = None
    ) -> ResolvedConstraints:
        pos = np.asarray(positions, dtype=float)
        if (
            pos.ndim != 2
            or pos.shape[1] != 3
            or len(pos) == 0
            or not np.all(np.isfinite(pos))
            or len(types) != len(pos)
        ):
            raise ValueError(
                "constraint source requires matching finite (N,3) positions and types"
            )
        ids = _indices(range(len(pos)) if atom_ids is None else atom_ids)
        if len(ids) != len(pos):
            raise ValueError("source identity count must match positions")
        if region is None:
            indices = ()
        else:
            _indices(region.indices, upper=len(pos))
            from .environments.region import region as classify_region

            indices = tuple(
                i
                for i, selected in enumerate(classify_region(region, pos, list(types)))
                if selected == "in"
            )
        return cls(
            ids,
            ids,
            tuple(ids[i] for i in indices),
            tuple(tuple(float(x) for x in pos[i]) for i in indices),
        )

    def validate(self, n_atoms: int) -> None:
        if not all(
            isinstance(v, tuple)
            for v in (
                self.source_ids,
                self.atom_ids,
                self.fixed_ids,
                self.fixed_positions,
            )
        ):
            raise ValueError("resolved constraints require immutable tuples")
        source = _indices(self.source_ids)
        atoms = _indices(self.atom_ids)
        fixed = _indices(self.fixed_ids)
        if (
            not source
            or len(atoms) != n_atoms
            or not set(atoms).issubset(source)
            or not set(fixed).issubset(source)
        ):
            raise ValueError("invalid source/global-to-crop constraint mapping")
        if len(self.fixed_positions) != len(fixed) or any(
            not isinstance(p, tuple)
            or len(p) != 3
            or not all(
                isinstance(v, (int, float))
                and not isinstance(v, bool)
                and math.isfinite(v)
                for v in p
            )
            for p in self.fixed_positions
        ):
            raise ValueError(
                "fixed reference coordinates must be finite immutable triples"
            )

    def crop(self, local_indices: Any) -> ResolvedConstraints:
        indices = _indices(local_indices, upper=len(self.atom_ids))
        return replace(self, atom_ids=tuple(self.atom_ids[i] for i in indices))

    @property
    def local_fixed_indices(self) -> tuple[int, ...]:
        return tuple(
            i for i, atom_id in enumerate(self.atom_ids) if atom_id in self.fixed_ids
        )

    @property
    def constraint_id(self) -> str:
        return _digest(
            (self.source_ids, self.atom_ids, self.fixed_ids, self.fixed_positions)
        )


@dataclass(frozen=True)
class CalculationIdentity:
    """Conservative full-geometry dependency record; transport keys are excluded."""

    descriptor_id: str | None
    direction: str
    geometry_digest: str
    atom_ids: tuple[int, ...]
    center_id: int
    free_ids: tuple[int, ...] | None
    constraint_id: str | None

    @property
    def identity_id(self) -> str:
        return _digest(
            (
                self.descriptor_id,
                self.direction,
                self.geometry_digest,
                self.atom_ids,
                self.center_id,
                self.free_ids,
                self.constraint_id,
            )
        )
