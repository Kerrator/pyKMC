"""Immutable physical identity and resolved atom-constraint transport.

These contracts describe a calculation; they do not turn an unknown historical
estimate into a current one. Cache consumers must retain the producing descriptor.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import shlex
from typing import Any

import numpy as np


def _digest(value: Any) -> str:
    def immutable_scalar(item: Any) -> Any:
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, np.floating):
            return float(item)
        if isinstance(item, np.bool_):
            return bool(item)
        raise TypeError(f"unsupported physical identity value: {type(item).__name__}")

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=immutable_scalar,
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
            "force_tol",
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
                "force_tol",
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


class ConstraintViolationError(ValueError):
    """An event displaces a fixed reference coordinate beyond the tolerance.

    Raised by :meth:`ResolvedConstraints.validate_positions`. It is the only
    constraint error that names a property of one catalogue row: the
    reconstruction, basin and refinement paths convert it (and only it) to
    ``Err(RECONSTRUCTION_INVALID_EVENT_DATA)`` so the reference is purged
    (contracts 7f policy 5). Every other ``ValueError`` of the constraint
    machinery (shape, cell/PBC or mapping mismatches) is a programming or
    data error and propagates: converting it would purge a reference per
    selection and drain the catalogue. A ``ValueError`` subclass, so callers
    catching the base class keep working.
    """


@dataclass(frozen=True)
class ResolvedConstraints:
    """Global source identities, crop correspondence and immutable fixed coordinates.

    ``source_ids`` names the full ordering; ``atom_ids`` names local rows, including
    after a crop. Frozen coordinates stay in the source frame. This endpoint
    restriction is separate from the Hessian's common free-atom set.
    ``user_policy`` records the original resolver policy before an AV union;
    ``None`` denotes unknown policy for a manually constructed legacy payload.

    ``user_fixed_ids`` names the subset of ``fixed_ids`` the USER declared
    (``config.frozen_atoms``). The remaining fixed identities are the
    active-volume shell: a crop/transport restriction held by ``fix setforce``
    during a search, not a fixed-coordinate contract (contracts 7f policy 5).
    Overlay validation therefore applies to the user subset only; the AV shell
    only enters ``protect_positions`` and the native freeze groups. ``None``
    denotes a legacy payload whose whole fixed set is treated as user-declared.
    The field is derived bookkeeping: it is excluded from equality and from
    ``constraint_id``.
    """

    source_ids: tuple[int, ...]
    atom_ids: tuple[int, ...]
    fixed_ids: tuple[int, ...]
    fixed_positions: tuple[tuple[float, float, float], ...]
    cell: tuple[tuple[float, float, float], ...] | None = None
    pbc: tuple[bool, bool, bool] | None = None
    center_id: int | None = None
    center_position: tuple[float, float, float] | None = None
    rmov: float | None = None
    user_policy: str | None = None
    user_fixed_ids: tuple[int, ...] | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        self.validate(len(self.atom_ids))

    @classmethod
    def resolve(
        cls,
        positions: Any,
        types: Any,
        region: Any = None,
        atom_ids: Any = None,
        *,
        cell: Any = None,
        pbc: Any = None,
        center_id: int | None = None,
        rmov: float | None = None,
    ) -> ResolvedConstraints:
        pos = np.asarray(positions, dtype=float)
        if pos.ndim != 2 or pos.shape[1] != 3 or len(pos) == 0:
            raise ValueError(
                "constraint source requires nonempty (N,3) positions, got shape "
                f"{pos.shape}"
            )
        if not np.all(np.isfinite(pos)):
            # Cause-specific: a NaN/inf coordinate must be named as such before
            # any native scatter (a per-rank "Non-numeric atom coords" error
            # would desynchronise a multi-rank engine).
            n_bad = int(np.count_nonzero(~np.isfinite(pos)))
            raise ValueError(
                f"constraint source positions contain {n_bad} non-finite value(s) "
                "(NaN/inf)"
            )
        if len(types) != len(pos):
            raise ValueError(
                f"constraint source has {len(types)} types for {len(pos)} positions"
            )
        ids = _indices(range(len(pos)) if atom_ids is None else atom_ids)
        if len(ids) != len(pos):
            raise ValueError("source identity count must match positions")
        if (cell is None) != (pbc is None):
            raise ValueError("constraint cell and PBC must be supplied together")
        if cell is not None:
            matrix = np.asarray(cell, dtype=float)
            axes = np.asarray(pbc)
            if axes.ndim == 0:
                axes = np.repeat(axes, 3)
            if (
                matrix.shape != (3, 3)
                or not np.all(np.isfinite(matrix))
                or axes.shape != (3,)
                or axes.dtype.kind != "b"
                or abs(np.linalg.det(matrix)) == 0
            ):
                raise ValueError(
                    "constraints require a finite cell and three boolean PBC axes"
                )
            cell = tuple(tuple(float(x) for x in row) for row in matrix)
            pbc = tuple(bool(x) for x in axes)
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
        # The user-declared subset, recorded before any AV shell is unioned in.
        user_indices = indices
        center_position = None
        if center_id is not None or rmov is not None:
            if (
                center_id is None
                or rmov is None
                or cell is None
                or isinstance(center_id, (bool, np.bool_))
                or not isinstance(center_id, Integral)
                or center_id not in ids
                or isinstance(rmov, (bool, np.bool_))
                or not isinstance(rmov, Real)
                or not math.isfinite(rmov)
                or rmov < 0
            ):
                raise ValueError(
                    "active-volume constraints require a source center, cell/PBC and finite radius"
                )
            from ase.geometry import find_mic

            center_position = tuple(float(x) for x in pos[ids.index(center_id)])
            _, distances = find_mic(pos - center_position, cell, pbc=pbc)
            indices = tuple(
                sorted(set(indices).union(np.flatnonzero(distances > rmov)))
            )
            center_id, rmov = int(center_id), float(rmov)
        return cls(
            ids,
            ids,
            tuple(ids[i] for i in indices),
            tuple(tuple(float(x) for x in pos[i]) for i in indices),
            cell,
            pbc,
            center_id,
            center_position,
            rmov,
            json.dumps(
                None if region is None else region.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            tuple(ids[i] for i in user_indices),
        )

    def validate(self, n_atoms: int) -> None:
        if self.user_policy is not None:
            if not isinstance(self.user_policy, str):
                raise ValueError("constraint user policy must be serialized JSON")
            json.loads(self.user_policy)
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
        if self.user_fixed_ids is not None:
            if not isinstance(self.user_fixed_ids, tuple):
                raise ValueError("user-fixed identities must be an immutable tuple")
            _indices(self.user_fixed_ids)
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
        if (self.cell is None) != (self.pbc is None):
            raise ValueError("constraint cell and PBC must be supplied together")
        if self.cell is not None:
            if (
                not isinstance(self.cell, tuple)
                or len(self.cell) != 3
                or any(not isinstance(row, tuple) or len(row) != 3 for row in self.cell)
                or not np.all(np.isfinite(self.cell))
                or np.linalg.det(self.cell) == 0
                or not isinstance(self.pbc, tuple)
                or len(self.pbc) != 3
                or any(not isinstance(x, bool) for x in self.pbc)
            ):
                raise ValueError("invalid immutable constraint cell/PBC")
        if any(
            x is not None for x in (self.center_id, self.center_position, self.rmov)
        ):
            if (
                self.cell is None
                or self.center_id not in source
                or isinstance(self.center_id, bool)
                or not isinstance(self.center_id, int)
                or not isinstance(self.center_position, tuple)
                or len(self.center_position) != 3
                or not np.all(np.isfinite(self.center_position))
                or isinstance(self.rmov, bool)
                or not isinstance(self.rmov, Real)
                or not math.isfinite(self.rmov)
                or self.rmov < 0
            ):
                raise ValueError("invalid immutable active-volume context")

    def _positions(self, positions: Any) -> np.ndarray:
        pos = np.asarray(positions, dtype=float)
        if pos.shape != (len(self.atom_ids), 3) or not np.all(np.isfinite(pos)):
            raise ValueError("constraints require matching finite local positions")
        return pos

    def validate_positions(
        self,
        positions: Any,
        *,
        cell: Any = None,
        pbc: Any = None,
        tolerance: float = 1e-10,
        user_only: bool = False,
    ) -> None:
        """Reject an event that changes the source's fixed physical coordinates.

        ``tolerance`` is the accepted displacement in Angstrom (exact, 1e-10,
        for native results; ``overlay_tolerance`` for PSR-mapped overlays).
        With ``user_only`` only the user-declared fixed atoms are checked: the
        active-volume shell is not a coordinate contract (contracts 7f policy 5).
        """
        pos = self._positions(positions)
        if (cell is None) != (pbc is None):
            raise ValueError("constraint validation needs cell and PBC together")
        if cell is not None:
            matrix = np.asarray(cell, dtype=float)
            axes = np.asarray(pbc)
            if (
                matrix.shape != (3, 3)
                or not np.all(np.isfinite(matrix))
                or axes.shape != (3,)
                or axes.dtype.kind != "b"
            ):
                raise ValueError("invalid constraint validation cell/PBC")
            if self.cell is not None and (
                not np.array_equal(matrix, self.cell)
                or not np.array_equal(axes, self.pbc)
            ):
                raise ValueError("constraint cell/PBC differs from event context")
        else:
            matrix, axes = self.cell, self.pbc
        _, rows, references = self._local_rows(user_only)
        if rows.size == 0:
            return
        delta = pos[rows] - references
        if matrix is not None:
            from ase.geometry import find_mic

            delta, _ = find_mic(delta, matrix, pbc=axes)
        if np.any(np.linalg.norm(delta, axis=1) > tolerance):
            raise ConstraintViolationError("event changes fixed reference coordinates")

    def protect_positions(
        self, positions: Any, *, user_only: bool = False
    ) -> np.ndarray:
        """Re-clamp fixed rows of a working push/overlay to their reference.

        With ``user_only`` only user-declared rows are re-clamped and the
        active-volume shell keeps the caller's coordinate (a pARTn overlay is
        placed as given and held by ``fix setforce``); endpoint minimisations
        protect the whole transport union.
        """
        pos = self._positions(positions).copy()
        _, rows, references = self._local_rows(user_only)
        if rows.size:
            pos[rows] = references
        return pos

    def user_view(self) -> ResolvedConstraints:
        """Return the user constraint set only: no AV shell, no AV context.

        This is what HTST/RPA requests receive (contracts 7f policy 5): the
        Vineyard free region excludes user-fixed atoms and never the shell.
        """
        user = self._user_ids()
        fixed_ids = tuple(i for i in self.fixed_ids if i in user)
        references = dict(zip(self.fixed_ids, self.fixed_positions))
        return replace(
            self,
            fixed_ids=fixed_ids,
            fixed_positions=tuple(references[i] for i in fixed_ids),
            center_id=None,
            center_position=None,
            rmov=None,
            user_fixed_ids=fixed_ids,
        )

    def _user_ids(self) -> frozenset:
        """User-declared fixed identities; a legacy payload is all user-declared."""
        if self.user_fixed_ids is None:
            return frozenset(self.fixed_ids)
        # Bookkeeping never widens the mask: an identity that is no longer
        # fixed (a hand-narrowed payload) is simply not a constraint.
        return frozenset(self.user_fixed_ids).intersection(self.fixed_ids)

    # -- derived row caches -------------------------------------------------
    # The instance is frozen, so anything derived from its fields is memoised
    # once per instance (a crop or replace() is a new instance with its own
    # cache). Membership is a frozenset and the row scan is a single O(N) pass;
    # the per-row tuple scan it replaces was O(N x F), quadratic under an
    # active-volume mask where F ~ N (contracts 7f policy 5).

    def _cache(self) -> dict:
        cache = self.__dict__.get("_derived")
        if cache is None:
            cache = {}
            object.__setattr__(self, "_derived", cache)
        return cache

    def __getstate__(self) -> dict:
        # Derived caches are rebuilt on demand; keep transport payloads lean.
        return {k: v for k, v in self.__dict__.items() if k != "_derived"}

    def _local_rows(
        self, user_only: bool
    ) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
        """Local rows of the (user-)fixed atoms, as a tuple, an index array and
        their ``(K, 3)`` reference coordinates in that row order."""
        key = "user" if user_only else "fixed"
        cache = self._cache()
        entry = cache.get(key)
        if entry is None:
            members = self._user_ids() if user_only else frozenset(self.fixed_ids)
            local = tuple(
                i for i, atom_id in enumerate(self.atom_ids) if atom_id in members
            )
            references = dict(zip(self.fixed_ids, self.fixed_positions))
            entry = (
                local,
                np.asarray(local, dtype=np.intp),
                np.array(
                    [references[self.atom_ids[i]] for i in local], dtype=float
                ).reshape(-1, 3),
            )
            cache[key] = entry
        return entry

    def require_preserves(
        self, required: ResolvedConstraints, *, cell: Any, pbc: Any
    ) -> None:
        """Require a union mask to retain authoritative user IDs and references."""
        if self.source_ids != required.source_ids:
            raise ValueError("constraint source ordering differs from user authority")
        references = dict(zip(self.fixed_ids, self.fixed_positions))
        if not set(required.fixed_ids).issubset(references):
            raise ValueError("event constraints omit user-fixed source identities")
        # Include references outside a crop: cropping changes atom_ids, never
        # the authoritative full-source fixed mask or its physical coordinates.
        reference_view = replace(required, atom_ids=required.fixed_ids)
        positions = np.array(
            [references[i] for i in required.fixed_ids], dtype=float
        ).reshape((-1, 3))
        reference_view.validate_positions(positions, cell=cell, pbc=pbc)

    def crop(self, local_indices: Any) -> ResolvedConstraints:
        indices = _indices(local_indices, upper=len(self.atom_ids))
        return replace(self, atom_ids=tuple(self.atom_ids[i] for i in indices))

    @property
    def local_fixed_indices(self) -> tuple[int, ...]:
        """Local rows of every fixed atom (user and AV shell); O(N), cached."""
        return self._local_rows(False)[0]

    @property
    def local_user_fixed_indices(self) -> tuple[int, ...]:
        """Local rows of the user-declared fixed atoms (the AV shell excluded)."""
        return self._local_rows(True)[0]

    @property
    def constraint_id(self) -> str:
        return _digest(
            (
                self.source_ids,
                self.atom_ids,
                self.fixed_ids,
                self.fixed_positions,
                self.cell,
                self.pbc,
                self.center_id,
                self.center_position,
                self.rmov,
                self.user_policy,
            )
        )


def resolve_event_constraints(
    config,
    positions,
    types,
    cell,
    pbc,
    center_index,
    atom_ids=None,
    *,
    user_constraints=None,
    active_volume=None,
):
    """Resolve the AV shell without reclassifying initialized user constraints.

    Spatial user policies select identities at initialization. An atom entering
    that region later must not silently become fixed, nor may an existing
    reference coordinate be replaced by its current position.
    """
    ids = _indices(range(len(positions)) if atom_ids is None else atom_ids)
    center_index = _indices((center_index,), upper=len(positions))[0]
    active = config.control.active_volume if active_volume is None else active_volume
    region = getattr(config, "frozen_atoms", None)
    resolved = ResolvedConstraints.resolve(
        positions,
        types,
        region if user_constraints is None else None,
        ids,
        cell=cell,
        pbc=pbc,
        center_id=ids[center_index] if active else None,
        rmov=config.activevolume.rmov if active else None,
    )
    if user_constraints is None:
        return resolved
    user_constraints.validate(len(ids))
    if set(ids) != set(user_constraints.source_ids):
        raise ValueError(
            "event source identities differ from initialized user constraints"
        )
    policy = json.dumps(
        None if region is None else region.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if user_constraints.user_policy not in (None, policy):
        raise ValueError("user constraint policy changed since initialization")
    user = replace(user_constraints, atom_ids=ids)
    user.validate_positions(positions, cell=cell, pbc=pbc)
    references = dict(zip(resolved.fixed_ids, resolved.fixed_positions))
    references.update(zip(user.fixed_ids, user.fixed_positions))
    fixed_ids = tuple(i for i in user.source_ids if i in references)
    return replace(
        resolved,
        source_ids=user.source_ids,
        fixed_ids=fixed_ids,
        fixed_positions=tuple(references[i] for i in fixed_ids),
        user_policy=user.user_policy,
        user_fixed_ids=tuple(user.fixed_ids),
    )


def overlay_tolerance(config) -> float:
    """Return the displacement tolerance for PSR-mapped overlays on user-fixed atoms.

    The pipeline accepts a registration whose matching score is at most
    ``psr.matching_score_thr``; a user-fixed atom displaced by more than that
    describes a genuinely different event, anything below is a mapping residual
    that ``protect_positions`` re-clamps (contracts 7f policy 5). A config
    without a PSR section keeps the exact 1e-10 A check.
    """
    thr = getattr(getattr(config, "psr", None), "matching_score_thr", None)
    if thr is None:
        return 1e-10
    thr = float(thr)
    if not math.isfinite(thr) or thr < 0:
        raise ValueError("psr.matching_score_thr must be a finite non-negative length")
    return max(thr, 1e-10)


def validate_event_constraints(
    config,
    positions,
    types,
    cell,
    pbc,
    center_index,
    constraints=None,
    *,
    user_constraints=None,
    active_volume=None,
):
    """Validate a full-source execution payload before any native mutation.

    With no payload, the standalone call declares this full input its source.
    Supplied payloads retain their already-resolved user identities; the AV
    shell is independently checked against this operation's actual source.
    """
    center_index = _indices((center_index,), upper=len(positions))[0]
    active = config.control.active_volume if active_volume is None else active_volume
    if constraints is None:
        return resolve_event_constraints(
            config,
            positions,
            types,
            cell,
            pbc,
            center_index,
            user_constraints=user_constraints,
            active_volume=active,
        )
    if not isinstance(constraints, ResolvedConstraints):
        raise ValueError("constraints must be a resolved source payload")
    constraints.validate(len(positions))
    if set(constraints.atom_ids) != set(constraints.source_ids):
        raise ValueError("event execution requires the full source constraint mapping")
    constraints.validate_positions(positions, cell=cell, pbc=pbc)
    region = getattr(config, "frozen_atoms", None)
    policy = json.dumps(
        None if region is None else region.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if constraints.user_policy not in (None, policy):
        raise ValueError(
            "event constraint policy differs from the configured user policy"
        )
    if user_constraints is None:
        # A standalone call without initialized authority declares this full
        # input to be its user source, just as the stateless HTST boundary does.
        user_constraints = ResolvedConstraints.resolve(
            positions,
            types,
            region,
            constraints.atom_ids,
            cell=cell,
            pbc=pbc,
        )
        user_constraints = replace(user_constraints, source_ids=constraints.source_ids)
    else:
        user_constraints.validate(len(positions))
        if user_constraints.user_policy not in (None, policy):
            raise ValueError("user constraint policy changed since initialization")
        if set(user_constraints.atom_ids) != set(user_constraints.source_ids):
            raise ValueError("user authority must retain the full source")
        replace(user_constraints, atom_ids=constraints.atom_ids).validate_positions(
            positions,
            cell=cell,
            pbc=pbc,
        )
    constraints.require_preserves(user_constraints, cell=cell, pbc=pbc)
    if active:
        expected = ResolvedConstraints.resolve(
            positions,
            types,
            atom_ids=constraints.atom_ids,
            cell=cell,
            pbc=pbc,
            center_id=constraints.atom_ids[center_index],
            rmov=config.activevolume.rmov,
        )
        expected = replace(expected, source_ids=constraints.source_ids)
        if (
            constraints.center_id != expected.center_id
            or constraints.rmov != expected.rmov
        ):
            raise ValueError(
                "event constraints differ from the AV source center/radius"
            )
        from ase.geometry import find_mic

        if constraints.center_position is None:
            raise ValueError("event constraints lack the AV source center")
        _, center_distance = find_mic(
            np.asarray(constraints.center_position) - expected.center_position,
            cell,
            pbc=pbc,
        )
        if center_distance > 1e-10:
            raise ValueError("event constraints differ from the AV source center")
        constraints.require_preserves(expected, cell=cell, pbc=pbc)
    return constraints


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
