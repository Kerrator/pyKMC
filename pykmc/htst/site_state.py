"""Full source dependencies and row bindings for transient active estimates."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .free_region import select_free_indices
from .provenance import RequestSnapshot
from .result import DirectionalCalculation
from ..physics import _indices
from ..utils.geometry import minimum_image_displacement

DEPENDENCY_POSITION_TOL: float = 1.0e-8
"""Largest displacement (Å) of a dependency-region atom that still counts as unmoved.

Far below any displacement a KMC event, a minimisation step or premin can
produce (1e-4 Å and above) and far above the last-bit noise of a double
precision coordinate round trip, so only representation noise is absorbed;
a physical move of a dependency atom always invalidates.
"""


def dependency_radius(settings: Any) -> float:
    """Return the radius (Å) of a site spectrum's geometric dependency.

    The Hessian is built over the free sphere (``free_radius`` about the
    centre). When the calculation was zone-cropped (``zone_radius``), the zone
    is the whole set of atoms the scratch calculation ever saw, so it bounds
    the dependency instead. Atoms beyond this radius (and outside the stored
    crop) cannot change the spectrum through the free-region Hessian; the
    recycler's movement/distance filters remain the guard for the executed
    event's own surroundings.
    """
    radius = float(settings.free_radius)
    if settings.zone_radius is not None:
        radius = max(radius, float(settings.zone_radius))
    return radius


def source_index_map(system: Any) -> dict[int, int]:
    """Map every stable source identity of ``system`` to its current row, once."""
    return {atom_id: row for row, atom_id in enumerate(_indices(system.index))}


def row_signature(row: Any, center_id: int) -> tuple:
    """Bind a calculation to an event independently of its DataFrame label."""
    ids = _indices(row["crop_atom_ids"])
    order = np.argsort(ids)

    def coordinates(name):
        value = row[name]
        if value is None:
            return None
        array = np.asarray(value, dtype=float)
        if array.shape != (len(ids), 3) or not np.isfinite(array).all():
            raise ValueError("active event has invalid crop correspondence")
        return tuple(tuple(float(x) for x in point) for point in array[order])

    return (
        int(center_id),
        int(row["num_reference_event"]),
        tuple(sorted(ids)),
        coordinates("saddle_positions"),
        coordinates("final_positions"),
        float(row["energy_barrier"]),
        str(row["refined"]),
        str(row["nu0_status"]),
        str(row["nu0_source"]),
        # A canonical string also represents the intentionally absent NaN value.
        str(float(row["nu0"])),
        bool(row["nu0_site_attempted"]),
    )


@dataclass(frozen=True)
class SiteState:
    """A submitted source or explicit fallback context, never invented results."""

    source: RequestSnapshot
    method: str
    signature: tuple
    calculation: DirectionalCalculation | None = None
    fresh_service: Any = field(default=None, compare=False, repr=False)
    dependency_rows: tuple[int, ...] | None = field(
        default=None, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        """Resolve the dependency rows once; ``replace`` carries them over."""
        if self.dependency_rows is None:
            object.__setattr__(self, "dependency_rows", self._resolve_dependency())

    @property
    def atom_ids(self) -> tuple[int, ...]:
        return self.source.constraints.atom_ids

    @property
    def center_id(self) -> int:
        return self.atom_ids[self.source.center_index]

    @property
    def dependency_ids(self) -> tuple[int, ...]:
        """Stable identities of the atoms whose positions enter the spectrum."""
        return tuple(self.atom_ids[k] for k in self.dependency_rows)

    def _resolve_dependency(self) -> tuple[int, ...]:
        """Snapshot rows of the stored crop and the free/zone sphere.

        The sphere is taken in both producing geometries so the selection is
        covered whichever ``free_region_center`` produced it.
        """
        source = self.source
        radius = dependency_radius(source.settings)
        cell = np.asarray(source.cell, dtype=float)
        rows: set[int] = set()
        for geometry in (source.min1_positions, source.saddle_positions):
            rows.update(
                int(i)
                for i in select_free_indices(
                    np.asarray(geometry, dtype=float),
                    source.center_index,
                    radius,
                    cell,
                    source.pbc,
                )
            )
        position = {atom_id: k for k, atom_id in enumerate(self.atom_ids)}
        for atom_id in self.signature[2]:
            row = position.get(int(atom_id))
            if row is None:
                raise ValueError("active event crop identities are not in its source")
            rows.add(row)
        return tuple(sorted(rows))

    def dependency_unchanged(self, system: Any, index_map: dict[int, int]) -> bool:
        """Whether the dependency region of ``system`` equals the producing one.

        Every dependency atom must still exist with the same type and sit
        within :data:`DEPENDENCY_POSITION_TOL` (minimum image) of its producing
        position, and no other atom may have entered the free/zone sphere
        about the centre's current position.
        """
        rows = list(self.dependency_rows)
        current_rows = []
        for row in rows:
            current = index_map.get(self.atom_ids[row])
            if current is None:
                return False
            current_rows.append(current)
        positions = np.asarray(system.positions, dtype=float)
        center_row = index_map[self.center_id]
        entered = select_free_indices(
            positions,
            center_row,
            dependency_radius(self.source.settings),
            system.cell,
            system.pbc,
        )
        region = set(self.atom_ids[row] for row in rows)
        index = system.index
        if any(int(index[k]) not in region for k in entered):
            return False
        snapshot = np.asarray(self.source.min1_positions, dtype=float)[rows]
        delta = minimum_image_displacement(
            positions[current_rows] - snapshot,
            np.asarray(self.source.cell, dtype=float),
            np.asarray(self.source.pbc, dtype=bool),
        )
        if delta.size and float(np.max(np.linalg.norm(delta, axis=1))) > (
            DEPENDENCY_POSITION_TOL
        ):
            return False
        return tuple(str(system.types[k]) for k in current_rows) == tuple(
            self.source.types[k] for k in rows
        )

    def matches(
        self,
        row: Any,
        system: Any,
        service: Any,
        *,
        index_map: dict[int, int] | None = None,
    ) -> bool:
        """Compare the current authority and the row's geometric dependency.

        Only the atoms whose positions enter the site spectrum are compared
        (:meth:`dependency_unchanged`): motion elsewhere in the source leaves a
        recycled row valid, an atom entering the region or a moved dependency
        atom invalidates it. The producing authority (method, descriptor,
        species and masses, settings, user constraints) must be compatible;
        for a site value the bound calculation must still be the actual
        producer of the row's frequency. Rate temperature and the acceptance
        window are applied separately by the table and do not require another
        Hessian.

        Raises
        ------
        HTSTRequestError
            From ``service.descriptor_for`` when the source types are not in
            the current engine species map; the caller reports it.

        """
        if service is None or row_signature(row, self.center_id) != self.signature:
            return False
        if index_map is None:
            index_map = source_index_map(system)
        if len(index_map) != len(self.atom_ids) or self.center_id not in index_map:
            return False
        if (
            not np.array_equal(system.cell, self.source.cell)
            or tuple(bool(p) for p in system.pbc) != self.source.pbc
            or service.method != self.method
        ):
            return False
        if not self.dependency_unchanged(system, index_map):
            return False
        descriptor = service.descriptor_for(self.source.types)
        same_producing_context = (
            self.fresh_service is service
            and descriptor.descriptor_id == self.source.descriptor.descriptor_id
        )
        if (
            descriptor.compare(self.source.descriptor).status != "compatible"
            and not same_producing_context
        ):
            return False
        if (
            descriptor.engine.species != self.source.species
            or descriptor.engine.masses != self.source.masses
        ):
            return False
        settings = replace(
            service.settings,
            nu0_min_hz=self.source.settings.nu0_min_hz,
            nu0_max_hz=self.source.settings.nu0_max_hz,
        )
        if settings != self.source.settings:
            return False
        if (
            row["nu0_status"] != "ok"
            or (
                self.calculation is not None
                and not self.calculation.estimate.ok
                and self.calculation.estimate.reason_code.value == "out_of_window"
            )
        ) and (
            service.settings.nu0_min_hz != self.source.settings.nu0_min_hz
            or service.settings.nu0_max_hz != self.source.settings.nu0_max_hz
        ):
            return False
        # Initialized user authority is the only fixed-coordinate contract of
        # a request (contracts 7f policy 5); a changed one ends the binding.
        user = service.global_constraints
        if user is not None and user != self.source.user_constraints:
            return False
        if row["nu0_source"] == "site":
            calc = self.calculation
            if calc is None:
                return False
            calc.validate()
            if (
                calc.direction != "forward"
                or not calc.estimate.ok
                or calc.provenance.source != self.source
                or calc.provenance.method != service.method
                or float(row["nu0"]) != calc.estimate.nu0_hz
            ):
                return False
        return True
