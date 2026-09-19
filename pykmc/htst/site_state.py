"""Full source dependencies and row bindings for transient active estimates."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .provenance import RequestSnapshot
from .result import DirectionalCalculation
from ..physics import _indices, resolve_event_constraints


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

    @property
    def atom_ids(self) -> tuple[int, ...]:
        return self.source.constraints.atom_ids

    @property
    def center_id(self) -> int:
        return self.atom_ids[self.source.center_index]

    def matches(self, row, system, service) -> bool:
        """Compare current initialized authority and the complete current source.

        Source ordering can change; physical changes cannot be justified by an
        unmoved center. Coordinate representation changes conservatively trigger
        rebuilding too. Rate temperature and the acceptance window are applied
        separately by the table and do not require another Hessian.
        """
        if service is None or row_signature(row, self.center_id) != self.signature:
            return False
        ids = _indices(system.index)
        if set(ids) != set(self.atom_ids):
            return False
        order = [ids.index(i) for i in self.atom_ids]
        positions = np.asarray(system.positions)[order]
        types = tuple(str(system.types[i]) for i in order)
        if (
            types != self.source.types
            or not np.array_equal(positions, self.source.min1_positions)
            or not np.array_equal(system.cell, self.source.cell)
            or tuple(bool(p) for p in system.pbc) != self.source.pbc
            or service.method != self.method
        ):
            return False
        descriptor = service.descriptor_for(types)
        if descriptor.compare(self.source.descriptor).status != "compatible":
            return False
        settings = replace(
            service.settings,
            nu0_min_hz=self.source.settings.nu0_min_hz,
            nu0_max_hz=self.source.settings.nu0_max_hz,
        )
        if settings != self.source.settings:
            return False
        if row["nu0_status"] != "ok" and (
            service.settings.nu0_min_hz != self.source.settings.nu0_min_hz
            or service.settings.nu0_max_hz != self.source.settings.nu0_max_hz
        ):
            return False
        user = service.global_constraints
        if user is None:
            user = self.source.user_constraints
        constraints = resolve_event_constraints(
            service.config,
            positions,
            types,
            system.cell,
            system.pbc,
            self.source.center_index,
            self.atom_ids,
            user_constraints=user,
        )
        request = service.build_request(
            event_key=(),
            min1_positions=positions,
            saddle_positions=np.asarray(self.source.saddle_positions),
            min2_positions=np.asarray(self.source.min2_positions),
            types=types,
            cell=system.cell,
            pbc=system.pbc,
            center_index=self.source.center_index,
            constraints=constraints,
        )
        current = RequestSnapshot.capture(request)
        if replace(current, settings=self.source.settings) != self.source:
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
                or float(row["nu0"]) != calc.estimate.nu0_hz
                or not service.calculation_context_matches(calc, request)
            ):
                return False
        return True
