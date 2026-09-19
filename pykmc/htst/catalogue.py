"""Content-addressed producing records for reference-catalogue estimates.

The local event crop is bound to a calculation explicitly by its admitting
caller. It is not a substitute for that calculation's full source snapshot.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from .result import DirectionalCalculation
from ..physics import PhysicalDescriptor, _digest


def row_digest(row: Any) -> str:
    """Bind persisted event geometry and barrier, independent of table labels."""
    return _digest(
        (
            *(
                np.asarray(row[name], dtype=float).tolist()
                for name in ("initial_positions", "saddle_positions", "final_positions")
            ),
            None if row.get("types") is None else list(row["types"]),
            int(row["move_atom_idx"]),
            float(row["energy_barrier"]),
        )
    )


@dataclass(frozen=True)
class EstimateReference:
    """An explicit logical-row association, separate from physical identity."""

    calculation_id: str
    row_digest: str


class PrefactorArchive:
    """Append-only producing facts and superseded/unknown estimate history."""

    def __init__(self) -> None:
        self.descriptors: dict[str, PhysicalDescriptor] = {}
        self.calculations: dict[str, DirectionalCalculation] = {}
        self.references: dict[int, EstimateReference | None] = {}
        self.history: dict[int, list[dict[str, Any]]] = {}
        self.legacy_metadata: list[dict[str, Any]] = []

    def record(
        self, idx_ref: int, row: Any, calculation: DirectionalCalculation
    ) -> None:
        """Associate an actual calculation; never infer one from a service."""
        calculation.validate()
        descriptor = calculation.provenance.produced.descriptor
        if descriptor is not None:
            self.descriptors.setdefault(descriptor.descriptor_id, descriptor)
        key = calculation.calculation_id
        self.calculations.setdefault(key, calculation)
        self.references[int(idx_ref)] = EstimateReference(key, row_digest(row))

    def retain(self, idx_ref: int, row: Any, reason: str) -> None:
        """Keep unavailable original values outside the selectable columns."""

        def number(name):
            value = row.get(name)
            return float(value) if value is not None and np.isfinite(value) else None

        entry = {
            "reference": self.references.get(int(idx_ref)),
            "nu0": number("nu0"),
            "nu0_status": str(row.get("nu0_status", "legacy")),
            "nu0_reason": str(row.get("nu0_reason", "")),
            "energy_barrier": number("energy_barrier"),
            "k_prefactor": number("k_prefactor"),
            "reason": reason,
        }
        history = self.history.setdefault(int(idx_ref), [])
        if entry not in history:
            history.append(entry)

    def calculation_for(self, idx_ref: int, row: Any) -> DirectionalCalculation | None:
        link = self.references.get(int(idx_ref))
        if link is None:
            return None
        if link.row_digest != row_digest(row):
            return None
        return self.calculations.get(link.calculation_id)

    def metadata(self) -> dict[str, Any]:
        """Return detached serialization state without any current-service claim."""
        return deepcopy(
            {
                "descriptors": self.descriptors,
                "calculations": self.calculations,
                "estimate_references": self.references,
                "estimate_history": self.history,
                "legacy_metadata": self.legacy_metadata,
            }
        )

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> PrefactorArchive:
        """Validate declared registry content; absence remains unknown."""
        archive = cls()
        for field, attribute in (
            ("descriptors", "descriptors"),
            ("calculations", "calculations"),
            ("estimate_references", "references"),
            ("estimate_history", "history"),
        ):
            value = metadata.get(field)
            if value is None:
                value = {}
            if not isinstance(value, dict):
                raise ValueError(f"reference table {field} must be a dictionary")
            setattr(archive, attribute, deepcopy(value))
        archive.legacy_metadata = deepcopy(metadata.get("legacy_metadata", []))
        for key, descriptor in archive.descriptors.items():
            if descriptor is None:
                continue
            if not isinstance(descriptor, PhysicalDescriptor):
                raise ValueError("unsupported producing descriptor registry value")
            descriptor.__post_init__()
            descriptor.engine.__post_init__()
            descriptor.engine.force_model.__post_init__()
            if key != descriptor.descriptor_id:
                raise ValueError("producing descriptor registry ID mismatch")
        for key, calculation in archive.calculations.items():
            if calculation is None:
                continue
            if not isinstance(calculation, DirectionalCalculation):
                raise ValueError("unsupported producing calculation registry value")
            calculation.validate()
            if key != calculation.calculation_id:
                raise ValueError("producing calculation registry ID mismatch")
        for key, link in archive.references.items():
            if type(key) is not int or key < 0:
                raise ValueError("producing references require nonnegative logical IDs")
            if link is not None and (
                not isinstance(link, EstimateReference)
                or not isinstance(link.calculation_id, str)
                or not isinstance(link.row_digest, str)
            ):
                raise ValueError("unsupported producing estimate reference")
        return archive
