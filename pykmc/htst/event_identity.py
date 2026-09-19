"""Conservative whole-event identity under one physical rigid map.

IRA proposes a witness; every vertex and the full producing environment must
then satisfy that *same* witness. Failure to find one is not an inequivalence
proof. Callers retain both channels whenever these sufficient checks fail.
"""

from __future__ import annotations

import numpy as np

from .free_region import common_free_indices, select_free_indices
from .provenance import RequestSnapshot
from ..point_set_registration import simple_ira
from ..utils.geometry import minimum_image_displacement

STRUCTURAL_ATOL = 1e-10
"""Rounding tolerance for orthogonality/lattice relations and fixed references."""


def _vertices(snapshot, direction):
    vertices = (
        np.asarray(snapshot.min1_positions),
        np.asarray(snapshot.saddle_positions),
        np.asarray(snapshot.min2_positions),
    )
    if direction == "forward":
        return vertices
    if direction == "backward":
        return vertices[::-1]
    raise ValueError("event direction must be forward or backward")


def _lattice_preserved(rotation, cell, pbc):
    """Require an onto automorphism of the actual periodic lattice."""
    basis = np.asarray(cell)[np.asarray(pbc, dtype=bool)]
    if not len(basis):
        return True
    transformed = basis @ rotation.T
    coefficients = transformed @ np.linalg.pinv(basis)
    integers = np.rint(coefficients)
    return (
        np.allclose(coefficients, integers, rtol=0, atol=STRUCTURAL_ATOL)
        and np.allclose(transformed, integers @ basis, rtol=0, atol=STRUCTURAL_ATOL)
        and abs(round(np.linalg.det(integers))) == 1
    )


def _near(mapped, target, cell, pbc, tolerance):
    residual = minimum_image_displacement(mapped - target, cell, pbc)
    return bool(np.all(np.linalg.norm(residual, axis=-1) <= tolerance))


def _selection_preserved(source, target, permutation, count):
    source_mask = np.zeros(count, dtype=bool)
    target_mask = np.zeros(count, dtype=bool)
    source_mask[list(source)] = True
    target_mask[list(target)] = True
    return np.array_equal(source_mask[permutation], target_mask)


def _restriction_preserved(
    source, target, permutation, rotation, translation, cell, pbc
):
    """Compare physical masks/references, translating global IDs into rows."""
    count = len(permutation)
    source_fixed = () if source is None else source.local_fixed_indices
    target_fixed = () if target is None else target.local_fixed_indices
    if not _selection_preserved(source_fixed, target_fixed, permutation, count):
        return False
    if target_fixed:
        source_references = dict(zip(source.fixed_ids, source.fixed_positions))
        target_references = dict(zip(target.fixed_ids, target.fixed_positions))
        old = np.array(
            [source_references[source.atom_ids[permutation[i]]] for i in target_fixed]
        )
        new = np.array([target_references[target.atom_ids[i]] for i in target_fixed])
        if not _near(old @ rotation.T + translation, new, cell, pbc, STRUCTURAL_ATOL):
            return False
    source_radius = None if source is None else source.rmov
    target_radius = None if target is None else target.rmov
    if source_radius != target_radius:
        return False
    if source_radius is not None:
        source_center = source.atom_ids.index(source.center_id)
        target_center = target.atom_ids.index(target.center_id)
        if permutation[target_center] != source_center or not _near(
            np.asarray(source.center_position) @ rotation.T + translation,
            np.asarray(target.center_position),
            cell,
            pbc,
            STRUCTURAL_ATOL,
        ):
            return False
    return True


def _context_preserved(source, target, permutation, rotation, translation):
    source_types, target_types = np.asarray(source.types), np.asarray(target.types)
    if not np.array_equal(source_types[permutation], target_types):
        return False
    if (
        not np.array_equal(
            source.to_request().masses_per_atom()[permutation],
            target.to_request().masses_per_atom(),
        )
        or permutation[target.center_index] != source.center_index
    ):
        return False
    for name in ("constraints", "user_constraints"):
        if not _restriction_preserved(
            getattr(source, name),
            getattr(target, name),
            permutation,
            rotation,
            translation,
            target.cell,
            target.pbc,
        ):
            return False
    return True


def _common_map(
    sources,
    targets,
    source_direction,
    target_direction,
    source_free,
    target_free,
    source_zone,
    target_zone,
    *,
    tolerance,
    kmax_factor,
):
    source, target = sources[0], targets[0]
    count = len(source.types)
    for old, new in zip(sources, targets, strict=True):
        old.validate()
        new.validate()
        if (
            len(old.types) != count
            or len(new.types) != count
            or not old.is_complete
            or not new.is_complete
            or old.cell != new.cell
            or old.pbc != new.pbc
            or old.descriptor is None
            or new.descriptor is None
            or old.descriptor.compare(new.descriptor).status != "compatible"
        ):
            return False

    def proves(rotation, translation, permutation):
        rotation, translation, permutation = map(
            np.asarray, (rotation, translation, permutation)
        )
        if (
            rotation.shape != (3, 3)
            or translation.shape != (3,)
            or permutation.shape != (count,)
            or permutation.dtype.kind not in "iu"
            or not np.array_equal(np.sort(permutation), np.arange(count))
            or not np.isfinite(rotation).all()
            or not np.isfinite(translation).all()
            or not np.allclose(
                rotation.T @ rotation, np.eye(3), rtol=0, atol=STRUCTURAL_ATOL
            )
            or not _lattice_preserved(rotation, target.cell, target.pbc)
        ):
            return False
        # Fingerprinted supported pair styles are invariant under proper and
        # improper orthogonal maps. Opaque force definitions failed above.
        if not all(
            _selection_preserved(a, b, permutation, count)
            for a, b in (
                (source_free, target_free),
                (source_zone, target_zone),
            )
        ):
            return False
        for old, new in zip(sources, targets, strict=True):
            if not _context_preserved(old, new, permutation, rotation, translation):
                return False
            for before, after in zip(
                _vertices(old, source_direction),
                _vertices(new, target_direction),
                strict=True,
            ):
                mapped = before[permutation] @ rotation.T + translation
                if not _near(mapped, after, new.cell, new.pbc, tolerance):
                    return False
        return True

    initial_source = _vertices(source, source_direction)[0]
    initial_target = _vertices(target, target_direction)[0]
    translation = (
        initial_target[target.center_index] - initial_source[source.center_index]
    )
    if proves(np.eye(3), translation, np.arange(count)):
        return True

    # Canonicalize copies only to propose a map. All checks above use the
    # actual full coordinates, one translation, and the actual periodic axes.
    def unwrapped(positions, snapshot):
        center = positions[snapshot.center_index]
        return center + minimum_image_displacement(
            positions - center, snapshot.cell, snapshot.pbc
        )

    result = simple_ira(
        count,
        list(target.types),
        unwrapped(initial_target, target),
        count,
        list(source.types),
        unwrapped(initial_source, source),
        kmax_factor,
    )
    if not result.is_ok():
        return False
    witness = result.ok_value()
    return proves(
        witness.rotation_matrix, witness.translation_matrix, witness.permutation_matrix
    )


def calculations_equivalent(first, second, *, tolerance, kmax_factor):
    """Compare two actual directional producers with one common full map."""
    first.validate()
    second.validate()
    a, b = first.provenance, second.provenance
    if a.method != b.method:
        return False
    return _common_map(
        (a.source, a.produced),
        (b.source, b.produced),
        first.direction,
        second.direction,
        a.free_indices,
        b.free_indices,
        a.zone_indices,
        b.zone_indices,
        tolerance=tolerance,
        kmax_factor=kmax_factor,
    )


def request_matches_calculation(
    request, direction, calculation, *, method, tolerance, kmax_factor
):
    """Conservative pre-dispatch duplicate proof, without inventing a producer.

    The incoming source must match both the stored source and the actual
    prepared geometry. Preprocessing differences can therefore cause a safe
    non-match; post-dispatch comparison uses each actual producing snapshot.
    """
    request.validate()
    calculation.validate()
    provenance = calculation.provenance
    if provenance.method != method:
        return False
    snapshot = RequestSnapshot.capture(request)
    free = tuple(int(i) for i in common_free_indices(request))
    center_positions = (
        request.saddle_positions
        if request.settings.free_region_center == "saddle"
        else request.min1_positions
    )
    zone = (
        tuple(range(len(request.types)))
        if request.settings.zone_radius is None or method != "lammps_eskm"
        else tuple(
            int(i)
            for i in select_free_indices(
                center_positions,
                request.center_index,
                request.settings.zone_radius,
                request.cell,
                request.pbc,
            )
        )
    )
    return _common_map(
        (snapshot, snapshot),
        (provenance.source, provenance.produced),
        direction,
        calculation.direction,
        free,
        provenance.free_indices,
        zone,
        provenance.zone_indices,
        tolerance=tolerance,
        kmax_factor=kmax_factor,
    )
