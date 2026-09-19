"""Immutable producing inputs and adapter correspondence without native launches."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from pykmc.engine.htst_lammps import LammpsHTSTExtension
from pykmc.htst import compute_event_prefactors

from .producing_helpers import (
    GEOMETRIES,
    IDS,
    TYPES,
    Scratch,
    assert_geometry,
    capture,
    matrix,
    produced_request,
    request_case,
    require_api,
)


def test_snapshot_copies_full_source_and_excludes_transport_key():
    api = require_api()
    request, _ = request_case()
    snapshot = api.RequestSnapshot.capture(request)
    before = snapshot.to_request(event_key=("before",))
    identity = snapshot.snapshot_id
    assert snapshot.is_complete
    assert (
        snapshot.snapshot_id
        == api.RequestSnapshot.capture(
            replace(request, event_key=("different-batch", 700))
        ).snapshot_id
    )
    assert_geometry(before, request)
    request.min1_positions[0, 0] += 3.0
    request.saddle_positions[3, 1] += 4.0
    request.min2_positions[2, 2] += 5.0
    request.cell[0, 0] += 10.0
    assert snapshot.snapshot_id == identity
    assert_geometry(snapshot.to_request(event_key=("after",)), before)
    detached = snapshot.to_request(event_key=("detached",))
    assert detached.event_key == ("detached",)
    detached.min1_positions[:] = 0.0
    detached.cell[:] = 0.0
    assert_geometry(snapshot.to_request(event_key=()), before)
    with pytest.raises((AttributeError, TypeError)):
        snapshot.center_index = 0


def test_source_produced_method_zone_energy_and_direction_are_distinct_content():
    api = require_api()
    source, _ = request_case(premin=True, zone=2.0)
    produced = produced_request(source)
    record = capture(api, source, produced, energies=(0.0, 1.0, 0.25))
    assert_geometry(record.source.to_request(event_key=()), source)
    assert_geometry(record.produced.to_request(event_key=()), produced)
    assert record.source.snapshot_id != record.produced.snapshot_id
    assert record.free_indices == (2,) and record.zone_indices == (1, 2, 3)
    assert record.method == "protocol-matrix"
    assert record.energies == (0.0, 1.0, 0.25)
    assert record.reusable
    forward = record.directional_identity("forward")
    backward = record.directional_identity("backward")
    assert forward.free_ids == backward.free_ids == (42,)
    assert forward.atom_ids == backward.atom_ids == IDS
    assert forward.center_id == backward.center_id == 42
    assert forward.descriptor_id == source.descriptor.descriptor_id
    assert forward.identity_id != backward.identity_id
    assert (
        forward.identity_id
        == produced.calculation_identity("forward", free_indices=(2,)).identity_id
    )
    other = [
        capture(
            api, source, produced, method="different-method", energies=record.energies
        ),
        capture(
            api, source, produced, zone_indices=(0, 1, 2, 3), energies=record.energies
        ),
        capture(api, source, produced, energies=(0.0, 2.0, 0.25)),
        capture(api, produced, produced, energies=record.energies),
    ]
    assert len({record.provenance_id, *(r.provenance_id for r in other)}) == 5
    old_id = record.provenance_id
    source.min1_positions[0, 0] += 1.0
    produced.min2_positions[3, 1] += 2.0
    assert record.provenance_id == old_id
    with pytest.raises((AttributeError, TypeError)):
        record.method = "changed"
    with pytest.raises(ValueError):
        record.directional_identity("sideways")


@pytest.mark.parametrize(
    "changes",
    [
        {"free_indices": (2, 2)},
        {"free_indices": (4,)},
        {"free_indices": (2.0,)},
        {"free_indices": (1, 2)},
        {"zone_indices": (1, 3)},
        {"energies": (0.0, float("nan"), 0.0)},
    ],
    ids=[
        "duplicate-free",
        "out-of-range",
        "noninteger",
        "fixed-in-free",
        "free-outside-zone",
        "nonfinite-energy",
    ],
)
def test_malformed_actual_sets_and_energy_are_rejected(changes):
    api = require_api()
    request, _ = request_case()
    with pytest.raises(ValueError):
        capture(api, request, **changes)


@pytest.mark.parametrize("change", ["center", "cell", "malformed-geometry"])
def test_premin_cannot_change_source_context_or_hide_malformed_geometry(change):
    api = require_api()
    request, _ = request_case(premin=True)
    if change == "center":
        produced = replace(request, center_index=3)
    elif change == "cell":
        produced = replace(request, cell=request.cell * 2.0)
    else:
        bad = request.min1_positions.copy()
        bad[0, 0] = float("nan")
        produced = replace(request, min1_positions=bad)
    with pytest.raises(ValueError):
        capture(api, request, produced)


def test_partial_and_unknown_contexts_remain_truthful_nonreusable_records():
    api = require_api()
    request, _ = request_case(zone=2.0)
    zone = (1, 2, 3)
    cropped = replace(
        request,
        **{key: getattr(request, key)[list(zone)].copy() for key in GEOMETRIES},
        types=tuple(request.types[i] for i in zone),
        constraints=request.constraints.crop(zone),
        center_index=1,
    )
    snapshot = api.RequestSnapshot.capture(cropped)
    assert not snapshot.is_complete
    assert snapshot.to_request(event_key=()).constraints.source_ids == IDS
    partial = api.CalculationProvenance.capture(
        cropped, cropped, method="local-crop", free_indices=(1,)
    )
    assert not partial.reusable
    assert partial.directional_identity("forward").free_ids == (42,)
    missing = replace(request, descriptor=None)
    assert not capture(api, missing).reusable
    opaque_model = replace(
        request.descriptor.engine.force_model, limitation="opaque protocol model"
    )
    opaque = replace(
        request,
        descriptor=replace(
            request.descriptor,
            engine=replace(request.descriptor.engine, force_model=opaque_model),
        ),
    )
    assert not capture(api, opaque).reusable


@pytest.mark.parametrize("mode", ["both", "forward-only", "bad-saddle", "empty"])
def test_real_kernel_emits_actual_provenance_and_directional_records(mode):
    require_api()
    request, _ = request_case()
    calls = []

    def hessian(positions, free):
        calls.append(tuple(free))
        return np.eye(3) if mode == "bad-saddle" else matrix(positions, free)

    result = compute_event_prefactors(
        request,
        hessian,
        method="protocol-matrix",
        free_indices=np.array([], dtype=int) if mode == "empty" else (2,),
        compute_backward=mode != "forward-only",
    )
    provenance = result.provenance
    assert provenance is not None
    assert_geometry(provenance.source.to_request(event_key=()), request)
    assert_geometry(provenance.produced.to_request(event_key=()), request)
    assert provenance.free_indices == (() if mode == "empty" else (2,))
    assert provenance.zone_indices == (0, 1, 2, 3)
    assert provenance.energies is None, "Hessian-only kernel cannot invent energies"
    assert result.calculation("forward").identity.free_ids == (
        () if mode == "empty" else (42,)
    )
    for direction in ("forward", "backward"):
        estimate = getattr(result, direction)
        record = result.calculation(direction)
        if estimate.skipped:
            assert record is None
            continue
        assert record.direction == direction
        assert record.provenance == provenance
        assert record.estimate == estimate
        assert record.descriptor_id == request.descriptor.descriptor_id
        assert record.reusable == estimate.ok
        with pytest.raises((AttributeError, TypeError)):
            record.direction = "changed"
    if mode == "both":
        assert result.forward.ok and result.backward.ok
        assert (
            result.calculation("forward").calculation_id
            != result.calculation("backward").calculation_id
        )
        relabelled = compute_event_prefactors(
            replace(request, event_key=("different", 999)),
            matrix,
            method="protocol-matrix",
            free_indices=(2,),
        )
        assert (
            result.calculation("forward").calculation_id
            == relabelled.calculation("forward").calculation_id
        )
        changed = replace(
            result, forward=replace(result.forward, nu0_hz=result.forward.nu0_hz * 2.0)
        )
        assert (
            changed.calculation("forward").calculation_id
            != result.calculation("forward").calculation_id
        )
        naked = replace(result, provenance=None)
        assert naked.calculation("forward") is None
    if mode == "empty":
        assert calls == []
    with pytest.raises(ValueError):
        result.calculation("sideways")


@pytest.mark.parametrize("premin", [False, True])
@pytest.mark.parametrize("zone", [None, 2.0], ids=["full", "crop"])
def test_native_adapter_replaces_local_record_with_full_producing_context(premin, zone):
    require_api()
    request, native = request_case(premin=premin, zone=zone)
    scratch = Scratch(native)
    extension = object.__new__(LammpsHTSTExtension)
    extension.engine = SimpleNamespace(config=native, comm=None, engine_id=0)
    extension._new_scratch = lambda: scratch

    def local_matrix(resource, positions, free, step):
        assert resource is scratch
        local_ni = scratch.types.index("Ni")
        assert tuple(free) == (local_ni,)
        x = positions[local_ni, 0]
        return np.diag([-1.0, 2.0, 3.0] if x == 3.0 else [4.0, 5.0, 6.0])

    extension._eskm_hessian = local_matrix
    result = extension.compute_event_prefactors(request)
    assert result.forward.ok and result.backward.ok
    assert scratch.closed
    assert scratch.builds[-1] == (("Cu", "Ni", "Fe") if zone else TYPES)
    record = result.provenance
    assert record is not None and record.reusable
    assert record.method == "lammps_eskm"
    assert record.source.is_complete and record.produced.is_complete
    assert record.free_indices == (2,)
    assert record.zone_indices == ((1, 2, 3) if zone else (0, 1, 2, 3))
    assert_geometry(record.source.to_request(event_key=()), request)
    expected = produced_request(request) if premin else request
    assert_geometry(record.produced.to_request(event_key=()), expected)
    # In crop+premin this outer atom is absent from every Hessian input, but
    # remains a dependency of the full premin calculation and restart source.
    assert record.produced.to_request(event_key=()).min1_positions[0, 1] == (
        0.1 if premin else 0.0
    )
    assert request.min1_positions[0, 1] == 0.0
    assert result.calculation("forward").identity.free_ids == (42,)
    assert result.calculation("backward").identity.atom_ids == IDS
    if record.energies is not None:
        assert record.energies == (0.0, 1.0, 0.25)
