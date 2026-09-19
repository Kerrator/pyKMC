"""N04: an explicit constraint payload cannot weaken known user constraints.

No native launch. The Hessian is the literal independent polynomial from the
earlier protocol oracle; fixed global ID 0 is local row 1 before permutation.
Service pair commands are descriptor metadata, not its analytic force model.
"""

from copy import deepcopy
from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest

from pykmc.config import RateConstantConfig, RegionConfig
from pykmc.htst import HTSTRequestError, compute_event_prefactors
from pykmc.physics import ResolvedConstraints
from pykmc.rate_constant import create_rate_constant
from pykmc.rate_constant.prefactors import PrefactorService


IDS = (42, 0, 91)
CELL = np.diag([10.0, 12.0, 14.0])
PBC = (True, False, True)
MASS = 58.6934
EXPECTED_HZ = math.sqrt(8.0 * 1.602176634e-19 / (MASS * 1.66053906660e-27 * 1e-20)) / (
    2.0 * math.pi
)
FIELDS = ("min1_positions", "saddle_positions", "min2_positions")


def triplet():
    source = np.array([[2.0, 3.0, 4.0], [3.0, 5.0, 4.0], [3.0, 3.0, 6.0]])
    saddle, final = source.copy(), source.copy()
    saddle[0, 0] = 3.0
    final[0, 0] = 4.0
    return source, saddle, final


def resolved(source, *, fixed=True, active=False):
    return ResolvedConstraints.resolve(
        source,
        ("Ni",) * 3,
        RegionConfig(indices=[1]) if fixed else None,
        IDS,
        cell=CELL,
        pbc=PBC,
        **({"center_id": 42, "rmov": 1.5} if active else {}),
    )


def service(*, snapshot=True, fixed=True):
    config = SimpleNamespace(
        lammps=SimpleNamespace(
            pair_style="zero 10.0",
            pair_coeff="* *",
            min_style="cg",
            frz_min="0.0 1e-12 100 1000",
        ),
        frozen_atoms=RegionConfig(indices=[1]) if fixed else None,
        rateconstant=RateConstantConfig(
            style="htst", free_radius=4.0, nu0_min_THz=0.01
        ),
    )
    global_snapshot = resolved(triplet()[0], fixed=fixed) if snapshot else None
    result = PrefactorService(
        config,
        SimpleNamespace(),
        create_rate_constant(config.rateconstant),
        species_masses=(("Ni",), (MASS,)),
        global_constraints=global_snapshot,
    )
    return result


def build(current, geometries, constraints):
    center = constraints.atom_ids.index(42) if constraints is not None else 0
    return current.build_request(
        event_key=("authority",),
        **dict(zip(FIELDS, geometries, strict=True)),
        types=("Ni",) * 3,
        cell=CELL.copy(),
        pbc=PBC,
        center_index=center,
        constraints=constraints,
    )


class Hessian:
    def __init__(self, ids):
        self.ids = tuple(ids)
        self.calls = []

    def __call__(self, positions, free):
        self.calls.append(tuple(self.ids[int(i)] for i in free))
        q = positions[self.ids.index(42), 0] - 3.0
        values = {
            42: (12.0 * q * q - 4.0, 4.0, 4.0),
            0: (4.0 + 12.0 * q * q,) * 3,
            91: (9.0, 16.0, 25.0),
        }
        return np.diag([v / MASS for i in free for v in values[self.ids[int(i)]]])


def check_result(request, expected_ids, expected_frequency):
    ids = request.constraints.atom_ids if request.constraints is not None else IDS
    oracle = Hessian(ids)
    result = compute_event_prefactors(request, oracle)
    assert len(oracle.calls) == 3
    assert all(set(selected) == set(expected_ids) for selected in oracle.calls)
    assert result.n_free == len(expected_ids)
    for direction in (result.forward, result.backward):
        assert direction.ok, direction.reason
        assert direction.n_positive_min == 3 * len(expected_ids)
        assert direction.n_negative_saddle == 1
        assert direction.nu0_hz == pytest.approx(expected_frequency, rel=1e-9, abs=0)


def preserve_inputs(current, geometries, payload):
    return (
        tuple(p.copy() for p in geometries),
        deepcopy(payload),
        deepcopy(current.global_constraints),
        current.current_descriptor,
    )


def assert_preserved(current, geometries, payload, before):
    for actual, expected in zip(geometries, before[0], strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert payload == before[1]
    assert current.global_constraints == before[2]
    assert current.current_descriptor == before[3]


@pytest.mark.parametrize(
    "snapshot", [True, False], ids=("known-snapshot", "policy-only")
)
def test_explicit_empty_payload_cannot_drop_required_user_fixed_id(snapshot):
    current = service(snapshot=snapshot)
    geometries = triplet()
    empty = resolved(geometries[0], fixed=False)
    assert empty.atom_ids == IDS and empty.fixed_ids == ()
    before = preserve_inputs(current, geometries, empty)
    with pytest.raises(HTSTRequestError):
        build(current, geometries, empty)
    assert_preserved(current, geometries, empty, before)


@pytest.mark.parametrize(
    "dy", [0.25, 12.0], ids=("physical-shift", "open-axis-cell-shift")
)
def test_rebased_triplet_cannot_replace_authoritative_fixed_reference(dy):
    current = service(snapshot=True)
    geometries = tuple(p + [0.0, dy, 0.0] for p in triplet())
    rebased = resolved(geometries[0])
    assert rebased.fixed_ids == current.global_constraints.fixed_ids == (0,)
    assert rebased.fixed_positions != current.global_constraints.fixed_positions
    # The entire new triplet agrees with the invented references. Only comparison
    # with the service's original authoritative snapshot can detect the bypass.
    for p in geometries:
        rebased.validate_positions(p)
    before = preserve_inputs(current, geometries, rebased)
    with pytest.raises(HTSTRequestError):
        build(current, geometries, rebased)
    assert_preserved(current, geometries, rebased, before)


def test_periodic_equivalent_explicit_reference_is_not_rejected_as_rebased():
    current = service(snapshot=True)
    geometries = triplet()
    for p in geometries:
        p[1] += [10.0, 0.0, -14.0]
    payload = resolved(geometries[0])
    assert payload.fixed_positions != current.global_constraints.fixed_positions
    before = preserve_inputs(current, geometries, payload)
    request = build(current, geometries, payload)
    check_result(request, (42, 91), EXPECTED_HZ)
    assert_preserved(current, geometries, payload, before)


@pytest.mark.parametrize(
    "snapshot", [True, False], ids=("known-snapshot", "policy-only")
)
def test_explicit_av_union_and_permuted_rows_preserve_user_source_identity(snapshot):
    current = service(snapshot=snapshot)
    original = triplet()
    payload = resolved(original[0], active=True)
    assert payload.fixed_ids == (0, 91)
    order = np.array([2, 0, 1])
    geometries = tuple(p[order].copy() for p in original)
    payload = payload.crop(order)
    # source_ids retains the original policy ordering, atom_ids maps local rows.
    assert payload.source_ids == IDS
    assert payload.atom_ids == (91, 42, 0)
    assert payload.local_fixed_indices == (0, 2)
    before = preserve_inputs(current, geometries, payload)
    request = build(current, geometries, payload)
    assert request.constraints.atom_ids == (91, 42, 0)
    assert set(request.constraints.fixed_ids) == {0, 91}
    check_result(request, (42,), EXPECTED_HZ)
    assert_preserved(current, geometries, payload, before)


@pytest.mark.parametrize(
    "snapshot", [True, False], ids=("empty-snapshot", "no-snapshot")
)
def test_unconstrained_service_retains_valid_explicit_empty_payload(snapshot):
    current = service(snapshot=snapshot, fixed=False)
    geometries = triplet()
    payload = resolved(geometries[0], fixed=False)
    before = preserve_inputs(current, geometries, payload)
    request = build(current, geometries, payload)
    check_result(request, IDS, 8.0 * EXPECTED_HZ)
    assert_preserved(current, geometries, payload, before)


@pytest.mark.parametrize("payload", ["none", "empty"])
def test_direct_constrained_descriptor_cannot_produce_unconstrained_estimate(payload):
    current = service(snapshot=True)
    geometries = triplet()
    valid = build(current, geometries, current.global_constraints)
    request = replace(
        valid,
        constraints=None if payload == "none" else resolved(geometries[0], fixed=False),
    )
    before = tuple(getattr(request, name).copy() for name in FIELDS)
    before_constraints = deepcopy(request.constraints)
    oracle = Hessian(IDS)
    try:
        result = compute_event_prefactors(request, oracle)
    except HTSTRequestError:
        # Public API may require a resolved payload, or derive it from the full
        # descriptor policy. Both are safe; an unconstrained result is not.
        assert oracle.calls == []
    else:
        assert oracle.calls == [(42, 91)] * 3
        assert result.n_free == 2
        for direction in (result.forward, result.backward):
            assert direction.ok, direction.reason
            assert direction.nu0_hz == pytest.approx(EXPECTED_HZ, rel=1e-9, abs=0)
    for name, expected in zip(FIELDS, before, strict=True):
        np.testing.assert_array_equal(getattr(request, name), expected)
    assert request.constraints == before_constraints


def test_legacy_descriptorless_request_remains_unconstrained():
    current = service(snapshot=False, fixed=False)
    request = build(current, triplet(), resolved(triplet()[0], fixed=False))
    request = replace(request, descriptor=None, constraints=None)
    check_result(request, IDS, 8.0 * EXPECTED_HZ)
