"""Seven no-launch N04 checks for resolved user-policy provenance and row maps.

Expected masks are literal. The analytic callback gives the moving atom minimum
curvature (8,4,4), saddle (-4,4,4), and constant spectator factors; no production
helper supplies the expected frequency. No native engine is constructed.
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
EXACT_HZ = math.sqrt(8.0 * 1.602176634e-19 / (MASS * 1.66053906660e-27 * 1e-20)) / (
    2.0 * math.pi
)
FIELDS = ("min1_positions", "saddle_positions", "min2_positions")


def triplet():
    initial = np.array([[2.0, 3.0, 4.0], [3.0, 5.0, 4.0], [3.0, 3.0, 6.0]])
    saddle, final = initial.copy(), initial.copy()
    saddle[0, 0], final[0, 0] = 3.0, 4.0
    return initial, saddle, final


def resolve(policy):
    return ResolvedConstraints.resolve(
        triplet()[0], ("Ni",) * 3, policy, IDS, cell=CELL, pbc=PBC
    )


def geometric_policy():
    return RegionConfig(region_type="plane", normal="y", threshold=4.0, side="above")


def make_service(policy, authority):
    config = SimpleNamespace(
        lammps=SimpleNamespace(
            pair_style="zero 10.0",
            pair_coeff="* *",
            min_style="cg",
            frz_min="0.0 1e-12 100 1000",
        ),
        frozen_atoms=policy,
        rateconstant=RateConstantConfig(
            style="htst", free_radius=4.0, nu0_min_THz=0.01
        ),
    )
    return PrefactorService(
        config,
        SimpleNamespace(),
        create_rate_constant(config.rateconstant),
        species_masses=(("Ni",), (MASS,)),
        global_constraints=authority,
    )


def build(service, geometries, *, constraints=None, atom_ids=None, center=0):
    return service.build_request(
        event_key=("policy-provenance",),
        **dict(zip(FIELDS, geometries, strict=True)),
        types=("Ni",) * len(geometries[0]),
        cell=CELL.copy(),
        pbc=PBC,
        center_index=center,
        constraints=constraints,
        atom_ids=atom_ids,
    )


def check_result(request, ids, expected_free):
    calls = []

    def hessian(positions, free):
        calls.append(tuple(ids[int(i)] for i in free))
        q = positions[ids.index(42), 0] - 3.0
        diagonal = {
            42: (12.0 * q * q - 4.0, 4.0, 4.0),
            0: (4.0 + 12.0 * q * q,) * 3,
            91: (9.0, 16.0, 25.0),
        }
        return np.diag([v / MASS for i in free for v in diagonal[ids[int(i)]]])

    result = compute_event_prefactors(request, hessian)
    assert calls == [expected_free] * 3
    assert result.n_free == len(expected_free)
    for direction in (result.forward, result.backward):
        assert direction.ok, direction.reason
        assert direction.nu0_hz == pytest.approx(EXACT_HZ, rel=1e-9, abs=0.0)


@pytest.mark.parametrize("route", ["service", "direct"])
@pytest.mark.parametrize("wrong", ["no-policy", "same-mask-different-policy"])
def test_resolved_authority_must_belong_to_the_captured_user_policy(route, wrong):
    expected_policy = RegionConfig(indices=[1])
    wrong_authority = resolve(None if wrong == "no-policy" else geometric_policy())
    assert wrong_authority.fixed_ids == (() if wrong == "no-policy" else (0,))
    before = deepcopy(wrong_authority)
    geometries = triplet()
    positions_before = tuple(p.copy() for p in geometries)
    if route == "service":
        # Either construction or first request validation may diagnose the clash.
        with pytest.raises((HTSTRequestError, ValueError)):
            current = make_service(expected_policy, wrong_authority)
            build(current, geometries, constraints=wrong_authority)
    else:
        valid_authority = resolve(expected_policy)
        current = make_service(expected_policy, valid_authority)
        valid = build(current, geometries, constraints=valid_authority)
        invalid = replace(
            valid, constraints=wrong_authority, user_constraints=wrong_authority
        )
        calls = []

        def forbidden(*args):
            calls.append(True)
            raise AssertionError("Wrong-policy authority reached a Hessian")

        with pytest.raises(HTSTRequestError):
            compute_event_prefactors(invalid, forbidden)
        assert calls == []
    assert wrong_authority == before
    for actual, expected in zip(geometries, positions_before, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_original_geometric_snapshot_survives_changed_surroundings_and_actual_crop():
    policy = geometric_policy()
    authority = resolve(policy)
    assert authority.fixed_ids == (0,)
    before_authority = deepcopy(authority)
    current = make_service(policy, authority)
    geometries = triplet()
    # Model a post-premin surrounding atom crossing the original region boundary.
    # This is protocol geometry only, not a claim to have run a minimization.
    for positions in geometries:
        positions[2, 1] = 4.5
    assert all(p[2, 1] > 4.0 for p in geometries)
    full = build(current, geometries, constraints=authority)
    assert full.user_constraints.fixed_ids == (0,)
    check_result(full, IDS, (42, 91))
    assert full.calculation_identity(free_indices=(0, 1, 2)).free_ids == (42, 91)
    assert (
        full.calculation_identity(free_indices=(0, 1, 2)).identity_id
        == full.calculation_identity(free_indices=(0, 2)).identity_id
    )
    rows = np.array([2, 0])
    cropped = replace(
        full,
        **{name: getattr(full, name)[rows].copy() for name in FIELDS},
        types=("Ni", "Ni"),
        center_index=1,
        constraints=full.constraints.crop(rows),
    )
    assert cropped.constraints.atom_ids == (91, 42)
    assert cropped.constraints.fixed_ids == (0,)  # required reference is outside crop
    assert cropped.user_constraints.atom_ids == IDS  # full authority retained
    check_result(cropped, (91, 42), (91, 42))
    assert cropped.calculation_identity(free_indices=(0, 1)).free_ids == (91, 42)

    bad_reference = replace(
        cropped.constraints,
        fixed_positions=((3.0, 5.25, 4.0),),
    )
    with pytest.raises(HTSTRequestError):
        replace(cropped, constraints=bad_reference).validate()
    with pytest.raises(HTSTRequestError):
        replace(cropped, user_constraints=authority.crop(rows)).validate()
    assert authority == before_authority


def test_service_atom_ids_route_remaps_known_snapshot_before_applying_policy():
    policy = RegionConfig(indices=[1])
    authority = resolve(policy)
    before_authority = deepcopy(authority)
    current = make_service(policy, authority)
    rows = np.array([2, 0, 1])
    geometries = tuple(p[rows].copy() for p in triplet())
    before = tuple(p.copy() for p in geometries)
    # Policy index1 means original source ID0, not current row1 (the mover).
    request = build(current, geometries, atom_ids=(91, 42, 0), center=1)
    assert request.constraints.source_ids == IDS
    assert request.constraints.atom_ids == (91, 42, 0)
    assert request.constraints.fixed_ids == (0,)
    check_result(request, (91, 42, 0), (91, 42))
    assert request.calculation_identity(free_indices=(0, 1, 2)).free_ids == (91, 42)
    assert authority == before_authority
    for actual, expected in zip(geometries, before, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_compute_revalidates_foreign_source_references_before_any_batch_submission():
    policy = RegionConfig(indices=[1])
    authority = resolve(policy)
    original_service = make_service(policy, authority)
    valid = replace(
        build(original_service, triplet(), constraints=authority),
        event_key=("original-source",),
    )
    stateless_service = make_service(policy, None)
    shifted = tuple(p + [0.0, 0.25, 0.0] for p in triplet())
    new_source = ResolvedConstraints.resolve(
        shifted[0], ("Ni",) * 3, policy, IDS, cell=CELL, pbc=PBC
    )
    foreign = replace(
        build(stateless_service, shifted, constraints=new_source),
        event_key=("foreign-source",),
    )
    foreign.validate()  # valid for its own newly declared source and identical policy
    assert foreign.descriptor.descriptor_id == valid.descriptor.descriptor_id
    assert foreign.constraints.fixed_positions != authority.fixed_positions
    before_authority = deepcopy(authority)
    foreign_before = tuple(getattr(foreign, name).copy() for name in FIELDS)
    submissions = []

    def forbidden_submit(*args, **kwargs):
        submissions.append((args, kwargs))
        raise AssertionError(
            "Batch dispatched before detecting foreign fixed references"
        )

    original_service.manager = SimpleNamespace(submit=forbidden_submit)
    with pytest.raises(HTSTRequestError):
        original_service.compute([valid, foreign])
    assert submissions == []
    assert authority == before_authority
    for name, expected in zip(FIELDS, foreign_before, strict=True):
        np.testing.assert_array_equal(getattr(foreign, name), expected)
