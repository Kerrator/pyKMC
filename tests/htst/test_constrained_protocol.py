"""Independent constraint protocol; no native engine is started.

The mobile ID 42 has U=(q**2-1)**2+2*(y**2+z**2), q=x-3.
Fixed ID 0 has curvature 4+12*q**2 in each of its three coordinates;
mobile spectator ID 91 has constant curvatures (9,16,25). Fixed ID 17
has constant curvatures (36,49,64). Spectator factors cancel in Vineyard.
Thus excluding fixed IDs gives sqrt(8*eV/(mass*amu*A**2))/(2*pi).
Including fixed ID 0 spuriously multiplies the result by eight.

The force-model descriptor in service tests is only protocol metadata;
the literal callback describes this polynomial, not a native pair potential.
"""

from copy import deepcopy
from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest

from pykmc.config import RateConstantConfig, RegionConfig
from pykmc.htst import (
    HTSTEventRequest,
    HTSTRequestError,
    HTSTSettings,
    PrefactorRejection,
    compute_event_prefactors,
)
from pykmc.physics import ResolvedConstraints
from pykmc.rate_constant import create_rate_constant
from pykmc.rate_constant.prefactors import PrefactorService


IDS = (42, 0, 91, 17)
FIXED_IDS = (0, 17)
MASS = 58.6934
CELL = np.diag([10.0, 12.0, 14.0])
PBC = (True, False, True)
EXACT_HZ = math.sqrt(8.0 * 1.602176634e-19 / (MASS * 1.66053906660e-27 * 1e-20)) / (
    2.0 * math.pi
)
POSITION_FIELDS = ("min1_positions", "saddle_positions", "min2_positions")


def native_config():
    return SimpleNamespace(
        pair_style="zero 10.0",
        pair_coeff="* *",
        min_style="cg",
        frz_min="0.0 1e-12 100 1000",
        minimize="0.0 1e-12 100 1000",
        verbosity=0,
    )


def request_fixture(*, context=True):
    source = np.array(
        [[2.0, 3.0, 4.0], [3.0, 5.0, 4.0], [3.0, 3.0, 6.0], [7.0, 7.0, 7.0]]
    )
    saddle, final = source.copy(), source.copy()
    saddle[0, 0] = 3.0
    final[0, 0] = 4.0
    constraints = ResolvedConstraints.resolve(
        source,
        ("Ni",) * 4,
        RegionConfig(indices=[1, 3]),
        IDS,
        **({"cell": CELL.copy(), "pbc": PBC} if context else {}),
    )
    assert constraints.atom_ids == IDS
    assert constraints.fixed_ids == FIXED_IDS
    # Deliberate trap: fixed global ID 0 is on local row 1, not row 0.
    assert constraints.local_fixed_indices == (1, 3)
    return HTSTEventRequest(
        event_key=("n04-protocol",),
        min1_positions=source,
        saddle_positions=saddle,
        min2_positions=final,
        types=("Ni",) * 4,
        species=("Ni",),
        masses=(MASS,),
        cell=CELL.copy(),
        pbc=PBC,
        center_index=0,
        settings=HTSTSettings(free_radius=3.0, nu0_min_hz=1e10),
        constraints=constraints,
    )


def row_view(request, rows):
    """Apply the same literal row map to geometry and resolved source identity."""
    rows = np.array(rows, dtype=int)
    old_center = request.center_index
    return replace(
        request,
        **{name: getattr(request, name)[rows].copy() for name in POSITION_FIELDS},
        types=tuple(request.types[i] for i in rows),
        center_index=list(rows).index(old_center),
        constraints=request.constraints.crop(rows),
    )


def snapshot(request):
    return (
        tuple(getattr(request, name).copy() for name in POSITION_FIELDS),
        request.cell.copy(),
        deepcopy(request.constraints),
    )


def assert_unchanged(request, before):
    for name, positions in zip(POSITION_FIELDS, before[0], strict=True):
        np.testing.assert_array_equal(getattr(request, name), positions)
    np.testing.assert_array_equal(request.cell, before[1])
    assert request.constraints == before[2]


class LiteralHessian:
    """No production selector, mode helper, mass helper or frequency oracle."""

    def __init__(self, atom_ids):
        self.ids = tuple(atom_ids)
        self.calls = []

    def __call__(self, positions, free):
        self.calls.append((np.array(positions, copy=True), tuple(int(i) for i in free)))
        q = float(positions[self.ids.index(42), 0]) - 3.0
        diagonal = {
            42: (12.0 * q * q - 4.0, 4.0, 4.0),
            0: (4.0 + 12.0 * q * q,) * 3,
            91: (9.0, 16.0, 25.0),
            17: (36.0, 49.0, 64.0),
        }
        values = [value / MASS for i in free for value in diagonal[self.ids[int(i)]]]
        return np.diag(values)


def assert_accepted(request, result, oracle, expected_ids, *, backward=True):
    assert result.event_key == request.event_key
    assert result.n_free == len(expected_ids)
    assert len(oracle.calls) == (3 if backward else 2)
    for positions, rows in oracle.calls:
        assert set(oracle.ids[i] for i in rows) == set(expected_ids)
        assert len(rows) == len(expected_ids)
        assert not set(oracle.ids[i] for i in rows).intersection(FIXED_IDS)
        # Callback positions are the exact event points, never projected/repaired.
        assert any(
            np.array_equal(positions, getattr(request, f)) for f in POSITION_FIELDS
        )
    for direction in (
        (result.forward, result.backward) if backward else (result.forward,)
    ):
        assert direction.ok, direction.reason
        assert direction.n_free == len(expected_ids)
        assert direction.n_positive_min == 3 * len(expected_ids)
        assert direction.n_negative_saddle == 1
        assert direction.nu0_hz == pytest.approx(EXACT_HZ, rel=1e-9, abs=0.0)
    if not backward:
        assert result.backward.status == "skipped"
        assert result.backward.nu0_hz is None
        assert result.backward.reason_code is None


@pytest.mark.parametrize(
    ("rows", "explicit", "expected_ids"),
    [
        ((0, 1, 2, 3), None, (42, 91)),
        ((3, 1, 2, 0), None, (91, 42)),
        ((0, 1, 2, 3), (3, 2, 1, 0), (42, 91)),
        ((0, 1, 2, 3), (1, 0), (42,)),
        ((1, 0), (1, 0), (42,)),
    ],
    ids=("automatic", "permuted", "explicit-all", "explicit-subset", "actual-crop"),
)
def test_common_set_subtracts_fixed_global_ids(rows, explicit, expected_ids):
    original = request_fixture()
    original_before = snapshot(original)
    request = row_view(original, rows)
    before = snapshot(request)
    assert request.constraints.source_ids == IDS
    assert request.constraints.fixed_ids == FIXED_IDS
    if len(rows) == 2:
        assert request.constraints.atom_ids == (0, 42)
        assert 17 not in request.constraints.atom_ids
        assert 17 in request.constraints.fixed_ids
    oracle = LiteralHessian(tuple(IDS[i] for i in rows))
    free = None if explicit is None else np.array(explicit)
    free_before = None if free is None else free.copy()
    result = compute_event_prefactors(request, oracle, free_indices=free)
    assert_accepted(request, result, oracle, expected_ids)
    assert_unchanged(request, before)
    assert_unchanged(original, original_before)
    if free is not None:
        np.testing.assert_array_equal(free, free_before)


@pytest.mark.parametrize("context", [False, True], ids=("contextless", "bound"))
def test_periodic_fixed_images_use_actual_request_axes(context):
    request = request_fixture(context=context)
    request.saddle_positions[1, 0] += 10.0
    request.min2_positions[1, 2] -= 14.0
    before = snapshot(request)
    oracle = LiteralHessian(IDS)
    request.validate()
    result = compute_event_prefactors(request, oracle)
    assert_accepted(request, result, oracle, (42, 91))
    assert_unchanged(request, before)


@pytest.mark.parametrize("route", ["global-snapshot", "policy-with-source-ids"])
def test_service_uses_actual_pbc_without_replacing_source_references(route):
    original = request_fixture(context=False)
    original.saddle_positions[1, 0] += 10.0
    original.min2_positions[1, 2] -= 14.0
    before = snapshot(original)
    config = SimpleNamespace(
        lammps=native_config(),
        frozen_atoms=RegionConfig(indices=[1, 3]),
        rateconstant=RateConstantConfig(
            style="htst", free_radius=3.0, nu0_min_THz=0.01
        ),
    )
    service = PrefactorService(
        config,
        SimpleNamespace(),
        create_rate_constant(config.rateconstant),
        species_masses=(("Ni",), (MASS,)),
        global_constraints=original.constraints if route == "global-snapshot" else None,
    )
    arguments = dict(
        event_key=("service", route),
        **{field: getattr(original, field) for field in POSITION_FIELDS},
        cell=original.cell,
        pbc=original.pbc,
        types=original.types,
        center_index=0,
    )
    if route == "policy-with-source-ids":
        arguments["atom_ids"] = IDS
    request = service.build_request(**arguments)
    assert request.constraints.atom_ids == IDS
    assert request.constraints.fixed_ids == FIXED_IDS
    assert request.constraints.fixed_positions == original.constraints.fixed_positions
    oracle = LiteralHessian(IDS)
    result = compute_event_prefactors(request, oracle)
    assert_accepted(request, result, oracle, (42, 91))
    assert_unchanged(original, before)
    for field in POSITION_FIELDS:
        assert not np.shares_memory(getattr(request, field), getattr(original, field))
    invalid_saddle = original.saddle_positions.copy()
    invalid_saddle[1, 1] += 12.0
    invalid_before = invalid_saddle.copy()
    with pytest.raises(HTSTRequestError):
        service.build_request(**(arguments | {"saddle_positions": invalid_saddle}))
    np.testing.assert_array_equal(invalid_saddle, invalid_before)
    assert_unchanged(original, before)


def assert_invalid_before_work(request, monkeypatch):
    """Exercise public validation, kernel and native entry before scratch startup."""
    before = snapshot(request)
    with pytest.raises(HTSTRequestError):
        request.validate()
    calls = []

    def forbidden_hessian(*args):
        calls.append("hessian")
        raise AssertionError("Invalid constrained event reached Hessian work")

    with pytest.raises(HTSTRequestError):
        compute_event_prefactors(request, forbidden_hessian)

    # Construction alone starts no LAMMPS handle or MPI worker/session.
    from pykmc.engine.htst_lammps import LammpsHTSTExtension
    from pykmc.engine.lammps import LammpsEngine

    parent = LammpsEngine(native_config(), comm=None)
    extension = LammpsHTSTExtension(parent)

    def forbidden_scratch():
        calls.append("scratch")
        raise AssertionError("Invalid constrained event reached scratch creation")

    monkeypatch.setattr(extension, "_new_scratch", forbidden_scratch)
    with pytest.raises(HTSTRequestError):
        extension.compute_event_prefactors(request)
    assert parent.lmp is None
    assert parent.full_system is None
    assert calls == []
    assert_unchanged(request, before)


@pytest.mark.parametrize("field", POSITION_FIELDS)
@pytest.mark.parametrize("context", [False, True], ids=("contextless", "bound"))
def test_changed_fixed_triplet_rejected_before_hessian_or_native(
    field, context, monkeypatch
):
    request = request_fixture(context=context)
    # Y is nonperiodic. A full cell-length translation must not be treated as an image.
    getattr(request, field)[1, 1] += 12.0
    assert_invalid_before_work(request, monkeypatch)


@pytest.mark.parametrize("mismatch", ["cell", "pbc"])
def test_bound_constraint_context_must_match_request(mismatch, monkeypatch):
    request = request_fixture()
    if mismatch == "cell":
        wrong_cell = CELL.copy()
        wrong_cell[0, 0] = 11.0
        constraints = replace(
            request.constraints,
            cell=tuple(tuple(float(x) for x in row) for row in wrong_cell),
        )
    else:
        constraints = replace(request.constraints, pbc=(True, True, True))
    request = replace(request, constraints=constraints)
    assert_invalid_before_work(request, monkeypatch)


@pytest.mark.parametrize("backward", [True, False])
@pytest.mark.parametrize("selection", ["all-fixed", "explicit-fixed-only"])
def test_empty_common_set_is_unavailable_without_hessian(selection, backward):
    request = request_fixture()
    free = np.array([1])
    if selection == "all-fixed":
        # A fully fixed event is coordinate-consistent; no Hessian should classify it.
        source = request.min1_positions.copy()
        constraints = ResolvedConstraints.resolve(
            source,
            request.types,
            RegionConfig(indices=[0, 1, 2, 3]),
            IDS,
            cell=CELL,
            pbc=PBC,
        )
        request = replace(
            request,
            saddle_positions=source.copy(),
            min2_positions=source.copy(),
            constraints=constraints,
        )
        free = None
    before = snapshot(request)
    calls = []

    def forbidden_hessian(*args):
        calls.append(True)
        raise AssertionError("Empty common set requested a Hessian")

    result = compute_event_prefactors(
        request, forbidden_hessian, free_indices=free, compute_backward=backward
    )
    assert result.n_free == 0
    assert result.forward.status == "rejected"
    assert result.forward.reason_code is PrefactorRejection.EMPTY_FREE_REGION
    assert result.forward.nu0_hz is None
    if backward:
        assert result.backward.reason_code is PrefactorRejection.EMPTY_FREE_REGION
        assert result.backward.status == "rejected"
    else:
        assert result.backward.status == "skipped"
        assert result.backward.reason_code is None
    assert result.backward.nu0_hz is None
    assert calls == []
    assert_unchanged(request, before)


def test_forward_only_uses_same_constrained_set_and_skips_second_minimum():
    request = request_fixture()
    before = snapshot(request)
    oracle = LiteralHessian(IDS)
    result = compute_event_prefactors(request, oracle, compute_backward=False)
    assert_accepted(request, result, oracle, (42, 91), backward=False)
    np.testing.assert_array_equal(oracle.calls[0][0], request.saddle_positions)
    np.testing.assert_array_equal(oracle.calls[1][0], request.min1_positions)
    assert_unchanged(request, before)
