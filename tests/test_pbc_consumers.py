"""Actual-axis geometry and consumer regression tests.

Only the two geometry helpers need new API bindings: a backward-compatible
``pbc=`` keyword. Missing bindings are labeled API_ALIGNMENT, never emulated.
Analytical orthorhombic coordinates distinguish periodic and open axes; IRA,
minimization and refinement dispatch are explicitly substituted boundaries.
"""

import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import pykmc.basins.basin as basin_module
import pykmc.point_set_registration as psr_module
import pykmc.refinement as refinement_module
from pykmc.activevolume.active_volume import define_AV
from pykmc.neighbors_list import NeighborsList
from pykmc.physics import ResolvedConstraints
from pykmc.reconstruction import Reconstruction
from pykmc.result import Ok
from pykmc.system import System
from pykmc.utils.geometry import compute_delr, push_towards


CELL = np.diag([10.0, 10.0, 10.0])
MIXED = (True, False, True)
ATOL = 2e-12


def helper(function, *args, **kwargs):
    assert "pbc" in inspect.signature(function).parameters, (
        f"API_ALIGNMENT: {function.__name__} requires an explicit pbc keyword"
    )
    return function(*args, **kwargs)


def system(positions, pbc):
    return System(
        types=np.array(["Cu"] * len(positions)),
        positions=positions,
        cell=CELL.copy(),
        pbc=pbc,
        index=np.arange(len(positions)),
    )


@pytest.mark.parametrize(
    "pbc,axes",
    [
        (True, (True, True, True)),
        (False, (False, False, False)),
        ([True, False, True], MIXED),
        (np.array([True, False, True]), MIXED),
    ],
)
def test_system_normalizes_scalar_or_vector_and_preserves_open_coordinates(pbc, axes):
    entry = np.array([[1.0, 2.0, 3.0]])
    obj = system(entry.copy(), pbc)
    assert np.asarray(obj.pbc).shape == (3,)
    assert np.asarray(obj.pbc).dtype.kind == "b"
    assert tuple(obj.pbc) == axes
    proposed = np.array([[21.0, -12.0, 3.0]])
    saved = proposed.copy()
    obj.update_positions(proposed)
    expected = np.array([[1.0 if axes[0] else 21.0, 8.0 if axes[1] else -12.0, 3.0]])
    np.testing.assert_allclose(obj.positions, expected, atol=ATOL, rtol=0)
    np.testing.assert_array_equal(proposed, saved)
    # Mutating caller-owned flags cannot silently change a constructed system.
    if isinstance(pbc, np.ndarray):
        pbc[:] = False
        assert tuple(obj.pbc) == axes


@pytest.mark.parametrize("bad", [[], [True], [True, False], [[True, False, True]]])
def test_invalid_pbc_shape_rejects_before_system_or_input_mutation(bad):
    entry = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    saved = entry.copy()
    with pytest.raises(ValueError):
        system(entry, bad)
    np.testing.assert_array_equal(entry, saved)
    obj = system(entry.copy(), MIXED)
    proposed = np.array([[9.0, -2.0, 3.0]])
    before = obj.positions.copy()
    # A validating property may reject assignment itself; an ordinary field
    # must reject at the consumer before changing the selected row.
    with pytest.raises(ValueError):
        obj.pbc = bad
        obj.update_positions(proposed, atom_idx=np.array([0]))
    np.testing.assert_array_equal(obj.positions, before)
    np.testing.assert_array_equal(proposed, [[9.0, -2.0, 3.0]])


@pytest.mark.parametrize(
    "pbc,expected_push,delta",
    [
        (True, [0.0, 1.4, 1.0], [0.4, 2.4, 0.0]),
        (False, [5.0, -3.6, 1.0], [-9.6, -7.6, 0.0]),
        (MIXED, [0.0, -3.6, 1.0], [0.4, -7.6, 0.0]),
    ],
)
def test_push_and_distance_use_only_the_declared_periodic_axes(
    pbc, expected_push, delta
):
    current = np.array([[9.8, 0.2, 1.0]])
    target = np.array([[0.2, -7.4, 1.0]])
    before = [current.copy(), target.copy()]
    actual = helper(push_towards, current, target, fraction=0.5, cell=CELL, pbc=pbc)
    np.testing.assert_allclose(actual, [expected_push], rtol=0, atol=ATOL)
    distance = helper(compute_delr, current, target, cell=CELL, pbc=pbc)
    assert abs(distance - np.linalg.norm(delta)) < ATOL
    for got, expected in zip((current, target), before):
        np.testing.assert_array_equal(got, expected)


class EndpointManager:
    def __init__(self, outputs, proposals=None):
        self.outputs = [p.copy() for p in outputs]
        self.proposals = proposals
        self.calls = []

    def group_minimize_with_results(self, **kwargs):
        i = len(self.calls)
        positions = np.array(kwargs["positions"], copy=True)
        self.calls.append(positions)
        if self.proposals is not None:
            np.testing.assert_allclose(positions, self.proposals[i], atol=ATOL, rtol=0)
        return self.outputs[i].copy(), -1.0


def reconstruct(manager, pbc, first, saddle, final):
    cfg = SimpleNamespace(
        reconstruction=SimpleNamespace(push_fraction=0.5),
        psr=SimpleNamespace(matching_score_thr=1e-10),
    )
    return Reconstruction(cfg, manager, types=["Cu"], pbc=pbc).reconstruct(
        first, final, saddle, CELL, 1e-10
    )


@pytest.mark.parametrize(
    "pbc,proposals",
    [
        (True, [[[0.0, 1.4, 1.0]], [[9.6, 9.0, 1.0]]]),
        (False, [[[5.0, -3.6, 1.0]], [[9.6, 4.0, 1.0]]]),
        (MIXED, [[[0.0, -3.6, 1.0]], [[9.6, 4.0, 1.0]]]),
    ],
)
def test_reconstruction_transports_actual_axes_to_both_pushes_and_comparisons(
    pbc, proposals
):
    first = np.array([[0.2, -7.4, 1.0]])
    saddle = np.array([[9.8, 0.2, 1.0]])
    final = np.array([[9.4, 7.8, 1.0]])
    before = [p.copy() for p in (first, saddle, final)]
    # Independent per-endpoint x image changes are physically equivalent.
    outputs = (
        [first, final]
        if pbc is False
        else [first + [20.0, 0.0, 0.0], final - [10.0, 0.0, 0.0]]
    )
    manager = EndpointManager(outputs, proposals)
    result = reconstruct(manager, pbc, first, saddle, final)
    assert result.is_ok()
    assert len(manager.calls) == 2
    for actual, saved in zip((first, saddle, final), before):
        np.testing.assert_array_equal(actual, saved)


def test_reconstruction_rejects_a_whole_box_error_on_a_nonperiodic_axis():
    first = np.array([[2.0, -1.0, 1.0]])
    saddle = np.array([[2.1, -0.5, 1.0]])
    final = np.array([[2.2, 0.0, 1.0]])
    manager = EndpointManager([first + [0.0, 10.0, 0.0], final])
    result = reconstruct(manager, MIXED, first, saddle, final)
    assert not result.is_ok(), (
        "a physically different open-axis endpoint was accepted as an image"
    )
    assert len(manager.calls) == 1


def test_invalid_reconstruction_axes_reject_before_endpoint_dispatch_or_input_mutation():
    first = np.array([[1.0, 1.0, 1.0]])
    saddle, final = first + 0.1, first + 0.2
    saved = [p.copy() for p in (first, saddle, final)]
    manager = EndpointManager([first, final])
    with pytest.raises(ValueError):
        reconstruct(manager, [True, False], first, saddle, final)
    assert manager.calls == []
    for actual, expected in zip((first, saddle, final), saved):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "pbc,has_neighbor", [(True, True), (False, False), (MIXED, False)]
)
def test_no_false_periodic_neighbors_and_av_membership_agrees(pbc, has_neighbor):
    positions = np.array([[1.0, 0.1, 1.0], [1.0, 9.9, 1.0]])
    obj = system(positions.copy(), pbc)
    neighbors = NeighborsList(obj, rnei=0.3, rcut=0.3)
    assert set(neighbors.get_neighbors("rnei", 0)) == ({1} if has_neighbor else set())
    assert set(neighbors.get_neighbors("rcut", 0)) == ({0, 1} if has_neighbor else {0})
    cfg = SimpleNamespace(activevolume=SimpleNamespace(ract=0.3, rmov=0.25))
    _, av_ids, buffer_ids = define_AV(cfg, 0, positions, CELL, pbc=pbc)
    assert set(av_ids) == ({0, 1} if has_neighbor else {0})
    assert len(buffer_ids) == 0
    resolved = ResolvedConstraints.resolve(
        positions,
        ["Cu", "Cu"],
        atom_ids=(17, 42),
        cell=CELL,
        pbc=pbc,
        center_id=17,
        rmov=0.25,
    )
    assert set(resolved.fixed_ids) == (set() if has_neighbor else {42})
    np.testing.assert_array_equal(obj.positions, positions)


def test_psr_unwraps_only_periodic_axes_before_the_ira_boundary(monkeypatch):
    positions = np.array([[0.1, 0.1, 1.0], [0.1, 9.9, 1.0], [9.9, 0.1, 1.0]])
    expected = np.array([[0.1, 0.1, 1.0], [0.1, 9.9, 1.0], [-0.1, 0.1, 1.0]])
    obj = system(positions.copy(), MIXED)
    seen = []

    class IRA:
        def match(self, n1, t1, p1, n2, t2, p2, factor):
            seen.append(p1.copy())
            return np.eye(3), np.zeros(3), np.arange(3), 0.0

    monkeypatch.setattr(psr_module.ira_mod, "IRA", IRA)
    cfg = SimpleNamespace(
        psr=SimpleNamespace(style="ira"),
        atomicenvironment=SimpleNamespace(atom_coloring_mode="full"),
        ira=SimpleNamespace(kmax_factor=2.0),
    )
    ref = pd.Series(dict(initial_positions=expected.copy(), types=["Cu"] * 3))
    neighbors = SimpleNamespace(get_neighbors=lambda *_: np.arange(3))
    assert psr_module.PointSetRegistration(cfg, obj, ref, neighbors, 0).match().is_ok()
    assert len(seen) == 1
    np.testing.assert_allclose(seen[0], expected, atol=ATOL, rtol=0)
    np.testing.assert_array_equal(obj.positions, positions)


def test_refinement_final_context_preserves_nonperiodic_coordinate(monkeypatch):
    first = np.array([[1.0, 0.2, 1.0]])
    final = np.array([[1.2, -0.5, 1.0]])
    obj = system(first.copy(), MIXED)
    identity = SimpleNamespace(
        rotation_matrix=np.eye(3),
        translation_matrix=np.zeros(3),
        permutation_matrix=np.arange(1),
        matching_score=0.0,
    )
    monkeypatch.setattr(
        refinement_module,
        "PointSetRegistration",
        lambda *_: SimpleNamespace(match=lambda: Ok(identity)),
    )
    ref = pd.Series(
        dict(
            initial_positions=first.copy(),
            saddle_positions=first + 0.1,
            final_positions=final.copy(),
            energy_barrier=1.0,
            idx_ref=31,
            sym_matrix=[np.eye(3)],
            sym_perm=[np.arange(1)],
        )
    )
    instance = refinement_module.Refinement.__new__(refinement_module.Refinement)
    instance.config = SimpleNamespace(
        control=SimpleNamespace(active_volume=False),
        psr=SimpleNamespace(matching_score_thr=1e-10),
    )
    instance.system = obj
    instance.neighbors_list = SimpleNamespace(get_neighbors=lambda *_: np.arange(1))
    instance._carry_prefactors = False
    instance.global_constraints = None  # constructor attribute the double bypasses
    context = {}
    futures = instance.refine_single(0, ref, 0.0, context, e_thr=0.0)
    assert len(futures) == 1
    np.testing.assert_allclose(
        context[futures[0]]["min2_positions"], final, atol=ATOL, rtol=0
    )
    np.testing.assert_allclose(obj.positions, first, atol=ATOL, rtol=0)


def test_basin_absorbing_refinement_copies_source_axes_before_any_energy_call(
    monkeypatch,
):
    source = system(np.array([[1.0, -0.2, 1.0]]), MIXED)
    seen = []
    stop = RuntimeError("intentional stop at System construction boundary")

    def system_boundary(**kwargs):
        seen.append(kwargs)
        assert tuple(np.broadcast_to(np.asarray(kwargs["pbc"]), (3,))) == MIXED
        raise stop

    monkeypatch.setattr(basin_module, "System", system_boundary)
    basin = basin_module.BasinsGenericEvents.__new__(basin_module.BasinsGenericEvents)
    basin.states = {0: SimpleNamespace(system=source)}
    basin.connectivity_table = SimpleNamespace(
        df=pd.DataFrame([dict(transient=False, state=0)])
    )
    with pytest.raises(RuntimeError) as raised:
        basin.refine_absorbing(source)
    assert raised.value is stop
    assert len(seen) == 1
    np.testing.assert_array_equal(source.positions, [[1.0, -0.2, 1.0]])


@pytest.mark.parametrize("flags", [0, 1, [0, 1, 0], [True, False, "True"]])
def test_nonboolean_pbc_rejects_without_changing_positions(flags):
    source = np.array([[1.0, -2.0, 3.0]])
    with pytest.raises(ValueError):
        system(source, flags)
    np.testing.assert_array_equal(source, [[1.0, -2.0, 3.0]])


def test_reference_recentering_retains_nonperiodic_event_displacement():
    from pykmc.eventsearch import EventSearch

    search = EventSearch.__new__(EventSearch)
    search.system = system(np.array([[1.0, 1.0, 1.0]]), MIXED)
    # Constructor attribute the double bypasses; constant style keeps the
    # catalogue-centred representation this test pins.
    search.config = SimpleNamespace(rateconstant=SimpleNamespace(style="artn"))
    event = SimpleNamespace(
        move_atom_index=0,
        min1_positions=np.array([[1.0, 1.0, 1.0]]),
        saddle_positions=np.array([[1.0, -3.0, 1.0]]),
        min2_positions=np.array([[1.0, -7.0, 1.0]]),
    )
    search._center_event_positions(event)
    np.testing.assert_allclose(event.min1_positions, [[5.0, 5.0, 5.0]])
    np.testing.assert_allclose(event.saddle_positions, [[5.0, 1.0, 5.0]])
    np.testing.assert_allclose(event.min2_positions, [[5.0, -3.0, 5.0]])
