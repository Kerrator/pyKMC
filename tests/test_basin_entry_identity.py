"""Basin entry must preserve initialized global identities before reconstruction.

Real System, basin initialization, constraint resolver, reconstruction caller,
and both basin branches run. Only PSR and endpoint minimization are recorded.
No native engine, manager pool, eigenproblem, or basin selector is executed.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import pykmc.basins.basin as basin_module
from pykmc.config import RegionConfig
from pykmc.physics import ResolvedConstraints
from pykmc.result import Ok, PSROutput
from pykmc.system import System


@pytest.mark.parametrize("style", ["global", "global/reconstruction"])
def test_basin_initialize_then_reconstruct_keeps_source_global_ids(monkeypatch, style):
    ids = np.array([42, 17, 8, 91])
    types = np.array(["Ni"] * 4)
    cell = np.diag([10.0, 12.0, 14.0])
    pbc = np.array([True, False, True])
    initial = np.array(
        [[2.25, 3.0, 3.0], [3.5, 3.0, 3.0], [3.5, 3.5, 3.0], [8.0, 3.0, 3.0]]
    )
    saddle, final = initial.copy(), initial.copy()
    saddle[0, 0] += 0.4
    final[0, 0] += 0.8
    cfg = SimpleNamespace(
        control=SimpleNamespace(active_volume=True),
        activevolume=SimpleNamespace(rmov=1.6, ract=5.0),
        frozen_atoms=RegionConfig(
            region_type="plane", normal="y", side="above", threshold=3.3
        ),
        basin=SimpleNamespace(style=style),
        psr=SimpleNamespace(matching_score_thr=1e-10),
        reconstruction=SimpleNamespace(push_fraction=0.25),
    )
    full_source = System(
        positions=initial.copy(),
        types=types.copy(),
        cell=cell.copy(),
        pbc=pbc.copy(),
        index=ids.copy(),
    )
    user = ResolvedConstraints.resolve(
        initial, types, cfg.frozen_atoms, ids, cell=cell, pbc=pbc
    )
    assert user.fixed_ids == (8,)
    neighbors = np.array([2, 0, 1, 3])
    reference = pd.DataFrame(
        [
            dict(
                idx_ref=31,
                initial_positions=initial[neighbors].copy(),
                saddle_positions=saddle[neighbors].copy(),
                final_positions=final[neighbors].copy(),
            )
        ]
    )
    calls = []
    outputs = [final] if style == "global" else [initial, final]

    def endpoint(**kwargs):
        payload = kwargs["constraints"]
        assert payload.source_ids == tuple(ids)
        assert payload.atom_ids == tuple(ids)
        assert payload.center_id == 17  # center is source row 1, not global ID 1
        assert payload.fixed_ids == (8, 91)
        payload.require_preserves(user, cell=cell, pbc=pbc)
        np.testing.assert_array_equal(payload.pbc, pbc)
        np.testing.assert_array_equal(payload.cell, cell)
        np.testing.assert_array_equal(kwargs["positions"][[2, 3]], initial[[2, 3]])
        np.testing.assert_array_equal(kwargs["types"], types)
        calls.append(payload)
        return outputs[len(calls) - 1].copy(), -1.0

    basin = basin_module.BasinsGenericEvents.__new__(basin_module.BasinsGenericEvents)
    basin.config = cfg
    basin.reference_table = SimpleNamespace(table=reference)
    basin.manager = SimpleNamespace(group_minimize_with_results=endpoint)
    basin.global_constraints = user
    basin.states = {}
    # Crucially run the actual entry copy, rather than seeding states by hand.
    basin._initialize(full_source)
    state = basin.states[0]
    np.testing.assert_array_equal(state.system.index, ids)
    assert not np.shares_memory(state.system.index, full_source.index)
    assert not np.shares_memory(state.system.positions, full_source.positions)
    state.environment = object()
    state.neighbors_list = SimpleNamespace(get_neighbors=lambda *_: neighbors.copy())
    match = PSROutput(
        rotation_matrix=np.eye(3),
        translation_matrix=np.zeros(3),
        permutation_matrix=np.arange(4),
        matching_score=0.0,
    )
    monkeypatch.setattr(
        basin_module,
        "PointSetRegistration",
        lambda *_args, **_kwargs: SimpleNamespace(match=lambda: Ok(match)),
    )
    result = basin.system_from_state(0, 31, 1, 0)
    assert result.is_ok()
    assert len(calls) == (1 if style == "global" else 2)
    actual = result.ok_value()
    np.testing.assert_allclose(actual.positions, final, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(actual.index, ids)
    assert not np.shares_memory(actual.index, state.system.index)
    for source in (full_source, state.system):
        np.testing.assert_array_equal(source.positions, initial)
        np.testing.assert_array_equal(source.index, ids)
        np.testing.assert_array_equal(source.types, types)
        np.testing.assert_array_equal(source.cell, cell)
        np.testing.assert_array_equal(source.pbc, pbc)
    assert basin.global_constraints is user
    assert all(payload == calls[0] for payload in calls)
