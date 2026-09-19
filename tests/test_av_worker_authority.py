"""Independent worker-side N02 authority supplement; no native execution.

Uses the frozen v1 recording endpoint. Native engine entrypoints/helpers remain
real. A supplied execution union cannot replace captured user authority, and
the source AV center is compared using actual PBC before any native mutation.
"""

from dataclasses import replace

import numpy as np
import pytest

from pykmc.activevolume import active_volume as av
from pykmc.engine.lammps import FullSystem, LammpsEngine
from pykmc.physics import ResolvedConstraints


from tests import test_av_search_transport as v1


def assert_untouched(engine, source):
    assert engine.lmp.commands == []
    assert engine.lmp.scatters == []
    np.testing.assert_array_equal(engine.lmp.positions, source)
    assert not engine.system_is_cropped


def crop_user(cfg, types):
    return ResolvedConstraints.resolve(
        v1.CROP_SOURCE,
        types,
        cfg.frozen_atoms,
        v1.CROP_IDS,
        cell=v1.CROP_CELL,
        pbc=v1.PBC,
    )


@pytest.mark.parametrize(
    "center", [True, -1, 1.0, 5], ids=["bool", "negative", "float", "out-of-range"]
)
def test_public_search_rejects_invalid_local_center_before_native_mutation(center):
    cfg, types, execution, engine = v1.crop_context()
    user = crop_user(cfg, types)
    with pytest.raises(ValueError):
        engine.partn_search(
            cfg,
            center,
            positions=v1.CROP_SOURCE.copy(),
            cell=v1.CROP_CELL,
            types=types,
            constraints=execution,
            user_constraints=user,
        )
    assert_untouched(engine, v1.CROP_SOURCE)


@pytest.mark.parametrize("entry", ["public-known", "helper-standalone"])
def test_policy_string_cannot_authorize_omitting_inner_user_fixed_atom(entry):
    cfg, types, execution, engine = v1.crop_context()
    user = crop_user(cfg, types)
    assert user.fixed_ids == (8,)
    # Keep the correct source shell, context and policy string; drop only user
    # ID 8, which lies inside rmov and cannot be caught by an outer-shell check.
    references = dict(zip(execution.fixed_ids, execution.fixed_positions, strict=True))
    weak = replace(
        execution,
        fixed_ids=(999, 91),
        fixed_positions=(references[999], references[91]),
    )
    assert weak.user_policy == user.user_policy
    assert weak.center_id == 42 and weak.rmov == 1.0
    with pytest.raises(ValueError):
        if entry == "public-known":
            engine.partn_search(
                cfg,
                1,
                positions=v1.CROP_SOURCE.copy(),
                cell=v1.CROP_CELL,
                types=types,
                constraints=weak,
                user_constraints=user,
            )
        else:
            # Without an initialized snapshot, full input source policy is the
            # authority. A matching string alone must not validate weak masks.
            av.partn_refine_AV(
                engine,
                cfg,
                1,
                v1.CROP_SOURCE.copy(),
                v1.CROP_CELL,
                types,
                np.array([3]),
                v1.CROP_SOURCE[[3]].copy(),
                constraints=weak,
            )
    assert_untouched(engine, v1.CROP_SOURCE)


@pytest.mark.parametrize(
    "known", [True, False], ids=["captured-user", "standalone-source"]
)
def test_spatial_membership_uses_captured_user_authority_when_present(known):
    cfg = v1.config()
    user = v1.user_snapshot(cfg)
    original = v1.geometries()[0]
    source = v1.geometries(crossed=True)[0]
    # The event's source center and fixed references have not changed. Only
    # previously movable ID 42 entered the configured y>=3.3 region.
    execution = ResolvedConstraints.resolve(
        original,
        v1.TYPES,
        cfg.frozen_atoms,
        v1.IDS,
        cell=v1.CELL,
        pbc=v1.PBC,
        center_id=v1.IDS[v1.SEARCH_CENTER],
        rmov=1.6,
    )
    assert set(execution.fixed_ids) == {8, 91} and user.fixed_ids == (8,)
    engine = LammpsEngine(cfg.lammps, comm=None)
    engine.full_system = FullSystem(
        types=v1.TYPES,
        species=("Ni",),
        masses=(58.6934,),
        cell=v1.CELL.copy(),
        pbc=v1.PBC,
    )
    engine._is_orthorhombic = True
    engine.lmp = v1.CommandEndpoint(source)
    kwargs = dict(constraints=execution, user_constraints=user) if known else {}
    atom_map, center = av.partn_search_AV(
        engine,
        cfg,
        v1.SEARCH_CENTER,
        source.copy(),
        v1.CELL,
        v1.TYPES,
        **kwargs,
    )
    np.testing.assert_array_equal(atom_map, [0, 1, 2, 3])
    np.testing.assert_array_equal(center, [2])
    assert {3, 4}.issubset(engine.lmp.locked())  # user ID 8 and outer ID 91
    assert (1 in engine.lmp.locked()) is (not known)
    assert user.fixed_ids == (8,)
    np.testing.assert_array_equal(source, v1.geometries(crossed=True)[0])


@pytest.mark.parametrize("axis", [0, 1], ids=["periodic-image", "nonperiodic-shift"])
def test_source_center_context_uses_actual_mic_before_crop_mutation(axis):
    cfg, types, execution, engine = v1.crop_context()
    user = crop_user(cfg, types)
    source = v1.CROP_SOURCE.copy()
    source[1] += v1.CROP_CELL[axis]
    if axis == 1:
        with pytest.raises(ValueError):
            av.partn_search_AV(
                engine,
                cfg,
                1,
                source,
                v1.CROP_CELL,
                types,
                constraints=execution,
                user_constraints=user,
            )
        assert_untouched(engine, v1.CROP_SOURCE)
    else:
        atom_map, center = av.partn_search_AV(
            engine,
            cfg,
            1,
            source,
            v1.CROP_CELL,
            types,
            constraints=execution,
            user_constraints=user,
        )
        np.testing.assert_array_equal(atom_map, [1, 2, 3, 4])
        np.testing.assert_array_equal(center, [1])
        assert {2, 4}.issubset(engine.lmp.locked())
    np.testing.assert_array_equal(execution.center_position, v1.CROP_SOURCE[1])
    assert execution.center_id == 42 and tuple(execution.pbc) == v1.PBC
    expected = v1.CROP_SOURCE.copy()
    expected[1] += v1.CROP_CELL[axis]
    np.testing.assert_array_equal(source, expected)
