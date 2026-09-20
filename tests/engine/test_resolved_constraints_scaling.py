"""Resolved constraints stay O(N) on the active-volume hot path.

Under active volume the fixed set is ~N (every atom beyond ``rmov``), so a
per-row tuple scan makes ``local_fixed_indices``, ``validate_positions`` and
``protect_positions`` quadratic (seconds per call at 32k atoms, several calls
per refinement). Membership is a frozenset, the local rows and their reference
coordinates are cached on the frozen payload, and the engine validates a
transported payload once per public operation instead of once per helper
(contracts 7f policy 5, "local_fixed_indices must be O(N)").
"""

import time
from types import SimpleNamespace

import numpy as np
import pytest

from pykmc.activevolume import active_volume as av
from pykmc.engine import lammps as lammps_module
import pykmc.physics as physics
from pykmc.physics import ResolvedConstraints, resolve_event_constraints

from tests import test_av_search_transport as v1


def av_payload(n_atoms, seed=0):
    rng = np.random.default_rng(seed)
    positions = rng.random((n_atoms, 3)) * 60.0
    cfg = SimpleNamespace(
        control=SimpleNamespace(active_volume=True),
        activevolume=SimpleNamespace(rmov=4.0, ract=6.0),
        frozen_atoms=None,
    )
    payload = resolve_event_constraints(
        cfg, positions, ["Ni"] * n_atoms, np.eye(3) * 60.0, (True, True, True), 0
    )
    assert len(payload.fixed_ids) > 0.9 * n_atoms, "the AV mask must be ~N"
    return payload, positions


def test_local_fixed_indices_is_cached_on_the_frozen_payload():
    payload, _ = av_payload(200)
    first = payload.local_fixed_indices
    assert first and first is payload.local_fixed_indices
    user = payload.local_user_fixed_indices
    assert user is payload.local_user_fixed_indices
    # A derived payload is a new instance with its own (consistent) cache.
    cropped = payload.crop(tuple(range(0, 200, 2)))
    assert cropped.local_fixed_indices == tuple(
        i for i, a in enumerate(cropped.atom_ids) if a in set(payload.fixed_ids)
    )


def _seconds(fn):
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def test_constraint_hot_path_is_linear_in_the_atom_count():
    # Quadratic scan: ~0.6 s each at 16k atoms on this class of machine;
    # a linear implementation is two orders of magnitude below the bound.
    payload, positions = av_payload(16000)
    bound = 0.25
    assert _seconds(lambda: payload.local_fixed_indices) < bound
    assert _seconds(lambda: payload.validate_positions(positions)) < bound
    assert _seconds(lambda: payload.protect_positions(positions)) < bound
    assert (
        _seconds(
            lambda: payload.validate_positions(positions, tolerance=0.1, user_only=True)
        )
        < bound
    )


def test_vectorised_validation_and_protection_match_the_reference_semantics():
    payload, positions = av_payload(300)
    fixed = dict(zip(payload.fixed_ids, payload.fixed_positions, strict=True))
    rows = [i for i, a in enumerate(payload.atom_ids) if a in fixed]
    movable = [i for i in range(300) if i not in set(rows)]
    assert rows and movable
    payload.validate_positions(positions)
    shifted = positions.copy()
    shifted[movable] += 0.3
    payload.validate_positions(shifted)  # movable rows are free
    shifted[rows[0]] += 0.3
    with pytest.raises(ValueError, match="fixed reference"):
        payload.validate_positions(shifted)
    protected = payload.protect_positions(shifted)
    np.testing.assert_array_equal(protected[rows], positions[rows])
    np.testing.assert_array_equal(protected[movable], shifted[movable])
    assert not np.shares_memory(protected, shifted)


class _Sentinel(RuntimeError):
    """Raised by the pARTn stand-in once the crop has been prepared."""


@pytest.mark.parametrize("operation", ["search", "refine"])
def test_engine_validates_a_transported_av_payload_once(monkeypatch, operation):
    cfg, types, payload, engine = v1.crop_context()
    cfg.eventsearch.delr_thr = 0.1  # read by the search before pARTn starts
    calls = []
    original = physics.validate_event_constraints

    def counting(*args, **kwargs):
        calls.append(kwargs.get("constraints", args[6] if len(args) > 6 else None))
        return original(*args, **kwargs)

    monkeypatch.setattr(lammps_module, "validate_event_constraints", counting)
    monkeypatch.setattr(av, "validate_event_constraints", counting)
    monkeypatch.setattr(
        lammps_module,
        "pypARTn",
        SimpleNamespace(artn=lambda engine: (_ for _ in ()).throw(_Sentinel())),
    )
    monkeypatch.setattr(engine, "ensure_full_system", lambda positions=None: True)
    common = dict(
        positions=v1.CROP_SOURCE.copy(),
        cell=v1.CROP_CELL,
        types=types,
        constraints=payload,
    )
    with pytest.raises(_Sentinel):
        if operation == "search":
            engine.partn_search(cfg, 1, **common)
        else:
            saddle = v1.CROP_SOURCE[[3]].copy()
            saddle[0, 1] += 0.1
            engine.partn_refine(
                cfg, 1, saddle_idx=np.array([3]), saddle_positions=saddle, **common
            )
    assert "clear" in engine.lmp.commands, "the crop was built before pARTn fired"
    assert len(calls) == 1, "the helpers must trust the payload the engine validated"


def test_direct_helper_calls_still_validate_their_payload():
    cfg, types, payload, engine = v1.crop_context()
    weak = ResolvedConstraints(
        payload.source_ids,
        payload.atom_ids,
        (999, 91),
        tuple(
            dict(zip(payload.fixed_ids, payload.fixed_positions))[i] for i in (999, 91)
        ),
        payload.cell,
        payload.pbc,
        payload.center_id,
        payload.center_position,
        payload.rmov,
        payload.user_policy,
    )
    with pytest.raises(ValueError):
        av.partn_search_AV(
            engine, cfg, 1, v1.CROP_SOURCE.copy(), v1.CROP_CELL, types, constraints=weak
        )
    assert engine.lmp.commands == []
