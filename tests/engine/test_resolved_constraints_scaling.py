"""Resolved constraints stay O(N) on the active-volume hot path.

Under active volume the fixed set is ~N (every atom beyond ``rmov``), so a
per-row tuple scan makes ``local_fixed_indices``, ``validate_positions`` and
``protect_positions`` quadratic (seconds per call at 32k atoms, several calls
per refinement). Membership is a frozenset, the local rows and their reference
coordinates are cached on the frozen payload, so ``local_fixed_indices``
stays O(N).

Only the ``pykmc.physics`` payload tests are here. The engine tests that
check a transported AV payload is validated once per operation belong with
the search-side constraint transport, which is not implemented yet.
"""

import time
from types import SimpleNamespace

import numpy as np
import pytest

from pykmc.physics import resolve_event_constraints


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
