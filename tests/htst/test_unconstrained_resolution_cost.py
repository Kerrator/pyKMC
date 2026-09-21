"""Constant-mode constraint resolution without constraints costs no O(N) Python.

``resolve_event_constraints`` runs once per searched atom, once per
(reference event, atom) refinement pair and once per reconstruction. With
``frozen_atoms`` unset and the active volume off it resolves an empty payload
whose only inputs are the source identities, the cell and the periodic axes,
so an unchanged source returns the same immutable payload instead of
rebuilding it through per-element identity loops. Every input the payload
depends on breaks the cache; a user constraint or an active volume never
enters it; identity strictness is unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pykmc.config import Config
from pykmc.physics import ResolvedConstraints, _indices, resolve_event_constraints
from tests.lifecycle.conftest import DATA_INPUT

N = 64


@pytest.fixture
def unconstrained() -> Config:
    config = Config.from_ini_file(DATA_INPUT)
    assert config.frozen_atoms is None and not config.control.active_volume
    return config


def _source(
    seed: int = 0,
) -> tuple[np.ndarray, list[int], np.ndarray, tuple[bool, ...]]:
    rng = np.random.default_rng(seed)
    cell = np.eye(3) * 30.0
    positions = rng.uniform(0.0, 30.0, size=(N, 3))
    return positions, [1] * N, cell, (True, True, True)


def _global(config: Config, positions: Any, types: Any, cell: Any, pbc: Any) -> Any:
    return ResolvedConstraints.resolve(
        positions, types, config.frozen_atoms, np.arange(N), cell=cell, pbc=pbc
    )


def test_unchanged_unconstrained_source_returns_the_cached_payload(
    unconstrained: Config,
) -> None:
    positions, types, cell, pbc = _source()
    user = _global(unconstrained, positions, types, cell, pbc)
    first = resolve_event_constraints(
        unconstrained,
        positions,
        types,
        cell,
        pbc,
        3,
        np.arange(N),
        user_constraints=user,
    )
    second = resolve_event_constraints(
        unconstrained,
        positions + 0.1,
        types,
        cell,
        pbc,
        11,
        np.arange(N),
        user_constraints=user,
    )
    assert second is first, "an unchanged unconstrained source is resolved once"
    assert first.fixed_ids == () and first.user_fixed_ids == ()
    assert first.source_ids == tuple(range(N)) and first.atom_ids == first.source_ids
    assert first.cell == tuple(tuple(row) for row in cell) and first.pbc == pbc
    assert first.user_policy == user.user_policy
    # The cached payload equals what the full resolution builds.
    rebuilt = ResolvedConstraints.resolve(
        positions, types, None, np.arange(N), cell=cell, pbc=pbc
    )
    assert first == rebuilt and first.user_fixed_ids == rebuilt.user_fixed_ids


@pytest.mark.parametrize("change", ["identities", "cell", "pbc", "count", "authority"])
def test_every_cached_input_breaks_the_cache(
    unconstrained: Config, change: str
) -> None:
    positions, types, cell, pbc = _source()
    user = _global(unconstrained, positions, types, cell, pbc)
    first = resolve_event_constraints(
        unconstrained,
        positions,
        types,
        cell,
        pbc,
        0,
        np.arange(N),
        user_constraints=user,
    )
    ids, other_cell, other_pbc, other_positions, other_types, other_user = (
        np.arange(N),
        cell,
        pbc,
        positions,
        types,
        user,
    )
    if change == "identities":
        ids = np.arange(N)[::-1].copy()
        other_user = ResolvedConstraints.resolve(
            positions, types, None, ids, cell=cell, pbc=pbc
        )
    elif change == "cell":
        other_cell = np.eye(3) * 31.0
        other_user = ResolvedConstraints.resolve(
            positions, types, None, ids, cell=other_cell, pbc=pbc
        )
    elif change == "pbc":
        other_pbc = (True, True, False)
        other_user = ResolvedConstraints.resolve(
            positions, types, None, ids, cell=cell, pbc=other_pbc
        )
    elif change == "count":
        other_positions, other_types = positions[: N - 1], types[: N - 1]
        ids = np.arange(N - 1)
        other_user = ResolvedConstraints.resolve(
            other_positions, other_types, None, ids, cell=cell, pbc=pbc
        )
    elif change == "authority":
        other_user = ResolvedConstraints.resolve(
            positions, types, None, ids, cell=cell, pbc=pbc
        )
    second = resolve_event_constraints(
        unconstrained,
        other_positions,
        other_types,
        other_cell,
        other_pbc,
        0,
        ids,
        user_constraints=other_user,
    )
    assert second is not first
    if change == "identities":
        assert second.source_ids == tuple(int(i) for i in ids)
    elif change == "cell":
        assert second.cell == tuple(tuple(row) for row in other_cell)
    elif change == "pbc":
        assert second.pbc == other_pbc
    elif change == "count":
        assert len(second.source_ids) == N - 1
    else:
        assert second == first and second.user_policy == other_user.user_policy


def test_constraints_and_active_volume_never_use_the_cache(
    unconstrained: Config,
) -> None:
    from pykmc.config import RegionConfig

    positions, types, cell, pbc = _source()
    frozen = unconstrained.model_copy(
        update={"frozen_atoms": RegionConfig(indices=[5])}
    )
    user = _global(frozen, positions, types, cell, pbc)
    assert user.user_fixed_ids == (5,)
    first = resolve_event_constraints(
        frozen, positions, types, cell, pbc, 0, np.arange(N), user_constraints=user
    )
    second = resolve_event_constraints(
        frozen, positions, types, cell, pbc, 0, np.arange(N), user_constraints=user
    )
    assert first.fixed_ids == (5,) and second == first and second is not first
    from pykmc.config import ActiveVolume

    control = unconstrained.control.model_copy(update={"active_volume": True})
    av = unconstrained.model_copy(
        update={
            "control": control,
            "activevolume": ActiveVolume(rmov=8.0, ract=12.0),
        }
    )
    plain = _global(unconstrained, positions, types, cell, pbc)
    shell = resolve_event_constraints(
        av, positions, types, cell, pbc, 0, np.arange(N), user_constraints=plain
    )
    assert shell.center_id == 0 and shell.rmov == av.activevolume.rmov
    assert shell.fixed_ids, "atoms beyond rmov form the active-volume shell"
    again = resolve_event_constraints(
        av, positions, types, cell, pbc, 0, np.arange(N), user_constraints=plain
    )
    assert again == shell and again is not shell


def test_fast_path_still_validates_the_source(unconstrained: Config) -> None:
    positions, types, cell, pbc = _source()
    user = _global(unconstrained, positions, types, cell, pbc)
    resolve_event_constraints(
        unconstrained,
        positions,
        types,
        cell,
        pbc,
        0,
        np.arange(N),
        user_constraints=user,
    )
    bad = positions.copy()
    bad[7, 1] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        resolve_event_constraints(
            unconstrained, bad, types, cell, pbc, 0, np.arange(N), user_constraints=user
        )
    with pytest.raises(ValueError):
        resolve_event_constraints(
            unconstrained,
            positions,
            types,
            cell,
            pbc,
            N,
            np.arange(N),
            user_constraints=user,
        )
    with pytest.raises(ValueError, match="types"):
        resolve_event_constraints(
            unconstrained,
            positions,
            types[:-1],
            cell,
            pbc,
            0,
            np.arange(N),
            user_constraints=user,
        )
    with pytest.raises(ValueError, match="identities differ"):
        resolve_event_constraints(
            unconstrained,
            positions,
            types,
            cell,
            pbc,
            0,
            np.arange(N) + 1,
            user_constraints=user,
        )


@pytest.mark.parametrize(
    "values",
    [
        (1, True),
        [0, 1, 1.0],
        np.array([0.0, 1.0]),
        np.array([True, False]),
        (3, 3),
        (-1, 0),
        range(5),
        np.array([[0, 1], [2, 3]]),
        (0, None),
    ],
    ids=[
        "mixed-bool",
        "float",
        "float-array",
        "bool-array",
        "duplicate",
        "negative",
        "beyond-upper",
        "two-dimensional",
        "object",
    ],
)
def test_identity_strictness_is_unchanged(values: Any) -> None:
    with pytest.raises(ValueError):
        _indices(values, upper=4)


def test_identity_conversion_matches_the_scalar_rule() -> None:
    assert _indices(()) == ()
    assert _indices(range(4)) == (0, 1, 2, 3)
    assert _indices(np.array([2, 0, 1], dtype=np.int32), upper=3) == (2, 0, 1)
    assert _indices((np.int64(4), 1)) == (4, 1)
    assert all(type(i) is int for i in _indices(np.arange(3)))
    assert _indices(iter([5, 6])) == (5, 6)
