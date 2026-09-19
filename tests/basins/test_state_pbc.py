"""Independent actual-axis basin state lookup oracle; no native operation.

Exercise the public is_new_state caller. Both systems have the same ordered
species, so this deliberately leaves permutation/state-matching policy alone.
Literal orthorhombic geometry distinguishes open-axis separation from images.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from pykmc.basins.basin import BasinsGenericEvents
from pykmc.system import System


@pytest.mark.parametrize(
    "known_positions,query_positions,expected",
    [
        pytest.param(
            [[1.0, -0.2, 1.0], [2.0, -0.2, 1.0]],
            [[1.0, -0.2, 1.0], [2.0, -0.2, 1.0]],
            91,
            id="negative-open-coordinate-identical-state",
        ),
        pytest.param(
            [[1.0, 0.2, 1.0], [2.0, 0.2, 1.0]],
            [[1.0, 10.2, 1.0], [2.0, 10.2, 1.0]],
            -1,
            id="whole-open-axis-box-is-distinct",
        ),
        pytest.param(
            [[1.0, 0.2, 1.0], [2.0, 0.2, 1.0]],
            [[21.0, 0.2, -9.0], [22.0, 0.2, -9.0]],
            91,
            id="periodic-xz-images-remain-equivalent",
        ),
    ],
)
def test_basin_state_lookup_obeys_source_axes_without_mutating_systems(
    known_positions, query_positions, expected
):
    systems = [
        System(
            positions=np.array(positions, dtype=float),
            types=np.array(["Cu", "Cu"]),
            cell=np.diag([10.0, 10.0, 10.0]),
            pbc=np.array([True, False, True]),
            index=np.array([17, 42]),
        )
        for positions in (known_positions, query_positions)
    ]
    known, query = systems
    saved = [
        {
            key: np.array(getattr(system, key), copy=True)
            for key in ("positions", "types", "cell", "pbc", "index")
        }
        for system in systems
    ]
    basin = BasinsGenericEvents.__new__(BasinsGenericEvents)
    basin.config = SimpleNamespace(
        atomicenvironment=SimpleNamespace(atom_coloring_mode="full")
    )
    basin.states = {91: SimpleNamespace(system=known)}

    # A native periodic-tree domain error on negative open coordinates is a
    # failing consumer result, not a reason to clamp/change the fixture.
    assert basin.is_new_state(query) == expected

    for system, before in zip(systems, saved):
        for key, value in before.items():
            np.testing.assert_array_equal(getattr(system, key), value)
