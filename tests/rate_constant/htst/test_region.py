"""Tests for the Hessian region builder."""

import ase.build
import numpy as np
from ase.geometry import find_mic

from pykmc.rate_constant.strategies.htst.region import (
    build_region,
    frozen_overlap,
    wrapped_on_open_axis,
)
from pykmc.utils.geometry import translate

PBC = (True, True, True)


def _fcc_vacancy_hop():
    """Return min1, saddle, min2 and the cell of a vacancy hop in fcc Ni."""
    atoms = ase.build.bulk("Ni", "fcc", a=3.52, cubic=True) * (4, 4, 4)
    cell = np.asarray(atoms.cell)
    positions = atoms.get_positions()
    vacancy = positions[0]
    min1 = positions[1:]
    mover = int(np.argmin(np.linalg.norm(min1 - vacancy, axis=1)))
    min2 = min1.copy()
    min2[mover] = vacancy
    saddle = min1.copy()
    saddle[mover] = 0.5 * (min1[mover] + vacancy)
    return min1, saddle, min2, cell, mover


def test_region_is_centred_on_the_saddle_mover() -> None:
    min1, saddle, min2, cell, mover = _fcc_vacancy_hop()

    region = build_region(saddle, cell, PBC, mover, 4.0, 9.0)

    distances = find_mic(saddle - saddle[mover], cell, pbc=True)[1]
    assert np.array_equal(
        region.present[region.displaced], np.flatnonzero(distances <= 4.0)
    )
    # A sphere around the position of the mover at a minimum holds other atoms.
    for minimum in (min1, min2):
        distances = find_mic(saddle - minimum[mover], cell, pbc=True)[1]
        assert not np.array_equal(
            region.present[region.displaced], np.flatnonzero(distances <= 4.0)
        )


def test_present_contains_displaced() -> None:
    _, saddle, _, cell, mover = _fcc_vacancy_hop()

    region = build_region(saddle, cell, PBC, mover, 4.0, 6.0)

    distances = find_mic(saddle - saddle[mover], cell, pbc=True)[1]
    assert np.array_equal(region.present, np.flatnonzero(distances <= 6.0))
    assert np.array_equal(
        region.present[region.displaced], np.flatnonzero(distances <= 4.0)
    )
    assert 0 < region.displaced.size < region.present.size < len(saddle)
    assert region.n_zero_modes == 0


def test_region_covering_the_system_has_three_zero_modes() -> None:
    _, saddle, _, cell, mover = _fcc_vacancy_hop()
    open_z = (True, True, False)

    region = build_region(saddle, cell, open_z, mover, 100.0, 120.0)

    assert np.array_equal(region.present, np.arange(len(saddle)))
    assert np.array_equal(region.displaced, np.arange(len(saddle)))
    # the periodic axes of the system go to the scratch box
    assert region.pbc == open_z
    assert region.n_zero_modes == 3


def test_wrapped_on_open_axis_detects_recentring_wrap() -> None:
    slab = ase.build.fcc100("Ni", size=(4, 4, 8), a=3.52, vacuum=4.0)
    cell = np.asarray(slab.cell)
    pbc = (True, True, False)
    start = slab.get_positions()
    region = build_region(start, cell, pbc, 0, 100.0, 120.0)
    shift = np.array([0.0, 0.0, 0.5 * cell[2, 2]])

    wrapped = translate(start.copy(), shift, cell)
    unwrapped = start + shift

    assert wrapped_on_open_axis(region, (wrapped,), start, cell, pbc)
    assert not wrapped_on_open_axis(region, (unwrapped,), start, cell, pbc)


def test_frozen_overlap_ignores_shell_atoms() -> None:
    _, saddle, _, cell, mover = _fcc_vacancy_hop()
    region = build_region(saddle, cell, PBC, mover, 4.0, 6.0)
    displaced_ids = region.present[region.displaced]
    shell_ids = np.setdiff1d(region.present, displaced_ids)

    assert frozen_overlap(region, shell_ids[:3]) == 0
    assert frozen_overlap(region, np.array([shell_ids[0], displaced_ids[0]])) == 1
    assert frozen_overlap(region, np.array([], dtype=int)) == 0
