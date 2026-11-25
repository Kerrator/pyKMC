"""``define_AV`` picks the same atoms, in the same order, as its per-atom loop.

``define_AV`` used to call ASE's ``find_mic`` once per atom of the cell, 3 s per
call on a 54k-atom cell. ``_define_av_loop`` keeps that loop as the reference.
The cells include atoms exactly at ``rmov`` and ``ract``, by the loop's own
distances, so a change in the last bit of a distance would show.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from ase.build import bulk
from ase.cell import Cell
from ase.geometry import find_mic

pytest.importorskip("lammps")
pytest.importorskip("pypARTn")

from pykmc.activevolume.active_volume import define_AV  # noqa: E402


def _define_av_loop(config, central_atom_idx, positions, cell):
    """``define_AV`` as it was: one ``find_mic`` call per atom."""
    r_a = config.activevolume.ract
    r_m = config.activevolume.rmov
    center = positions[central_atom_idx]
    buffer_idx = []
    total_active_idx = []
    for i, pos in enumerate(positions):
        _, distance = find_mic(pos - center, cell, pbc=True)
        if np.abs(distance) <= r_m:
            total_active_idx.append(i)
        elif np.abs(distance) > r_m and np.abs(distance) <= r_a:
            buffer_idx.append(i)
            total_active_idx.append(i)
    buffer_idx = np.array(sorted(buffer_idx))
    av_idx = np.array(sorted(total_active_idx))
    return positions[av_idx], av_idx, buffer_idx


def _loop_distances(positions, centre, cell):
    return np.array(
        [find_mic(p - positions[centre], cell, pbc=True)[1] for p in positions]
    )


def _config(rmov, ract):
    return SimpleNamespace(activevolume=SimpleNamespace(rmov=rmov, ract=ract))


def _simple_cubic():
    """10^3 simple cubic, a = 2 A: shell distances 4 and 6 A are exact."""
    grid = np.indices((10, 10, 10)).reshape(3, -1).T
    return 2.0 * grid.astype(float), np.diag([20.0, 20.0, 20.0])


def _perturbed(positions, cell, seed):
    rng = np.random.default_rng(seed)
    return positions + rng.normal(scale=0.1, size=positions.shape), cell


def _bcc_primitive():
    """8^3 bcc primitive cells: a non-orthogonal cell, 512 atoms."""
    atoms = bulk("Fe", "bcc", a=2.855).repeat((8, 8, 8))
    return atoms.get_positions(), np.asarray(atoms.cell)


def _face_centre(positions, cell):
    """The atom nearest the cell's x = 0 face, so the AV crosses it."""
    scaled = Cell(cell).scaled_positions(positions)
    return int(np.argmin(scaled[:, 0]))


def _at_rank(distances, rank):
    """A distance the loop measured, to put rmov or ract exactly on an atom."""
    return float(np.unique(distances)[rank])


def _assert_same(config, centre, positions, cell):
    expected = _define_av_loop(config, centre, positions, cell)
    got = define_AV(config, centre, positions, cell)
    for name, e, g in zip(
        ("av_positions", "av_idx", "buffer_idx"), expected, got, strict=True
    ):
        np.testing.assert_array_equal(g, e, err_msg=name)
    assert got[1].dtype.kind == "i" and got[2].dtype.kind == "i"
    return got


def test_orthogonal_cell_shells_exactly_at_rmov_and_ract():
    positions, cell = _simple_cubic()
    centre = _face_centre(positions, cell)
    distances = _loop_distances(positions, centre, cell)
    assert np.count_nonzero(distances == 4.0) and np.count_nonzero(distances == 6.0)
    _, av_idx, buffer_idx = _assert_same(_config(4.0, 6.0), centre, positions, cell)
    # The AV crosses the x = 0 face, and both boundary shells are in it.
    assert np.any(positions[av_idx, 0] > 10.0)
    movable = np.setdiff1d(av_idx, buffer_idx)
    assert set(np.flatnonzero(distances == 4.0)) <= set(movable)
    assert set(np.flatnonzero(distances == 6.0)) <= set(buffer_idx)


@pytest.mark.parametrize("seed", [0, 1])
def test_orthogonal_cell_perturbed(seed):
    positions, cell = _perturbed(*_simple_cubic(), seed)
    centre = _face_centre(positions, cell)
    distances = _loop_distances(positions, centre, cell)
    rmov, ract = _at_rank(distances, 60), _at_rank(distances, 300)
    _, av_idx, buffer_idx = _assert_same(_config(rmov, ract), centre, positions, cell)
    assert np.flatnonzero(distances == rmov)[0] not in buffer_idx
    assert np.flatnonzero(distances == ract)[0] in buffer_idx


def test_rmov_on_an_atom_a_single_find_mic_call_would_move():
    """One ``find_mic`` call on all atoms changes some distances in the last
    bit. With rmov on such an atom, the atom must stay on the loop's side."""
    positions, cell = _perturbed(*_simple_cubic(), 0)
    centre = _face_centre(positions, cell)
    distances = _loop_distances(positions, centre, cell)
    _, one_call = find_mic(positions - positions[centre], cell, pbc=True)
    moved = np.flatnonzero((one_call != distances) & (distances < 6.0))
    assert moved.size
    rmov = min(distances[moved[0]], one_call[moved[0]])
    _assert_same(_config(rmov, 8.0), centre, positions, cell)


@pytest.mark.parametrize("seed", [0, 1])
def test_non_orthogonal_cell_perturbed(seed):
    positions, cell = _perturbed(*_bcc_primitive(), seed)
    assert np.count_nonzero(cell - np.diag(np.diag(cell)))
    centre = _face_centre(positions, cell)
    distances = _loop_distances(positions, centre, cell)
    # ract beyond half the shortest cell vector: per atom, find_mic switches
    # from its naive image to the general one inside the AV.
    rmov, ract = _at_rank(distances, 100), _at_rank(distances, 400)
    assert ract > 0.5 * min(Cell(cell).lengths())
    _, av_idx, buffer_idx = _assert_same(_config(rmov, ract), centre, positions, cell)
    assert np.flatnonzero(distances == rmov)[0] not in buffer_idx
    assert np.flatnonzero(distances == ract)[0] in buffer_idx
