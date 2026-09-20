"""Periodic scatter must preserve analytic pair physics and atom identity.

Real serial LAMMPS, two distinguishable atoms, built-in LJ potential. No MPI
worker pool, potential file, pARTn, minimization or numerical derivative.
The exact energy and force use the known physical separation; expected
positions use predeclared lattice translations and a known orthogonal rotation,
not the production coordinate-normalization implementation.

These native boundary regressions complement full-system HTST image tests.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from ase.cell import Cell
from mpi4py import MPI

from pykmc.activevolume import active_volume
from pykmc.engine.lammps import LammpsEngine


def rotation():
    az, ax = np.deg2rad([37.0, 23.0])
    rz = np.array(
        [[np.cos(az), -np.sin(az), 0.0], [np.sin(az), np.cos(az), 0.0], [0.0, 0.0, 1.0]]
    )
    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(ax), -np.sin(ax)], [0.0, np.sin(ax), np.cos(ax)]]
    )
    return rz @ rx


def geometry(kind):
    restricted = (
        np.diag([12.0, 11.0, 10.0])
        if kind.startswith("ortho")
        else np.array([[12.0, 0.0, 0.0], [2.0, 11.0, 0.0], [1.5, -1.0, 10.0]])
    )
    turn = rotation() if kind.startswith("rotated") else np.eye(3)
    pbc = (
        (True, False, True)
        if kind.endswith("mixed")
        else (False, False, False)
        if kind.endswith("nonperiodic")
        else (True, True, True)
    )
    first = np.array([0.27, 0.31, 0.42]) @ restricted
    positions = np.array([first, first + [1.3, 0.0, 0.0]]) @ turn
    return restricted @ turn, pbc, positions


@contextmanager
def initialized(kind):
    cell, pbc, positions = geometry(kind)
    config = SimpleNamespace(
        pair_style="lj/cut 2.5",
        pair_coeff="* * 1.0 1.0",
        min_style="cg",
        minimize="1e-8 1e-8 100 1000",
        frz_min="1e-8 1e-8 100 1000",
        verbosity=0,
    )
    engine = LammpsEngine(config, comm=MPI.COMM_SELF, engine_id=9032)
    try:
        engine.start()
        engine.initialize_parameters()
        engine.initialize_system(
            types=("Ni", "Fe"),
            positions=positions.copy(),
            cell=Cell(cell),
            pbc=pbc,
            species=("Fe", "Ni"),
            masses=(56.0, 60.0),
        )
        engine.initialize_potential()
        yield engine, cell, pbc, positions
    finally:
        engine.close()
        assert engine.lmp is None


def analytic_pair(positions):
    delta = positions[1] - positions[0]
    distance = float(np.linalg.norm(delta))
    assert 1.1 < distance < 2.0  # one pair inside cutoff, images all beyond it
    energy = 4.0 * (distance**-12 - distance**-6)
    first_force = 24.0 * (distance**-8 - 2.0 * distance**-14) * delta
    return energy, np.array([first_force, -first_force])


def assert_physical_pair(engine, positions):
    energy, forces = analytic_pair(positions)
    assert engine.get_total_energy() == pytest.approx(energy, rel=0, abs=2e-11)
    np.testing.assert_allclose(engine.get_forces(), forces, rtol=0, atol=2e-10)
    assert int(engine.lmp.get_natoms()) == 2
    ids = tuple(np.ctypeslib.as_array(engine.lmp.gather_atoms("id", 0, 1)))
    types = tuple(np.ctypeslib.as_array(engine.lmp.gather_atoms("type", 0, 1)))
    assert ids == (1, 2)
    assert types == (2, 1)


@pytest.mark.parametrize(
    "kind",
    [
        "ortho_full",
        "triclinic_full",
        "rotated_full",
        "rotated_mixed",
        "ortho_nonperiodic",
    ],
)
def test_independent_atom_images_preserve_frame_identity_and_physics(kind):
    with initialized(kind) as (engine, cell, pbc, initial):
        assert_physical_pair(engine, initial)
        expected = initial.copy()
        if not all(pbc):
            # A real relative displacement along a nonperiodic cell axis must
            # remain present; the analytic energy/force are recomputed for it.
            axis = next(i for i, periodic in enumerate(pbc) if not periodic)
            expected[1] += 0.23 * cell[axis] / np.linalg.norm(cell[axis])
        images = np.array([[3, -2, 4], [-4, 5, -3]], dtype=float)
        images[:, np.logical_not(pbc)] = 0
        supplied = expected + images @ cell
        # Deliberately non-contiguous and read-only: normalization cannot alter
        # caller data or require writable storage.
        storage = np.zeros((2, 6))
        storage[:, ::2] = supplied
        request = storage[:, ::2]
        request.setflags(write=False)
        saved = request.copy()
        descriptor = engine.full_system
        engine.set_positions(request)
        np.testing.assert_array_equal(request, saved)
        assert not request.flags.writeable
        assert engine.full_system is descriptor
        np.testing.assert_allclose(engine.get_positions(), expected, rtol=0, atol=2e-11)
        assert_physical_pair(engine, expected)


@pytest.mark.parametrize("kind", ["ortho_full", "ortho_mixed"])
def test_active_volume_scatter_helper_has_same_periodic_boundary(kind):
    # The helper is also used by the AV refine path; its supported cells are
    # orthorhombic. This tests that actual entry point without saddle search.
    with initialized(kind) as (engine, cell, pbc, expected):
        images = np.array([[3, -2, 4], [-4, 5, -3]], dtype=float)
        images[:, np.logical_not(pbc)] = 0
        request = expected + images @ cell
        request.setflags(write=False)
        saved = request.copy()
        descriptor = engine.full_system
        active_volume.set_positions(engine, request)
        np.testing.assert_array_equal(request, saved)
        assert engine.full_system is descriptor
        np.testing.assert_allclose(engine.get_positions(), expected, rtol=0, atol=2e-11)
        assert_physical_pair(engine, expected)


@pytest.mark.parametrize("poison", ["nan", "inf", "minus_inf", "short", "flat"])
def test_bad_request_rejects_before_native_or_caller_mutation(poison):
    with initialized("rotated_mixed") as (engine, cell, pbc, initial):
        assert_physical_pair(engine, initial)
        before = engine.get_positions().copy()
        descriptor = engine.full_system
        if poison == "short":
            request = initial[:1].copy()
        elif poison == "flat":
            request = initial.ravel().copy()
        else:
            request = initial.copy()
            request[1, 2] = {"nan": np.nan, "inf": np.inf, "minus_inf": -np.inf}[poison]
        request.setflags(write=False)
        saved = request.copy()
        with pytest.raises(ValueError):
            engine.set_positions(request)
        np.testing.assert_array_equal(request, saved)
        np.testing.assert_array_equal(engine.get_positions(), before)
        assert engine.full_system is descriptor
        assert_physical_pair(engine, initial)


@pytest.mark.parametrize("kind", ["rotated_mixed", "ortho_nonperiodic"])
def test_nonperiodic_lattice_translation_is_a_physical_displacement(kind):
    # A nonperiodic lattice translation is NOT a representation equivalence:
    # the scatter contract folds periodic images only and never a nonperiodic
    # direction. The displaced coordinate must therefore reach the native
    # instance as given (inspected without run 0), neither rejected nor
    # silently folded back onto the initial image; the caller's read-only
    # array is untouched. This does not claim support for evaluating atoms
    # outside a fixed box, so the pair is reset before native physics.
    with initialized(kind) as (engine, cell, pbc, initial):
        descriptor = engine.full_system
        axis = next(i for i, periodic in enumerate(pbc) if not periodic)
        supplied = initial.copy()
        supplied[1] += 2 * cell[axis]
        supplied.setflags(write=False)
        saved = supplied.copy()
        engine.set_positions(supplied)
        placed = engine.get_positions()
        np.testing.assert_allclose(placed, supplied, rtol=0, atol=2e-11)
        # Discriminating against folding: the nonperiodic shift survives.
        assert np.linalg.norm(placed[1] - initial[1]) == pytest.approx(
            2 * np.linalg.norm(cell[axis]), rel=1e-12
        )
        np.testing.assert_array_equal(supplied, saved)
        assert not supplied.flags.writeable
        engine.set_positions(initial)
        assert engine.full_system is descriptor
        assert_physical_pair(engine, initial)
