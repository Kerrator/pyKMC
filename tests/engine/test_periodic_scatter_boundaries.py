"""Periodic faces and reduced AV crops preserve native pair physics.

Imports shared independent LJ setup/oracles from the promoted regression file.
No native work occurs on import.
"""

from types import SimpleNamespace

import numpy as np
from ase.cell import Cell
from mpi4py import MPI

from pykmc.activevolume import active_volume
from pykmc.engine.lammps import LammpsEngine
from tests.engine.test_periodic_scatter import (
    analytic_pair,
    assert_physical_pair,
    geometry,
    initialized,
    rotation,
)


def test_periodic_face_and_near_faces_keep_exact_pair_physics():
    with initialized("rotated_mixed") as (engine, cell, pbc, initial):
        turn = rotation()
        images = np.array([[3, 0, 4], [-4, 0, -3]], dtype=float)
        # Zero is an exact lattice-face construction; +/-2^-40 probes both
        # sides without requiring a particular Cartesian image at the face.
        for face_offset in (0.0, -(2.0**-40), 2.0**-40):
            first = np.array([face_offset, 0.31, 0.42]) @ cell
            physical = np.array([first, first + np.array([1.3, 0.0, 0.0]) @ turn])
            request = physical + images @ cell
            request.setflags(write=False)
            saved = request.copy()
            descriptor = engine.full_system
            engine.set_positions(request)
            np.testing.assert_array_equal(request, saved)
            assert engine.full_system is descriptor
            actual = engine.get_positions()
            # Only periodic integer lattice shifts may differ. Analytic
            # forces/energy below are independent of this equivalence check.
            difference = np.linalg.solve(cell.T, (actual - physical).T).T
            periodic = np.asarray(pbc, dtype=bool)
            np.testing.assert_allclose(
                difference[:, periodic],
                np.rint(difference[:, periodic]),
                rtol=0,
                atol=2e-12,
            )
            np.testing.assert_allclose(
                difference[:, ~periodic], 0.0, rtol=0, atol=2e-12
            )
            assert_physical_pair(engine, physical)


def test_actual_reduced_av_crop_scatter_and_full_restore():
    cell, pbc, pair = geometry("ortho_mixed")
    positions = np.vstack([pair, [9.0, 8.0, 8.0]])
    types = ("Ni", "Fe", "Ni")
    native = SimpleNamespace(
        pair_style="lj/cut 2.5",
        pair_coeff="* * 1.0 1.0",
        min_style="cg",
        minimize="1e-8 1e-8 100 1000",
        frz_min="1e-8 1e-8 100 1000",
        verbosity=0,
    )
    config = SimpleNamespace(
        lammps=native, activevolume=SimpleNamespace(ract=2.0, rmov=1.6, AV_debug=False)
    )
    engine = LammpsEngine(native, comm=MPI.COMM_SELF, engine_id=9033)
    try:
        engine.start()
        engine.initialize_parameters()
        engine.initialize_system(
            types=types,
            positions=positions.copy(),
            cell=Cell(cell),
            pbc=pbc,
            species=("Fe", "Ni"),
            masses=(56.0, 60.0),
        )
        engine.initialize_potential()
        full = engine.full_system
        energy, forces = analytic_pair(pair)
        np.testing.assert_allclose(
            engine.get_total_energy(), energy, rtol=0, atol=2e-11
        )
        atom_map, center = active_volume.partn_search_AV(
            engine, config, 0, positions.copy(), cell.copy(), types
        )
        np.testing.assert_array_equal(atom_map, [0, 1])
        np.testing.assert_array_equal(center, [1])
        assert int(engine.lmp.get_natoms()) == 2
        assert full.natoms == 3
        assert engine.full_system is full
        assert engine.system_is_cropped
        images = np.array([[3, 0, 4], [-4, 0, -3]], dtype=float)
        request = pair + images @ cell
        request.setflags(write=False)
        saved = request.copy()
        active_volume.set_positions(engine, request)
        np.testing.assert_array_equal(request, saved)
        np.testing.assert_allclose(engine.get_positions(), pair, rtol=0, atol=2e-11)
        assert_physical_pair(engine, pair)
        assert engine.full_system is full
        assert engine.system_is_cropped
        assert engine.ensure_full_system(positions) is True
        assert not engine.system_is_cropped
        assert int(engine.lmp.get_natoms()) == 3
        assert engine.full_system.physics == full.physics
        assert engine.full_system.species == full.species
        assert engine.full_system.masses == full.masses
        np.testing.assert_allclose(
            engine.get_positions(), positions, rtol=0, atol=2e-11
        )
        np.testing.assert_allclose(
            engine.get_total_energy(), energy, rtol=0, atol=2e-11
        )
        np.testing.assert_allclose(
            engine.get_forces(),
            np.vstack([forces, [0.0, 0.0, 0.0]]),
            rtol=0,
            atol=2e-10,
        )
    finally:
        engine.close()
        assert engine.lmp is None
