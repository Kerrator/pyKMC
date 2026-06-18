"""Regression test: the HTST frozen-core selection must respect periodic boundaries.

The frozen core is every atom within ``rcut`` of the central atom under the
minimum-image convention. A LAMMPS ``region ... sphere`` does NOT wrap across
periodic boundaries (regions are never wrapped, per the LAMMPS ``region``
docs), so a sphere centred near a box corner is clipped at the box edges and
the wrapped-side neighbours are silently left out of the frozen group — they
relax when they must stay fixed.

Since PR #87 ``minimize_freeze_core`` freezes an explicit list of atom ids
rather than a geometric region; the minimum-image selection now lives in
``_core_indices_within_rcut`` (used by ``_premin_surroundings``). This test
builds a small fully periodic LJ fcc crystal, picks the atom at the box corner
(its ``rcut`` sphere wraps in all three directions), rattles every atom so
everything carries forces, then asserts (1) the selection helper returns the
minimum-image core that a naive sphere would miss, and (2) freezing that core
keeps every core atom fixed while the far field relaxes.
"""

from typing import Any

import numpy as np
import pytest

pytest.importorskip("lammps")

import ase.geometry

import pykmc.enginemanager.lmpi.lammps_operations as ops

_LATTICE = 3.6
_NCELL = 3
_BOX = _LATTICE * _NCELL
_RCUT = 4.0


class _SerialEngine:
    """Minimal serial stand-in for the MPI engine wrapper."""

    def __init__(self, lmp: Any) -> None:
        self.lmp = lmp
        self.rank = 0

    def command(self, cmd: str) -> None:
        """Forward a command to the raw LAMMPS instance."""
        self.lmp.command(cmd)


class _Lammps:
    """LAMMPS minimize-option shim that ``minimize_freeze_core`` reads."""

    min_style = "cg"
    frz_min = "1e-6 1e-8 20 20"


class _Cfg:
    """Config shim exposing only the minimize options the op needs."""

    lammps = _Lammps()


def _build_periodic_fcc_engine() -> _SerialEngine:
    """Create a fully periodic 108-atom LJ fcc crystal in a serial engine."""
    from lammps import lammps

    lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
    lmp.command("units metal")
    lmp.command("atom_style atomic")
    lmp.command("atom_modify map array")
    lmp.command("boundary p p p")
    lmp.command(f"lattice fcc {_LATTICE}")
    lmp.command(f"region box block 0 {_NCELL} 0 {_NCELL} 0 {_NCELL}")
    lmp.command("create_box 1 box")
    lmp.command("create_atoms 1 box")
    lmp.command("mass 1 58.693")
    lmp.command("pair_style lj/cut 5.0")
    lmp.command("pair_coeff * * 0.4 2.3")
    lmp.command("run 0")
    return _SerialEngine(lmp)


def _mic_distances(positions: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Minimum-image distances from ``reference`` to every position."""
    cell = np.diag([_BOX, _BOX, _BOX])
    _, dist = ase.geometry.find_mic(positions - reference, cell, pbc=True)
    return dist


def test_freeze_core_spans_periodic_boundaries() -> None:
    """Atoms within rcut across a periodic boundary must stay frozen."""
    engine = _build_periodic_fcc_engine()
    pristine = ops.get_positions(engine)

    # Central atom at the box corner: its rcut sphere wraps in x, y and z.
    central = int(np.argmin(np.linalg.norm(pristine, axis=1)))

    # Rattle everything (seeded) so every atom carries a force.
    rng = np.random.default_rng(42)
    rattled = pristine + rng.normal(0.0, 0.05, size=pristine.shape)
    ops.set_positions(engine, rattled)
    engine.command("run 0")

    # Expected frozen core under the minimum-image convention.
    dist_mic = _mic_distances(rattled, rattled[central])
    core = dist_mic <= _RCUT

    # Scenario sanity: the naive (unwrapped) sphere must MISS part of the
    # core, otherwise this test would not exercise boundary wrapping.
    dist_naive = np.linalg.norm(rattled - rattled[central], axis=1)
    naive_core = dist_naive <= _RCUT
    assert core.sum() > naive_core.sum(), (
        "test geometry does not span a periodic boundary"
    )

    # The selection helper must recover exactly the minimum-image core — i.e.
    # the wrapped-side neighbours the naive sphere drops.
    core_idx = ops._core_indices_within_rcut(engine, rattled, central, _RCUT)
    assert set(core_idx.tolist()) == set(np.where(core)[0].tolist()), (
        "_core_indices_within_rcut did not select the minimum-image core"
    )

    # Freezing that core and minimizing must hold every core atom fixed.
    ops.minimize_freeze_core(engine, _Cfg(), core_idx)
    after = ops.get_positions(engine)

    # Compare via minimum-image displacement so LAMMPS re-wrapping of
    # coordinates across the boundary does not register as motion.
    cell = np.diag([_BOX, _BOX, _BOX])
    _, moved = ase.geometry.find_mic(after - rattled, cell, pbc=True)

    # Every minimum-image core atom must have stayed exactly frozen.
    frozen_violations = np.where(core & (moved > 1e-8))[0]
    assert frozen_violations.size == 0, (
        "core atoms moved during freeze-core minimization "
        f"(indices {frozen_violations.tolist()}, "
        f"max displacement {moved[core].max():.3e} A) — the frozen region "
        "did not wrap across the periodic boundary"
    )

    # The minimization itself did real work on the far field.
    assert moved[~core].max() > 1e-6
