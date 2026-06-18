"""Timing harness for the two HTST Hessian modes (eskm vs Python finite-difference).

Both modes feed the SAME numpy ``eigh``/Vineyard math; only the Hessian assembly
differs:

- **eskm** -- one LAMMPS ``dynamical_matrix <group> eskm <dx> file <tmp>`` command
  per Hessian (a C++ finite difference over all free DOF), then a file read.
  Requires the LAMMPS PHONON package.
- **fd** -- :func:`pykmc.htst.hessian.mass_weighted_partial_hessian`: ``2 * 3F``
  ``get_forces`` calls per Hessian (no file I/O), assembled in Python.

This module is import-safe without a LAMMPS build (the ``lammps`` / engine-ops
imports are deferred inside the functions that need them, mirroring
:mod:`pykmc.htst.enrich`). It instruments COPIES of the production code paths so
the live functions in ``lammps_operations`` stay untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional

import numpy as np

from pykmc.config import PhysicalConstants
from pykmc.htst.constants import hz_to_thz
from pykmc.htst.free_region import select_free_indices
from pykmc.htst.hessian import mass_weighted_partial_hessian
from pykmc.htst.vineyard import vineyard_prefactor

# Frozen-boundary partial Hessian -> no translational zero modes (see
# prefactor.compute_event_prefactors). Both modes must use the same value.
_N_ZERO_MODES = 0


class PhononUnavailable(RuntimeError):
    """Raised when the LAMMPS ``dynamical_matrix`` command (PHONON package) is absent."""


@dataclass
class HessianTiming:
    """Wall-clock breakdown of assembling one mass-weighted Hessian."""

    t_hessian: float
    # FD-only fields:
    t_forces_total: Optional[float] = None
    n_calls: Optional[int] = None
    t_forces_mean: Optional[float] = None
    # eskm-only fields:
    t_set_run0: Optional[float] = None
    t_dynmat_cmd: Optional[float] = None
    t_file_read: Optional[float] = None


@dataclass
class Row:
    """One CSV record: timing + nu0 for a single (mode, event, radius, Hessian)."""

    mode: str
    system: str
    event: int
    free_radius: float
    n_free: int
    hessian_idx: int
    t_hessian: float
    t_forces_total: Optional[float]
    n_calls: Optional[int]
    t_forces_mean: Optional[float]
    t_set_run0: Optional[float]
    t_dynmat_cmd: Optional[float]
    t_file_read: Optional[float]
    t_eigh: float
    nu0_hz: Optional[float]
    nu0_thz: Optional[float]
    round_trips: int


def phonon_available(engine: object) -> bool:
    """Return True if the engine's LAMMPS build has the PHONON package."""
    lmp = engine.lmp  # type: ignore[attr-defined]
    return bool(lmp.has_package("PHONON"))


def build_serial_engine(
    positions: np.ndarray,
    potential: str,
    pair_style: str = "eam/alloy",
    element: str = "Ni",
) -> "tuple[object, np.ndarray]":
    """Create a serial in-memory LAMMPS engine holding ``positions`` as a cluster.

    Mirrors the box setup in :func:`pykmc.htst.enrich.lammps_forces_factory` and
    ``tests/htst/test_engine_prefactors._build_eam_engine`` (non-periodic
    ``boundary f f f``, single-type ``element``), but returns the engine itself so
    BOTH the eskm and FD paths can run on the identical instance. ``lammps`` and
    engine ops are imported lazily.

    Parameters
    ----------
    positions : np.ndarray
        (N, 3) atom positions defining the cluster.
    potential : str
        LAMMPS potential file path.
    pair_style : str
        LAMMPS pair_style (default ``eam/alloy``).
    element : str
        Single element symbol (all atoms are this species; v1 single-element).

    Returns
    -------
    tuple[object, np.ndarray]
        ``(engine, cell)`` -- the serial engine shim and its (3, 3) box.

    """
    from ase.data import atomic_masses, atomic_numbers

    from lammps import lammps

    mass = float(atomic_masses[atomic_numbers[element]])

    class _SerialEngine:
        def __init__(self, lmp: object) -> None:
            self.lmp = lmp
            self.rank = 0

        def command(self, cmd: str) -> None:
            self.lmp.command(cmd)

    lo = positions.min(axis=0) - 15.0
    hi = positions.max(axis=0) + 15.0
    lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
    lmp.command("units metal")
    lmp.command("atom_style atomic")
    lmp.command("atom_modify map array")
    lmp.command("boundary f f f")
    bounds = " ".join(f"{v:.3f}" for v in (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
    lmp.command(f"region box block {bounds} units box")
    lmp.command("create_box 1 box")
    n_atoms = positions.shape[0]
    lmp.create_atoms(
        n_atoms, None, [1] * n_atoms, positions.astype(float).reshape(-1).tolist()
    )
    lmp.command(f"mass 1 {mass}")
    lmp.command(f"pair_style {pair_style}")
    lmp.command(f"pair_coeff * * {potential} {element}")
    lmp.command("run 0")
    return _SerialEngine(lmp), np.diag(hi - lo)


def time_hessian_fd(
    engine: object,
    positions: np.ndarray,
    masses: np.ndarray,
    free_indices: np.ndarray,
    dx: float,
) -> "tuple[np.ndarray, HessianTiming]":
    """Build the FD mass-weighted Hessian, timing and counting force calls.

    Wraps ``ops.get_forces`` in a counting/timing shim, then delegates the
    assembly to the production :func:`mass_weighted_partial_hessian`.
    """
    from ..enginemanager.lmpi import lammps_operations as ops

    n_calls = 0
    t_forces_total = 0.0

    def forces_fn(pos: np.ndarray) -> np.ndarray:
        nonlocal n_calls, t_forces_total
        t0 = perf_counter()
        f = ops.get_forces(engine, pos)
        t_forces_total += perf_counter() - t0
        n_calls += 1
        return f

    t0 = perf_counter()
    hessian = mass_weighted_partial_hessian(forces_fn, positions, masses, free_indices, dx)
    t_hessian = perf_counter() - t0

    timing = HessianTiming(
        t_hessian=t_hessian,
        t_forces_total=t_forces_total,
        n_calls=n_calls,
        t_forces_mean=(t_forces_total / n_calls) if n_calls else None,
    )
    return hessian, timing


def time_hessian_eskm(
    engine: object,
    positions: np.ndarray,
    free_indices: np.ndarray,
    dx: float,
) -> "tuple[np.ndarray, HessianTiming]":
    """Build the eskm Hessian via LAMMPS ``dynamical_matrix``, timing each segment.

    Instrumented copy of
    :func:`pykmc.enginemanager.lmpi.lammps_operations.dynamical_matrix_eskm`
    (production function left untouched). Segments timed: (1) scatter + ``run 0``
    + group create, (2) the ``dynamical_matrix ... eskm ... file`` command,
    (3) file read + scale + symmetrize. Raises :class:`PhononUnavailable` if the
    command fails (e.g. PHONON package absent).
    """
    import os

    from ..enginemanager.lmpi.lammps_operations import set_positions

    free = np.asarray(free_indices, dtype=int)

    t0 = perf_counter()
    set_positions(engine, positions)
    engine.command("run 0")  # type: ignore[attr-defined]
    ids = " ".join(str(i + 1) for i in free)
    engine.command(f"group g_dyn id {ids}")  # type: ignore[attr-defined]
    nat = len(free)
    t_set_run0 = perf_counter() - t0

    tmp = f"/tmp/pykmc_dynmat_profile.{getattr(engine, 'engine_id', 0)}.dat"
    t1 = perf_counter()
    try:
        engine.command(f"dynamical_matrix g_dyn eskm {dx} file {tmp}")  # type: ignore[attr-defined]
    except Exception as exc:  # PHONON missing surfaces as a LAMMPS command error
        engine.command("group g_dyn delete")  # type: ignore[attr-defined]
        raise PhononUnavailable(
            f"LAMMPS 'dynamical_matrix' command failed ({exc}); PHONON package "
            "required for the eskm mode."
        ) from exc
    t_dynmat_cmd = perf_counter() - t1
    engine.command("group g_dyn delete")  # type: ignore[attr-defined]

    t2 = perf_counter()
    data = np.loadtxt(tmp)
    dim = 3 * nat
    hessian = np.empty((dim, dim))
    for i in range(dim):
        hessian[i] = data[i * nat:(i + 1) * nat].reshape(-1)
    hessian /= PhysicalConstants.eskm_div_eV_amu_A2
    hessian = 0.5 * (hessian + hessian.T)
    try:
        os.remove(tmp)
    except OSError:
        pass
    t_file_read = perf_counter() - t2

    t_hessian = t_set_run0 + t_dynmat_cmd + t_file_read
    timing = HessianTiming(
        t_hessian=t_hessian,
        t_set_run0=t_set_run0,
        t_dynmat_cmd=t_dynmat_cmd,
        t_file_read=t_file_read,
    )
    return hessian, timing


def _masses_for(types: list[str]) -> np.ndarray:
    """Return per-atom masses (amu) from ASE atomic data."""
    from ase.data import atomic_masses, atomic_numbers

    return np.array([atomic_masses[atomic_numbers[t]] for t in types], dtype=float)


def time_event(
    engine: object,
    mode: str,
    min1: np.ndarray,
    saddle: np.ndarray,
    min2: np.ndarray,
    types: list[str],
    central_index: int,
    free_radius: float,
    fd_step: float,
    cell: np.ndarray,
    pbc: np.ndarray,
    system: str = "",
    event: int = 0,
) -> list[Row]:
    """Time all three Hessians (min1, saddle, min2) for one event in one mode.

    Returns one :class:`Row` per Hessian. The forward Vineyard nu0
    (min1 vs saddle) is computed and its ``eigh``/Vineyard cost is recorded as
    ``t_eigh`` on the min1 row; saddle/min2 rows carry ``t_eigh=0`` and the
    forward ``nu0`` is repeated on each row for convenience.

    Parameters mirror :func:`pykmc.rate_constant.prefactor.compute_event_prefactors`
    (same ``select_free_indices`` free region, same ``n_zero_modes=0``).
    """
    if mode not in ("fd", "eskm"):
        raise ValueError(f"mode must be 'fd' or 'eskm', got {mode!r}")

    free = select_free_indices(min1, central_index, free_radius, cell, pbc)
    n_free = len(free)
    masses = _masses_for(types)
    geoms = (min1, saddle, min2)

    hessians: list[np.ndarray] = []
    timings: list[HessianTiming] = []
    for geom in geoms:
        if mode == "fd":
            h, t = time_hessian_fd(engine, geom, masses, free, fd_step)
        else:
            h, t = time_hessian_eskm(engine, geom, free, fd_step)
        hessians.append(h)
        timings.append(t)

    # Shared diagonalization cost: forward nu0 (min1 vs saddle).
    t0 = perf_counter()
    try:
        nu0_hz: Optional[float] = vineyard_prefactor(
            hessians[0], hessians[1], n_zero_modes=_N_ZERO_MODES
        )
    except Exception:
        nu0_hz = None
    t_eigh = perf_counter() - t0
    nu0_thz = hz_to_thz(nu0_hz) if nu0_hz is not None else None

    rows: list[Row] = []
    for idx, t in enumerate(timings):
        rows.append(
            Row(
                mode=mode,
                system=system,
                event=event,
                free_radius=free_radius,
                n_free=n_free,
                hessian_idx=idx,
                t_hessian=t.t_hessian,
                t_forces_total=t.t_forces_total,
                n_calls=t.n_calls,
                t_forces_mean=t.t_forces_mean,
                t_set_run0=t.t_set_run0,
                t_dynmat_cmd=t.t_dynmat_cmd,
                t_file_read=t.t_file_read,
                t_eigh=t_eigh if idx == 0 else 0.0,
                nu0_hz=nu0_hz,
                nu0_thz=nu0_thz,
                round_trips=(t.n_calls if mode == "fd" else 1) or 0,
            )
        )
    return rows
