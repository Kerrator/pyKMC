"""LAMMPS binding of the HTST prefactor plugin, as an :class:`EngineExtension`.

The three public methods become Manager operations automatically (extension
methods are collected by ``pykmc.manager.worker.build_registry`` through
``Engine.__dir__`` / ``Engine.__getattr__``), so ``manager.get_forces(...)``,
``manager.dynamical_matrix_eskm(...)`` and ``manager.compute_event_prefactors(...)``
work without touching the engine or manager modules. Every LAMMPS command is
collective on all ranks of the engine communicator; extraction returns on rank
0 only (``None`` elsewhere), the pyKMC engine convention.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np

from ..activevolume.active_volume import define_AV, make_AV, redefine_atoms, reset
from ..engine.base import Engine, EngineExtension
from ..rate_constant.prefactor import EventPrefactors
from ..rate_constant.prefactor import (
    compute_event_prefactors as _compute_event_prefactors,
)
from .constants import ESKM_DIV_EV_AMU_A2, thz_to_hz


def _prefactors_failure(reason: str) -> EventPrefactors:
    """Uniform graceful-fallback result (the op must never raise)."""
    return EventPrefactors(
        nu0_forward=None,
        nu0_backward=None,
        n_free=0,
        n_neg_saddle=-1,
        ok_forward=False,
        ok_backward=False,
        reason=reason,
    )


class HtstLammpsExtension(EngineExtension):
    """HTST engine operations on a :class:`pykmc.engine.lammps.LammpsEngine`.

    Uses only the engine's public surface: ``command``, ``rank``, ``comm``,
    ``lmp``, ``set_positions``, ``get_positions`` and ``minimize_freeze_core``.
    """

    def __init__(self, engine: Engine) -> None:
        super().__init__(engine)

    # ------------------------------------------------------------------
    # Forces and Hessian
    # ------------------------------------------------------------------

    def get_forces(self, positions: np.ndarray | None = None) -> np.ndarray | None:
        """Return the (N, 3) forces at ``positions`` (rank 0; ``None`` elsewhere)."""
        engine = self.engine
        if positions is not None:
            engine.set_positions(positions)
        engine.command("run 0")
        result = engine.lmp.gather_atoms("f", 1, 3)
        if engine.rank == 0:
            return np.ctypeslib.as_array(result).reshape(-1, 3)
        return None

    def _scratch_path(self) -> str:
        """Per-call scratch file for ``dynamical_matrix``, identical on every rank."""
        engine = self.engine
        path = None
        if engine.rank == 0:
            fd, path = tempfile.mkstemp(prefix="pykmc_dynmat.", suffix=".dat")
            os.close(fd)
        if engine.comm is not None:
            path = engine.comm.bcast(path, root=0)
        assert path is not None
        return path

    def dynamical_matrix_eskm(
        self,
        positions: np.ndarray,
        free_indices: np.ndarray | list[int] | None = None,
        dx: float = 1e-2,
    ) -> np.ndarray | None:
        """Mass-weighted Hessian at ``positions`` via LAMMPS ``dynamical_matrix eskm``.

        Computes the Hessian in-LAMMPS (a C++ finite difference per DOF) instead
        of the Python force loop. When ``free_indices`` is given, only those atoms
        vibrate (a LAMMPS group); the rest stay fixed at ``positions`` as the
        frozen boundary. Returns the Hessian in eV/(amu·Å²) on rank 0 (a drop-in
        for ``mass_weighted_partial_hessian``); ``None`` on other ranks.

        ``dx`` defaults to the FD step (0.01 Å) used by the validated FD path; a
        much smaller step shifts the soft saddle modes and inflates nu0.
        """
        engine = self.engine
        engine.set_positions(positions)
        engine.command("run 0")  # rebuild neighbour lists after the scatter
        if free_indices is not None:
            free = np.asarray(free_indices, dtype=int)
            ids = " ".join(str(i + 1) for i in free)  # 1-based LAMMPS ids
            engine.command(f"group g_dyn id {ids}")
            group, nat = "g_dyn", len(free)
        else:
            group, nat = "all", int(engine.lmp.get_natoms())
        tmp = self._scratch_path()
        engine.command(f"dynamical_matrix {group} eskm {dx} file {tmp}")
        if free_indices is not None:
            engine.command("group g_dyn delete")
        if engine.rank != 0:
            return None
        data = np.loadtxt(tmp)
        dim = 3 * nat
        hessian = np.empty((dim, dim))
        for i in range(dim):
            hessian[i] = data[i * nat : (i + 1) * nat].reshape(-1)
        hessian /= ESKM_DIV_EV_AMU_A2
        try:
            os.remove(tmp)
        except OSError:
            pass
        return 0.5 * (hessian + hessian.T)

    # ------------------------------------------------------------------
    # Pre-minimisation of the event surroundings
    # ------------------------------------------------------------------

    def _core_indices_within_rcut(
        self, positions: np.ndarray, central_atom_idx: int, rcut: float
    ) -> np.ndarray:
        """0-based indices of atoms within ``rcut`` of the central atom (minimum image).

        The frozen core is selected with the minimum-image convention from the
        engine's box, not with a LAMMPS ``region sphere``: regions are never
        wrapped across periodic boundaries, so a sphere overlapping a box edge
        silently drops the wrapped-side neighbours. Assumes an orthogonal box.
        The central atom is always included (distance 0), so the returned array
        is never empty.
        """
        import ase.geometry

        positions = np.asarray(positions, dtype=float)
        boxlo, boxhi, _xy, _yz, _xz, periodicity, _change = (
            self.engine.lmp.extract_box()
        )
        cell = np.diag(np.asarray(boxhi, dtype=float) - np.asarray(boxlo, dtype=float))
        pbc = [bool(p) for p in periodicity]
        diffs = positions - positions[central_atom_idx]
        _, dist = ase.geometry.find_mic(diffs, cell, pbc=pbc)
        return np.where(dist <= rcut)[0]

    def _premin_surroundings(
        self, config: Any, positions: np.ndarray, central_atom_idx: int
    ) -> np.ndarray:
        """Relax the surroundings of the event core before a Hessian.

        The core (atoms within ``atomicenvironment.rcut`` of the central atom)
        is frozen and the environment is minimized with the engine's own
        ``minimize_freeze_core`` (the ``partn_refine`` pattern). Freezing the
        core is what makes this safe at the saddle geometry. Returns the relaxed
        positions on every rank (broadcast from rank 0, so the following
        collective scatter is consistent).
        """
        engine = self.engine
        engine.set_positions(positions)
        engine.command("run 0")  # rebuild neighbour lists after the scatter
        core_idx = self._core_indices_within_rcut(
            positions, int(central_atom_idx), config.atomicenvironment.rcut
        )
        engine.minimize_freeze_core(core_idx)
        relaxed = engine.get_positions()
        if engine.comm is not None:
            relaxed = engine.comm.bcast(relaxed, root=0)
        assert relaxed is not None
        return relaxed

    # ------------------------------------------------------------------
    # Per-event Vineyard prefactor (the Manager operation)
    # ------------------------------------------------------------------

    def compute_event_prefactors(
        self,
        config: Any,
        central_atom_idx: int,
        min1_positions: np.ndarray,
        saddle_positions: np.ndarray,
        min2_positions: np.ndarray,
        types: list[str],
        cell: np.ndarray,
    ) -> EventPrefactors:
        """Per-event Vineyard nu0 on this engine: binds the eskm Hessian to the orchestrator.

        Runs entirely on this engine; concurrency comes from one job per event
        (see :class:`pykmc.htst.pool.EventPrefactorPool`). Never raises: returns
        an ``EventPrefactors`` whose ``nu0_*`` are ``None`` on any failure.

        Optional behaviours:

        - ``config.rateconstant.premin`` (default True): each geometry's
          surroundings are relaxed (event core within ``atomicenvironment.rcut``
          frozen) before its Hessian, via ``minimize_freeze_core``.
        - ``config.control.active_volume``: the engine is re-defined to the
          cropped AV subsystem around the central atom (buffer frozen, the
          ``partn_search_AV`` pattern); geometries/types/indices are remapped to
          AV-local before the Hessians. Requires
          ``activevolume.ract >= rateconstant.free_radius``.
        """
        engine = self.engine
        rc = config.rateconstant
        min1 = min1_positions
        sad = saddle_positions
        min2 = min2_positions
        central = int(central_atom_idx)
        types_used = list(types)

        try:
            if getattr(getattr(config, "control", None), "active_volume", False):
                ract = config.activevolume.ract
                if ract < rc.free_radius:
                    return _prefactors_failure(
                        f"active volume ract ({ract}) < free_radius ({rc.free_radius}); "
                        "the AV must contain the Hessian free region"
                    )
                # Mirror partn_search_AV: rebuild the engine as the AV subsystem.
                reset(engine, config, cell)
                av_positions, av_idx, buffer_idx = define_AV(
                    config, central, min1, cell
                )
                atom_map = np.array(av_idx, dtype=int)
                map_type = {
                    atom_type: {"ref": i + 1}
                    for i, atom_type in enumerate(sorted(set(types_used)))
                }
                int_types = np.array([map_type[t]["ref"] for t in types_used])
                redefine_atoms(engine, av_positions, int_types[atom_map])
                make_AV(engine, atom_map, buffer_idx)
                # Crop geometries and remap indices to the AV-local frame.
                min1 = min1[atom_map]
                sad = sad[atom_map]
                min2 = min2[atom_map]
                types_used = [types_used[i] for i in atom_map]
                central = int(np.where(atom_map == int(central_atom_idx))[0][0])

            if getattr(rc, "premin", True):
                min1 = self._premin_surroundings(config, min1, central)
                sad = self._premin_surroundings(config, sad, central)
                min2 = self._premin_surroundings(config, min2, central)
        except Exception as exc:  # AV setup / pre-min failure -> graceful k0 fallback
            return _prefactors_failure(f"{type(exc).__name__}: {exc}")

        def hessian_fn(positions: np.ndarray, free_indices: np.ndarray) -> np.ndarray:
            # Use the configured finite-difference step so the dynamical_matrix
            # Hessian matches the validated FD path; a too-small step shifts the
            # soft saddle modes and inflates nu0 (see HTST cross-validation).
            hessian = self.dynamical_matrix_eskm(
                positions, free_indices=free_indices, dx=rc.fd_step
            )
            if hessian is None:  # non-root rank: the orchestrator falls back
                raise RuntimeError(
                    "dynamical_matrix_eskm returns the Hessian on rank 0 only"
                )
            return hessian

        # pyKMC production systems are fully periodic; the free-region selector
        # only needs box lengths from `cell`.
        pbc = np.array([True, True, True])
        return _compute_event_prefactors(
            forces_fn=None,
            hessian_fn=hessian_fn,
            min1=min1,
            saddle=sad,
            min2=min2,
            types=types_used,
            central_index=central,
            free_radius=rc.free_radius,
            fd_step=rc.fd_step,
            cell=cell,
            pbc=pbc,
            nu0_min_hz=thz_to_hz(rc.nu0_min_THz),
            nu0_max_hz=thz_to_hz(rc.nu0_max_THz),
            require_one_negative_mode=rc.require_one_negative_mode,
        )
