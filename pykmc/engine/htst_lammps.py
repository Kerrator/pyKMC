"""LAMMPS-backed HTST prefactors: ``dynamical_matrix ... eskm`` on a scratch engine.

:class:`LammpsHTSTExtension` exposes one typed per-event operation,
``compute_event_prefactors(request)``, and one capability check,
``htst_preflight()``, on a :class:`~pykmc.engine.lammps.LammpsEngine`. Both are
dispatched on every rank of the worker's local communicator; only local rank 0
does any work, on a private serial scratch ``LammpsEngine`` over
``MPI.COMM_SELF`` that is created and closed inside the call. The search engine
the extension is attached to is never touched: no command, no collective, no
position change. This is parallelism across events (one event per worker root),
not a multi-rank Hessian.

Per call the scratch instance is built from the request (types, ``min1``
geometry, cell, pbc, the full ``species``/``masses`` map, the engine's
potential), the mass-weighted partial Hessian of the free atoms is obtained with
LAMMPS's own central differences (``dynamical_matrix <group> eskm <fd_step>``)
and handed to :func:`pykmc.htst.compute_event_prefactors`, which owns the
scientific verdict. Files go to a ``tempfile.mkdtemp`` directory removed in
``finally``; the group is deleted in ``finally``; the scratch engine is closed in
``finally``. There is no finite-difference production fallback: a LAMMPS build
without PHONON fails ``htst_preflight`` and every ``compute_event_prefactors``.

Free region and zone crop
-------------------------
The one common free set of the three Hessians is the sphere of
``settings.free_radius`` around the moving atom in the geometry named by
``settings.free_region_center``: the saddle by default (symmetric between the
two minima by construction), or ``min1`` for the original model. With
``settings.zone_radius`` set, the scratch instance holds only the atoms within
``zone_radius`` of the centre, selected on that same centring geometry (minimum
image in the full cell with the request's pbc), so the free set is a subset of
the zone by construction (``free_radius < zone_radius`` on one geometry; the
remap is checked, never assumed); the ``free_radius``..``zone_radius`` shell is
the frozen boundary and everything beyond it is absent. The shell must
therefore exceed the potential's interaction range or the outermost free atoms
see a truncated environment: every energy term coupling two free atoms must have
all of its atoms present, which for a three-body potential can reach twice the
cutoff. On the SW-Si fixture ``zone_radius=10`` with ``free_radius=6`` (a 4 Å
shell, cutoff 3.77 Å) reproduces the full system to 1e-12 relative; that margin
is a per-potential choice, not a general guarantee. The crop keeps the full
species map so integer types keep their meaning.

Only the forward direction can be requested (``compute_backward=False``): the
``min2`` Hessian is then never computed and the backward result is
``status="skipped"``; site requests use this.

Premin
------
With ``settings.premin`` the surroundings of each geometry are relaxed with the
event core and the source's fixed atoms frozen, using ``config.frz_min`` through
``LammpsEngine.minimize_freeze_core``, before that geometry's Hessian. The
relaxation always runs on the full system, before any crop, so that no shell atom
is relaxed against vacuum. Core positions are unchanged by construction (their
forces are zeroed), so the free set and the event identity are preserved.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import replace
from typing import Any

import numpy as np
from mpi4py import MPI

from ..htst import (
    ESKM_METAL_CONVERSION,
    EventPrefactors,
    HessianFn,
    HTSTEventRequest,
    HTSTRequestError,
    PrefactorRejected,
    PrefactorRejection,
    select_free_indices,
)
from ..htst import compute_event_prefactors as _kernel_compute_event_prefactors
from ..htst.free_region import common_free_indices
from ..htst.provenance import CalculationProvenance
from .base import EngineExtension
from .lammps import LammpsEngine
from ..physics import EnginePhysics, ForceModel

logger = logging.getLogger("log")

TMPDIR_PREFIX: str = "pykmc_htst_"
"""Prefix of the per-call ``tempfile.mkdtemp`` directory (tests count these)."""

_DYNMAT_FILENAME = "dynmat.dat"
_FREE_GROUP = "htst_free"
_SCRATCH_ID_STRIDE = 1_000_000
_PREFLIGHT_BOX = 40.0
_PREFLIGHT_SPACING = 3.0


class _QuietConfig:
    """Proxy the engine's LAMMPS config with log output disabled.

    Scratch engines are created once per event operation; inheriting the
    production ``verbosity`` would write one ``lammps.log.<id>`` file per
    Hessian call. Every other attribute is forwarded to the wrapped config.
    """

    verbosity: int = 0

    def __init__(self, inner: object) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class LammpsHTSTExtension(EngineExtension):
    """Per-event HTST prefactors for a ``LammpsEngine`` (see the module docstring).

    Parameters
    ----------
    engine : LammpsEngine
        The search engine to attach to. Only its ``config`` (potential and
        minimiser settings), ``comm`` (to find the local root), ``engine_id``
        (scratch id derivation) and ``full_system`` (preflight species map) are
        read; nothing on it is ever called or mutated.

    Raises
    ------
    TypeError
        If ``engine`` is not a ``LammpsEngine``: the operation needs the
        concrete LAMMPS ``dynamical_matrix`` capability, and the abstract
        ``Engine`` contract is not extended with HTST-only methods.

    """

    def __init__(self, engine: LammpsEngine) -> None:
        if not isinstance(engine, LammpsEngine):
            raise TypeError(
                "LammpsHTSTExtension requires a LammpsEngine (LAMMPS "
                f"dynamical_matrix), got {type(engine).__name__}"
            )
        super().__init__(engine)
        self._scratch_calls = 0

    # ------------------------------------------------------------------
    # Public operations (the only two; everything else is private)
    # ------------------------------------------------------------------

    def htst_preflight(self) -> dict[str, Any] | None:
        """Check that this build and potential can produce eskm Hessians.

        On the local root a scratch ``LammpsEngine`` is started, the PHONON
        package is required, and the search engine's potential is initialised
        on a tiny one-atom-per-species system built with the full
        ``full_system`` species/mass map (``run 0`` included, so a missing
        per-type mass or a bad ``pair_coeff`` surfaces here).

        Returns
        -------
        dict | None
            ``{"phonon": True, "lammps_version": int, "pair_style": str,
            "species": tuple, "masses": tuple}`` on the local root; ``None`` on
            every other rank.

        Raises
        ------
        RuntimeError
            If the LAMMPS build lacks PHONON, if the search engine has no
            initialised system yet, or if the potential cannot be initialised
            in a scratch instance (the LAMMPS error is chained).

        """
        if not self._is_root():
            return None
        full_system = self.engine.full_system
        if full_system is None:
            raise RuntimeError(
                "htst_preflight: the search engine has no initialised system; "
                "call initialize_system and initialize_potential first"
            )
        physics = full_system.physics
        if physics is None:
            raise RuntimeError("htst_preflight needs initialized engine physics")
        if ForceModel.capture(self.engine.config) != physics.force_model:
            raise RuntimeError(
                "force-model contents changed since engine initialization"
            )
        scratch = self._new_scratch()
        try:
            scratch.start()
            if not scratch.lmp.has_package("PHONON"):
                raise RuntimeError(
                    "HTST prefactors need a LAMMPS build with the PHONON package "
                    "(dynamical_matrix); this build lacks it and there is no "
                    "finite-difference production fallback"
                )
            n_species = len(full_system.species)
            positions = np.full((n_species, 3), _PREFLIGHT_SPACING, dtype=float)
            positions[:, 0] += _PREFLIGHT_SPACING * np.arange(n_species)
            try:
                self._build_scratch(
                    scratch,
                    types=full_system.species,
                    positions=positions,
                    cell=np.diag([_PREFLIGHT_BOX] * 3),
                    pbc=(True, True, True),
                    species=full_system.species,
                    masses=full_system.masses,
                    expected_physics=physics,
                )
                scratch.lmp.command("run 0 post no")
            except Exception as exc:
                raise RuntimeError(
                    "htst_preflight: the potential could not be initialised in a "
                    f"scratch LAMMPS instance (pair_style {self.engine.config.pair_style!r}, "
                    f"species {full_system.species}): {exc}"
                ) from exc
            return {
                "phonon": True,
                "lammps_version": int(scratch.lmp.version()),
                "pair_style": str(self.engine.config.pair_style),
                "species": full_system.species,
                "masses": full_system.masses,
                "engine_physics": physics,
            }
        finally:
            scratch.close()

    def compute_event_prefactors(
        self,
        request: HTSTEventRequest,
        *,
        compute_backward: bool = True,
        compute_energies: bool = False,
    ) -> EventPrefactors | None:
        """Compute the forward and backward Vineyard prefactors of one event.

        Parameters
        ----------
        request : HTSTEventRequest
            Full-system geometries, types, the full species/mass map, cell, pbc,
            centre index and settings. Validated before any engine work.
        compute_backward : bool, optional
            ``False`` skips the ``min2`` Hessian; the backward direction of the
            result is then ``status="skipped"`` (site requests).

        Returns
        -------
        EventPrefactors | None
            The typed result (``method == "lammps_eskm"``) on the local root;
            ``None`` on every other rank of the local communicator.

        Raises
        ------
        HTSTRequestError
            If the request fails validation or ``zone_radius <= free_radius``.
        RuntimeError
            If the ``dynamical_matrix`` output cannot be parsed.
        Exception
            Any LAMMPS error raised while building the scratch system, relaxing
            the surroundings or running ``dynamical_matrix``; the scratch engine
            and temporary directory are cleaned up first.

        Notes
        -----
        Scientific rejections (nonstationary geometry, unstable minimum, non-first-order saddle,
        non-finite Hessian, out-of-window prefactor) are returned as
        ``status="rejected"`` per direction by the kernel; they never raise.

        """
        if not self._is_root():
            return None
        if not isinstance(request, HTSTEventRequest):
            raise HTSTRequestError(
                f"request must be an HTSTEventRequest, got {type(request).__name__}"
            )
        request.validate()
        if not isinstance(compute_energies, bool):
            raise HTSTRequestError("compute_energies must be a bool")
        request = replace(request, user_constraints=request.resolved_user_constraints())
        if request.descriptor is not None:
            physics = request.descriptor.engine
            actual = ForceModel.capture(self.engine.config)
            if actual != physics.force_model:
                raise HTSTRequestError(
                    "request force model differs from scratch inputs: the "
                    f"request declares pair_style {' '.join(physics.force_model.style)!r} "
                    f"with pair_coeff {' '.join(physics.force_model.coefficients)!r}, "
                    f"the engine runs pair_style {' '.join(actual.style)!r} with "
                    f"pair_coeff {' '.join(actual.coefficients)!r}"
                    + (
                        f" (declared model limitation: {physics.force_model.limitation})"
                        if physics.force_model.limitation
                        else ""
                    )
                    + (
                        f" (engine model limitation: {actual.limitation})"
                        if actual.limitation
                        else ""
                    )
                )
            if request.settings.premin and physics.premin_solver != (
                str(self.engine.config.min_style),
                str(self.engine.config.frz_min),
            ):
                raise HTSTRequestError(
                    "request premin solver differs from scratch inputs"
                )
        settings = request.settings
        if settings.zone_radius is not None and not (
            settings.zone_radius > settings.free_radius
        ):
            raise HTSTRequestError(
                f"zone_radius ({settings.zone_radius}) must exceed free_radius "
                f"({settings.free_radius}): the free_radius..zone_radius shell is "
                "the frozen boundary of the scratch system and must itself be "
                "wider than the potential's full interaction range"
            )

        min1 = np.array(request.min1_positions, dtype=float)
        saddle = np.array(request.saddle_positions, dtype=float)
        min2 = np.array(request.min2_positions, dtype=float)
        cell = np.asarray(request.cell, dtype=float)
        center = int(request.center_index)
        # One common free set for the three Hessians, selected in the centring
        # geometry (saddle by default, min1 for the original model). The
        # vibrational set excludes fixed rows; premin instead locks the union
        # of the core and fixed rows. The zone uses the same geometry so
        # that the free set is a subset of the zone (checked below).
        centring = saddle if settings.free_region_center == "saddle" else min1
        core_global = select_free_indices(
            centring, center, settings.free_radius, cell, request.pbc
        )
        free_global = common_free_indices(request, core_global)
        fixed_global = (
            ()
            if request.constraints is None
            else request.constraints.local_fixed_indices
        )
        # The free set is the sphere minus the request's fixed rows: the
        # user-frozen atoms and, under active volume, the shell held during
        # the search. The service reports the exclusion per request on rank 0
        # (``free_set_report``); worker ranks have no INFO handler.
        relaxation_locks = np.union1d(core_global, fixed_global).astype(int)
        zone: np.ndarray | None = None
        if settings.zone_radius is not None:
            zone = select_free_indices(
                centring, center, settings.zone_radius, cell, request.pbc
            )
        if free_global.size == 0:
            # The kernel owns the unavailable/skipped result. No scratch engine
            # or matrix is needed when constraints leave no vibrational DOFs.
            # The provenance still records the zone these settings select, so
            # a later context comparison sees the same crop a computed result
            # would carry.
            empty = _kernel_compute_event_prefactors(
                request,
                lambda positions, indices: np.empty((0, 0)),
                method="lammps_eskm",
                free_indices=free_global,
                compute_backward=compute_backward,
            )
            provenance = CalculationProvenance.capture(
                request,
                request,
                method="lammps_eskm",
                free_indices=free_global,
                zone_indices=zone,
            )
            return replace(empty, provenance=provenance)

        geometries = [min1, saddle, min2]
        scratch = self._new_scratch()
        try:
            scratch.start()
            full_built = False
            if settings.premin or compute_energies:
                self._build_scratch(
                    scratch,
                    types=request.types,
                    positions=geometries[0],
                    cell=cell,
                    pbc=request.pbc,
                    species=request.species,
                    masses=request.masses,
                    expected_physics=None
                    if request.descriptor is None
                    else request.descriptor.engine,
                    check_premin=settings.premin,
                )
                full_built = True
            if settings.premin:
                geometries = [
                    self._premin(scratch, geometry, relaxation_locks)
                    for geometry in geometries
                ]
                if request.constraints is not None:
                    for geometry in geometries:
                        request.constraints.validate_positions(
                            geometry, cell=cell, pbc=request.pbc
                        )
            # Persist full post-premin geometry before any zone crop. The
            # kernel's local request cannot describe missing source neighbors.
            produced_request = replace(
                request,
                min1_positions=geometries[0],
                saddle_positions=geometries[1],
                min2_positions=geometries[2],
            )
            provenance = CalculationProvenance.capture(
                request,
                produced_request,
                method="lammps_eskm",
                free_indices=free_global,
                zone_indices=zone,
                energies=(
                    tuple(
                        scratch.get_potential_energy(positions=geometry)
                        for geometry in geometries
                    )
                    if compute_energies
                    else None
                ),
            )
            if zone is None:
                if not full_built:
                    self._build_scratch(
                        scratch,
                        types=request.types,
                        positions=geometries[0],
                        cell=cell,
                        pbc=request.pbc,
                        species=request.species,
                        masses=request.masses,
                        expected_physics=None
                        if request.descriptor is None
                        else request.descriptor.engine,
                        check_premin=settings.premin,
                    )
                local_request = replace(
                    request,
                    min1_positions=geometries[0],
                    saddle_positions=geometries[1],
                    min2_positions=geometries[2],
                )
                free_local = free_global
            else:
                if full_built:
                    scratch.lmp.command("clear")
                crop_types = tuple(request.types[i] for i in zone)
                cropped = [geometry[zone] for geometry in geometries]
                self._build_scratch(
                    scratch,
                    types=crop_types,
                    positions=cropped[0],
                    cell=cell,
                    pbc=request.pbc,
                    species=request.species,
                    masses=request.masses,
                    expected_physics=None
                    if request.descriptor is None
                    else request.descriptor.engine,
                    check_premin=settings.premin,
                )
                # free_radius < zone_radius on the same geometry, so every free
                # atom is in the zone and the remap is exact.
                free_local = np.searchsorted(zone, free_global)
                if not np.array_equal(zone[free_local], free_global):
                    raise RuntimeError(
                        "zone crop does not contain every free atom; this is a "
                        "plumbing error in the crop selection"
                    )
                local_request = replace(
                    request,
                    min1_positions=cropped[0],
                    saddle_positions=cropped[1],
                    min2_positions=cropped[2],
                    types=crop_types,
                    center_index=int(np.searchsorted(zone, center)),
                    constraints=None
                    if request.constraints is None
                    else request.constraints.crop(zone),
                )
            hessian = self._hessian_fn(scratch, settings.fd_step)

            def stationary_hessian(
                positions: np.ndarray, free_indices: np.ndarray
            ) -> np.ndarray:
                # Check the undisplaced geometry, after premin/crop and with
                # every temporary force fix removed. Frozen reaction forces
                # are physical and do not invalidate constrained stationarity.
                forces = np.asarray(scratch.get_forces(positions=positions))
                if forces.shape != positions.shape:
                    raise ValueError("native forces must match the geometry shape")
                norms = np.linalg.norm(forces[free_indices], axis=1)
                largest = float(np.max(norms))
                if not np.all(np.isfinite(norms)) or largest > settings.force_tol:
                    # The kernel only rejects; the caller decides the rate
                    # fallback (a site request keeps its inherited estimate
                    # and reports the rejection per row, a reference request
                    # falls back to k0), so this line names neither.
                    logger.warning(
                        "[htst] event %r: stationarity check rejected the "
                        "geometry, maximum free-atom force norm %.4e eV/A "
                        "exceeds force_tol %s eV/A (n_free %d); the request is "
                        "rejected and the caller decides the fallback",
                        request.event_key,
                        largest,
                        settings.force_tol,
                        int(len(free_indices)),
                    )
                    raise PrefactorRejected(
                        PrefactorRejection.NONSTATIONARY_GEOMETRY,
                        f"maximum free-atom force norm {largest!r} eV/Å exceeds "
                        f"stationarity tolerance {settings.force_tol} eV/Å "
                        "or is nonfinite",
                    )
                return hessian(positions, free_indices)

            result = _kernel_compute_event_prefactors(
                local_request,
                stationary_hessian,
                method="lammps_eskm",
                free_indices=free_local,
                compute_backward=compute_backward,
            )
            return replace(result, provenance=provenance)
        finally:
            scratch.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_root(self) -> bool:
        """Return True on the local root of the search engine's communicator."""
        comm = self.engine.comm
        return comm is None or comm.Get_rank() == 0

    def _new_scratch(self) -> LammpsEngine:
        """Create (not start) a serial scratch engine with a per-call unique id."""
        self._scratch_calls += 1
        engine_id = (
            _SCRATCH_ID_STRIDE * (int(self.engine.engine_id) + 1) + self._scratch_calls
        )
        return LammpsEngine(
            config=_QuietConfig(self.engine.config),
            comm=MPI.COMM_SELF,
            engine_id=engine_id,
        )

    @staticmethod
    def _build_scratch(
        scratch: LammpsEngine,
        *,
        types: tuple[str, ...],
        positions: np.ndarray,
        cell: np.ndarray,
        pbc: tuple[bool, bool, bool],
        species: tuple[str, ...],
        masses: tuple[float, ...],
        expected_physics: EnginePhysics | None = None,
        check_premin: bool = False,
    ) -> None:
        """Replay parameters / system / potential on a started scratch engine."""
        scratch.initialize_parameters()
        scratch.initialize_system(
            types=types,
            positions=positions,
            cell=cell,
            pbc=pbc,
            species=species,
            masses=masses,
        )
        scratch.initialize_potential()
        actual = scratch.full_system
        if actual is None or (actual.species, actual.masses) != (species, masses):
            raise HTSTRequestError(
                "initialized scratch species/masses differ from requested physics"
            )
        if expected_physics is not None and (
            actual.physics is None
            or actual.physics.force_model != expected_physics.force_model
        ):
            raise HTSTRequestError(
                "initialized scratch force model differs from requested physics"
            )
        if (
            check_premin
            and expected_physics is not None
            and (
                actual.physics is None
                or actual.physics.premin_solver != expected_physics.premin_solver
            )
        ):
            raise HTSTRequestError(
                "initialized scratch premin inputs differ from requested physics"
            )

    @staticmethod
    def _premin(
        scratch: LammpsEngine, positions: np.ndarray, core: np.ndarray
    ) -> np.ndarray:
        """Relax everything but ``core`` (0-based indices) from ``positions``."""
        scratch.set_positions(positions)
        scratch.minimize_freeze_core(core)
        return np.asarray(scratch.get_positions(), dtype=float)

    def _hessian_fn(self, scratch: LammpsEngine, fd_step: float) -> HessianFn:
        """Bind the scratch engine into the kernel's ``hessian_fn`` signature."""

        def hessian_fn(positions: np.ndarray, free_indices: np.ndarray) -> np.ndarray:
            return self._eskm_hessian(scratch, positions, free_indices, fd_step)

        return hessian_fn

    @staticmethod
    def _run_dynamical_matrix(
        scratch: LammpsEngine, group: str, fd_step: float, path: str
    ) -> None:
        """Issue the ``dynamical_matrix`` command (isolated for failure injection)."""
        scratch.lmp.command(f"dynamical_matrix {group} eskm {fd_step!r} file {path}")

    def _eskm_hessian(
        self,
        scratch: LammpsEngine,
        positions: np.ndarray,
        free_indices: np.ndarray,
        fd_step: float,
    ) -> np.ndarray:
        """Mass-weighted partial Hessian of ``free_indices`` at ``positions``.

        LAMMPS writes ``-(F(+dx) - F(-dx)) / (2 dx sqrt(m_i m_j)) * 9648.5`` for
        the group atoms in ascending id order, three rows per atom and
        ``3 * n_free`` values per row (three per line). Dividing by
        :data:`pykmc.htst.ESKM_METAL_CONVERSION` gives eV / (amu Å²); the
        result is symmetrised and permuted into ``free_indices`` order. The
        arguments are read-only views from the kernel and are never written.

        Parameters
        ----------
        scratch : LammpsEngine
            Started and initialised serial scratch engine.
        positions : np.ndarray
            ``(N, 3)`` geometry in the scratch system's atom order.
        free_indices : np.ndarray
            0-based indices of the free atoms in that order.
        fd_step : float
            Displacement in Å.

        Returns
        -------
        np.ndarray
            ``(3F, 3F)`` symmetric matrix; non-finite entries are returned as
            such so the kernel rejects with ``NONFINITE_HESSIAN``.

        Raises
        ------
        RuntimeError
            If the output file cannot be parsed or has the wrong size.

        """
        free = np.asarray(free_indices, dtype=int)
        order = np.argsort(free, kind="stable")
        ids = " ".join(str(int(i) + 1) for i in free[order])
        scratch.set_positions(positions)
        scratch.lmp.command("run 0 post no")
        tmpdir = tempfile.mkdtemp(prefix=TMPDIR_PREFIX)
        group_defined = False
        failure: BaseException | None = None
        try:
            path = os.path.join(tmpdir, _DYNMAT_FILENAME)
            scratch.lmp.command(f"group {_FREE_GROUP} id {ids}")
            group_defined = True
            self._run_dynamical_matrix(scratch, _FREE_GROUP, fd_step, path)
            raw = _parse_eskm_file(path, free.size)
        except BaseException as exc:
            failure = exc
            raise
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            if group_defined:
                try:
                    scratch.lmp.command(f"group {_FREE_GROUP} delete")
                except Exception:
                    # A LAMMPS instance that already raised is unusable and is
                    # closed by the caller; the first failure is the one to
                    # report. Without a primary failure the cleanup error is it.
                    if failure is None:
                        raise
        h_mw = raw / ESKM_METAL_CONVERSION
        h_mw = 0.5 * (h_mw + h_mw.T)
        if not np.array_equal(order, np.arange(free.size)):
            inverse = np.empty_like(order)
            inverse[order] = np.arange(free.size)
            rows = (3 * inverse[:, None] + np.arange(3)[None, :]).reshape(-1)
            h_mw = h_mw[np.ix_(rows, rows)]
        return h_mw


def _parse_eskm_file(path: str, n_free: int) -> np.ndarray:
    """Read a text ``dynamical_matrix`` file into a ``(3F, 3F)`` float array.

    Parameters
    ----------
    path : str
        File written by ``dynamical_matrix ... file <path>`` (text, not binary).
    n_free : int
        Number of atoms in the group.

    Returns
    -------
    np.ndarray
        The raw matrix in LAMMPS eskm units, row-major as written.

    Raises
    ------
    RuntimeError
        If the file is missing, not numeric, or holds a different number of
        values than ``(3 n_free)**2``.

    """
    try:
        data = np.loadtxt(path, dtype=float, ndmin=2)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"cannot parse the dynamical_matrix output {path!r}: {exc}"
        ) from exc
    dim = 3 * n_free
    if data.size != dim * dim:
        raise RuntimeError(
            f"dynamical_matrix output {path!r} holds {data.size} values, expected "
            f"{dim * dim} for {n_free} free atoms"
        )
    return data.reshape(dim, dim)


__all__ = ["TMPDIR_PREFIX", "LammpsHTSTExtension"]
