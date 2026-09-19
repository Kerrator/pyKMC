"""Batching service for per-event HTST prefactor requests.

One :class:`PrefactorService` is built by ``KMC`` for the htst/rpa styles and
shared by the reference and active event tables. It is the only place that:

- converts the configured ``nu0_min_THz``/``nu0_max_THz`` window to Hz
  (:func:`pykmc.rate_constant.thz_to_hz`, exactly once, into a frozen
  :class:`~pykmc.htst.settings.HTSTSettings`);
- builds :class:`~pykmc.htst.request.HTSTEventRequest` objects from the full
  system geometry, taking the potential species order and masses from the
  engine's authoritative map when the service was built with one
  (``species_masses``, the root worker's ``htst_preflight`` report in a live
  run) and otherwise from the one species rule
  (:func:`pykmc.engine.lammps.species_map`, ASE masses, imported lazily so
  the constant path never imports LAMMPS; the offline/test path);
- fans the requests out through ``Manager.submit("compute_event_prefactors",
  request=..., compute_backward=...)`` (one job per accepted event; both
  directions for a reference event, the forward direction only for a site
  request), keeps the request-to-Future association keyed by ``event_key`` and
  resolves every Future before any consumer reads a rate, returning the results
  mapped by ``event_key`` regardless of completion order with strict
  cardinality;
- measures the wall time of every batch around the submission and Future
  resolution and accumulates the request count and wall time per KMC step for
  the ``[htst]`` log lines and the per-step summary.

Exceptions raised by a worker operation propagate through the Future: a
transport or programming failure is never turned into a ``k0`` fallback. Only
the scientific rejections encoded in ``EventPrefactors`` fall back, and that
policy lives in the event tables, not here.

Nothing in the constant path imports this module.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from pykmc.htst.request import HTSTEventRequest, HTSTRequestError
from pykmc.htst.result import EventPrefactors
from pykmc.htst.settings import HTSTSettings
from pykmc.physics import (
    EnginePhysics,
    PhysicalDescriptor,
    ResolvedConstraints,
    _indices,
)

from .rate_constant import RateConstant
from .units import thz_to_hz

if TYPE_CHECKING:
    from pykmc.config import Config, RateConstantConfig

PREFACTOR_OPERATION: str = "compute_event_prefactors"
"""Worker operation exposed by ``LammpsHTSTExtension`` (contracts section 6)."""


def settings_from_config(rate_config: RateConstantConfig) -> HTSTSettings:
    """Build the frozen kernel settings from the rate-constant configuration.

    The THz window is converted to Hz here and nowhere else.

    Parameters
    ----------
    rate_config : RateConstantConfig
        Rate-constant section of the simulation configuration.

    Returns
    -------
    HTSTSettings
        Validated settings with ``nu0_min_hz``/``nu0_max_hz`` in Hz.

    """
    return HTSTSettings(
        free_radius=rate_config.free_radius,
        fd_step=rate_config.fd_step,
        force_tol=rate_config.force_tol,
        zone_radius=rate_config.zone_radius,
        premin=rate_config.premin,
        nu0_min_hz=thz_to_hz(rate_config.nu0_min_THz),
        nu0_max_hz=thz_to_hz(rate_config.nu0_max_THz),
        free_region_center=rate_config.free_region_center,
    )


def _validated_map(
    species_masses: tuple[tuple[str, ...], tuple[float, ...]],
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Normalise and validate an engine ``(species, masses)`` map.

    Parameters
    ----------
    species_masses : tuple[tuple[str, ...], tuple[float, ...]]
        Species in potential order and one finite positive mass (amu) each.

    Returns
    -------
    tuple[tuple[str, ...], tuple[float, ...]]
        The map as tuples of ``str`` and ``float``.

    Raises
    ------
    ValueError
        If the pair is malformed: not two sequences, empty, unequal lengths,
        duplicate species, or a non-finite or non-positive mass.

    """
    try:
        raw_species, raw_masses = species_masses
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "species_masses must be a (species, masses) pair of sequences"
        ) from exc
    species = tuple(str(s) for s in raw_species)
    masses = tuple(float(m) for m in raw_masses)
    if not species:
        raise ValueError("species_masses: the species tuple is empty")
    if len(species) != len(masses):
        raise ValueError(
            f"species_masses: {len(species)} species but {len(masses)} masses"
        )
    if len(set(species)) != len(species):
        raise ValueError(f"species_masses: duplicate species in {species}")
    if not all(math.isfinite(m) and m > 0.0 for m in masses):
        raise ValueError(f"species_masses: masses must be finite and > 0, got {masses}")
    return species, masses


class PrefactorService:
    """Build HTST requests and resolve them through the manager, in batches.

    Parameters
    ----------
    config : Config
        Full simulation configuration (``config.rateconstant`` is used).
    manager : Manager
        The engine manager; only ``submit`` is used.
    rate_constant : RateConstant
        The rate facade of the run; its backend must require per-event
        prefactors (htst/rpa), otherwise building the service is an error
        because the constant path must never construct one.
    species_masses : tuple[tuple[str, ...], tuple[float, ...]] or None, optional
        The engine's authoritative ``(species, masses)`` map, in potential
        species order (``htst_preflight`` returns it). When set, every request
        carries it and each ``types`` entry must be one of its species; when
        ``None`` the map is derived from ``types`` through ``species_map``
        (ASE masses). Contracts section 7d, N3.

    Attributes
    ----------
    settings : HTSTSettings
        Kernel settings shared by every request of the run.
    species_masses : tuple[tuple[str, ...], tuple[float, ...]] or None
        The map given at construction (validated), or ``None``.
    n_submitted : int
        Total number of requests submitted so far (diagnostics).
    last_batch_wall_s : float
        Wall time (s) of the most recent :meth:`compute` batch, measured
        around the submissions and the resolution of every Future.
    step_requests : int
        Requests submitted since :meth:`reset_step_counters` was last called.
    step_wall_s : float
        Wall time (s) of the batches run since :meth:`reset_step_counters`.

    """

    def __init__(
        self,
        config: Config,
        manager: Any,
        rate_constant: RateConstant,
        *,
        species_masses: tuple[tuple[str, ...], tuple[float, ...]] | None = None,
        engine_physics: EnginePhysics | None = None,
        global_constraints: ResolvedConstraints | None = None,
    ) -> None:
        if not rate_constant.backend.requires_event_prefactors:
            raise ValueError(
                "PrefactorService is only meaningful for a backend that requires "
                f"per-event prefactors; got style {config.rateconstant.style!r}"
            )
        if manager is None:
            raise RuntimeError(
                "PrefactorService needs the engine manager; it is None (the "
                "manager must be set before KMC initialisation)"
            )
        self.config = config
        self.manager = manager
        self.rate_constant = rate_constant
        self.settings = settings_from_config(config.rateconstant)
        self.species_masses = (
            None if species_masses is None else _validated_map(species_masses)
        )
        if engine_physics is not None:
            if not isinstance(engine_physics, EnginePhysics):
                raise ValueError("engine_physics must be an EnginePhysics")
            authoritative = (engine_physics.species, engine_physics.masses)
            if self.species_masses is not None and self.species_masses != authoritative:
                raise ValueError("preflight descriptor and species/masses disagree")
            self.species_masses = authoritative
        self.engine_physics = engine_physics
        if global_constraints is not None and not isinstance(
            global_constraints, ResolvedConstraints
        ):
            raise ValueError("global_constraints must be ResolvedConstraints")
        self.global_constraints = global_constraints
        self._descriptor = (
            None
            if self.species_masses is None
            else PhysicalDescriptor.from_config(
                config,
                engine_physics
                or EnginePhysics.capture(config.lammps, *self.species_masses),
                self.settings,
            )
        )
        self.n_submitted = 0
        self.last_batch_wall_s = 0.0
        self.step_requests = 0
        self.step_wall_s = 0.0

    def descriptor_for(self, types: Sequence[str]) -> PhysicalDescriptor:
        """Return the shared physical contract, retaining absent potential slots."""
        if self._descriptor is None:
            from pykmc.engine.lammps import species_map

            self.species_masses = species_map(list(types))
            self._descriptor = PhysicalDescriptor.from_config(
                self.config,
                EnginePhysics.capture(self.config.lammps, *self.species_masses),
                self.settings,
            )
        if not set(types).issubset(self._descriptor.engine.species):
            raise HTSTRequestError("types are not in the engine species map")
        return self._descriptor

    @property
    def current_descriptor(self) -> PhysicalDescriptor | None:
        """Current context, never a claim about a previously saved estimate."""
        return self._descriptor

    def reset_step_counters(self) -> None:
        """Zero the per-step request count and wall time (called at each KMC step)."""
        self.step_requests = 0
        self.step_wall_s = 0.0

    def build_request(
        self,
        *,
        event_key: tuple,
        min1_positions: np.ndarray,
        saddle_positions: np.ndarray,
        min2_positions: np.ndarray,
        types: Sequence[str],
        cell: np.ndarray,
        pbc: Sequence[bool],
        center_index: int,
        constraints: ResolvedConstraints | None = None,
        atom_ids: Sequence[int] | None = None,
    ) -> HTSTEventRequest:
        """Build and validate one request from full-system geometry.

        Parameters
        ----------
        event_key : tuple
            Hashable logical identity echoed back by the worker.
        min1_positions, saddle_positions, min2_positions : np.ndarray
            Full-system ``(N, 3)`` positions of the three stationary points,
            in one common atom order. They are copied.
        types : Sequence[str]
            Chemical symbol of every atom of the full system, same order.
        cell : np.ndarray
            ``(3, 3)`` simulation cell.
        pbc : Sequence[bool]
            Actual periodicity per axis of the system.
        center_index : int
            Global index of the moving atom.

        Returns
        -------
        HTSTEventRequest
            The validated request.

        Raises
        ------
        HTSTRequestError
            If the service carries an engine map and a symbol of ``types`` is
            not one of its species (the request cannot describe that atom),
            or if the request fails validation.

        """
        symbols = tuple(str(t) for t in types)
        descriptor = self.descriptor_for(symbols)
        species, masses = descriptor.engine.species, descriptor.engine.masses
        if constraints is not None and atom_ids is not None:
            raise HTSTRequestError(
                "pass resolved constraints or source atom_ids, not both"
            )
        if constraints is None:
            if self.global_constraints is not None:
                constraints = self.global_constraints
                if atom_ids is not None:
                    try:
                        rows = [
                            constraints.atom_ids.index(i) for i in _indices(atom_ids)
                        ]
                        constraints = constraints.crop(rows)
                    except ValueError as exc:
                        raise HTSTRequestError(
                            "atom_ids must map to the initialized source"
                        ) from exc
            else:
                constraints = descriptor.resolve_constraints(
                    min1_positions, symbols, atom_ids
                )
        axes = tuple(pbc)
        if len(axes) != 3 or not all(isinstance(p, (bool, np.bool_)) for p in axes):
            raise HTSTRequestError("pbc must contain three bools")
        request = HTSTEventRequest(
            event_key=event_key,
            min1_positions=np.array(min1_positions, dtype=float, copy=True),
            saddle_positions=np.array(saddle_positions, dtype=float, copy=True),
            min2_positions=np.array(min2_positions, dtype=float, copy=True),
            types=symbols,
            species=species,
            masses=masses,
            cell=np.array(cell, dtype=float, copy=True),
            pbc=tuple(bool(p) for p in axes),
            center_index=center_index,
            settings=self.settings,
            descriptor=descriptor,
            constraints=constraints,
            user_constraints=self.global_constraints,
        )
        request.validate()
        # Freeze a stateless policy resolution before downstream premin/crop.
        # A supplied AV union may add locks, never replace the known user mask.
        request = replace(request, user_constraints=request.resolved_user_constraints())
        return request

    def compute(
        self, requests: Sequence[HTSTEventRequest], *, compute_backward: bool = True
    ) -> dict[tuple, EventPrefactors]:
        """Submit every request, wait for all of them and map results by key.

        Parameters
        ----------
        requests : Sequence[HTSTEventRequest]
            Requests with pairwise distinct ``event_key`` values.
        compute_backward : bool, optional
            Forwarded to the worker operation. ``False`` (site requests)
            skips the ``min2`` Hessian; the backward direction of every result
            is then ``status="skipped"``.

        Returns
        -------
        dict[tuple, EventPrefactors]
            One result per request keyed by its ``event_key``; the completion
            order of the workers is irrelevant because each Future is held
            against the key it was submitted for.

        Raises
        ------
        ValueError
            If two requests share an ``event_key``.
        RuntimeError
            If a worker returns something that is not an ``EventPrefactors``
            for the submitted key (a transport or contract failure).
        Exception
            Whatever a worker operation raised, re-raised from its Future.

        """
        keys = [req.event_key for req in requests]
        if len(set(keys)) != len(keys):
            raise ValueError(f"event_key values must be distinct, got {keys}")
        if not isinstance(compute_backward, bool):
            raise ValueError(
                f"compute_backward must be a bool, got {compute_backward!r}"
            )
        # Validate the complete batch before submitting any work. Requests may
        # come from another builder; they cannot replace this run's initialized
        # fixed references with an otherwise internally consistent snapshot.
        for req in requests:
            req.validate()
            if self.global_constraints is not None:
                if req.constraints is None:
                    raise HTSTRequestError(
                        "initialized constraints need an execution mask"
                    )
                try:
                    req.constraints.require_preserves(
                        self.global_constraints, cell=req.cell, pbc=req.pbc
                    )
                except ValueError as exc:
                    raise HTSTRequestError(str(exc)) from exc
        start = time.perf_counter()
        pending: list[tuple[tuple, Any]] = []
        try:
            for req in requests:
                future = self.manager.submit(
                    PREFACTOR_OPERATION,
                    request=req,
                    compute_backward=compute_backward,
                )
                pending.append((req.event_key, future))
            self.n_submitted += len(pending)
            self.step_requests += len(pending)

            results: dict[tuple, EventPrefactors] = {}
            for key, future in pending:
                value = future.result()  # a worker exception propagates here
                if not isinstance(value, EventPrefactors):
                    raise RuntimeError(
                        f"{PREFACTOR_OPERATION} returned {type(value).__name__} for "
                        f"event {key!r}; expected EventPrefactors"
                    )
                if value.event_key != key:
                    raise RuntimeError(
                        f"{PREFACTOR_OPERATION} echoed event_key {value.event_key!r} "
                        f"for the request submitted as {key!r}"
                    )
                results[key] = value
        finally:
            # The batch time covers submission and resolution, also when a
            # worker failure propagates (the step summary then never prints).
            self.last_batch_wall_s = time.perf_counter() - start
            self.step_wall_s += self.last_batch_wall_s
        if len(results) != len(requests):
            raise RuntimeError(
                f"resolved {len(results)} results for {len(requests)} requests"
            )
        return results


__all__ = ["PREFACTOR_OPERATION", "PrefactorService", "settings_from_config"]
