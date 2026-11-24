"""HTST rate constant: a Vineyard prefactor nu0 per event, k0 when there is none.

The LAMMPS engine extension, `lammps_extension`, is not imported here: it loads
`lammps` and `pypARTn`, and `engine_extensions` imports it only when
`[RateConstant] style = htst`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Protocol

import numpy as np

from pykmc._core.result import Err, ErrorInfo, Ok, Result
from pykmc.region import Region

from ...result import RateConstantOutput
from ..base import RateConstantStrategy
from .errors import PrefactorError
from .region import build_region, frozen_overlap, wrapped_on_open_axis
from .thresholds import NU0_WARN_BAND_THZ

if TYPE_CHECKING:
    from pykmc.config import Config
    from pykmc.engine.base import EngineExtension
    from pykmc.log import LogKMC
    from pykmc.manager import Manager
    from pykmc.system import System

    from ...rate_constant import RateRequest


class HTSTConfig(Protocol):
    """Configuration required by `HTSTStrategy`."""

    k0: float  # ps^-1
    T: float  # K
    rmov: float  # A
    rbuff: Optional[float]  # A
    fd_step: float  # A


class HTSTStrategy(RateConstantStrategy):
    r"""Rate constant with a prefactor `nu0` per event, and `k0` for events without one.

    $$
    k = \nu_{0} \, e^{-\frac{dE}{k_{b}T}}
    $$

    `nu0` is the harmonic (Vineyard) prefactor from the Hessians at the minimum and
    at the saddle, each computed by `dynamical_matrix` in a serial scratch LAMMPS
    on a session of the engine manager. It is computed for every request that
    comes with its geometry and without a prefactor.
    """

    name = "htst"

    def __init__(self, config: HTSTConfig) -> None:
        self.config = config
        # set by setup
        self._full_config: Any = None
        self._manager: Any = None
        self._system: Any = None
        self._loggers: Any = None
        self._rbuff: Optional[float] = None

    @property
    def engine_extensions(self) -> tuple[type[EngineExtension], ...]:
        """The serial-LAMMPS extension that computes the Hessians."""
        from .lammps_extension import HTSTLammpsExtension

        return (HTSTLammpsExtension,)

    def setup(
        self, config: Config, manager: Manager, system: System, loggers: LogKMC
    ) -> None:
        """Check that HTST prefactors can be computed and set the Hessian region.

        A run that could never compute a prefactor stops here, with the reason in
        the log.

        Parameters
        ----------
        config : Config
            The full simulation configuration.
        manager : Manager
            The engine session pool: every session checks its own LAMMPS.
        system : System
            The simulated system. It is kept: `compute` reads its cell, its
            species and its current positions.
        loggers : LogKMC
            The simulation loggers.

        Raises
        ------
        RuntimeError
            If HTST prefactors can never be computed in this run.

        """
        rc = self.config
        species = sorted(set(system.types))
        try:
            if not system.cell.orthorhombic:
                raise RuntimeError(
                    "HTST: style = htst needs an orthogonal cell; this cell has "
                    "off-diagonal terms. Use an orthogonal cell or set [RateConstant] "
                    "style = constant."
                )
            if not all(system.pbc):
                raise RuntimeError(
                    "HTST: style = htst needs a cell that is periodic in every "
                    "direction: the atoms of a Hessian are selected with periodic "
                    "images on every axis. Use a periodic cell with a vacuum gap or "
                    "set [RateConstant] style = constant."
                )
            # Every session checks its own LAMMPS and its access to hessian_dir.
            manager.broadcast("htst_capability_check", config=config, species=species)
            # One session reads the force cutoff, when the shell width needs it.
            info = manager.htst_capability_check(
                config=config, species=species, read_cutoff=rc.rbuff is None
            ).result()
            if rc.rbuff is not None:
                shell = rc.rbuff
            elif info["force_cutoff"] is None or info["force_cutoff"] <= 0:
                raise RuntimeError(
                    "HTST: the force cutoff could not be read from LAMMPS "
                    "('Communication cutoff' missing or not positive in info "
                    "communication); set [RateConstant] rbuff."
                )
            else:
                shell = 2.0 * info["force_cutoff"]
        except Exception as exc:
            # run.py aborts before the exception is printed: log the reason.
            loggers.error("log", str(exc))
            raise
        self._full_config, self._manager = config, manager
        self._system, self._loggers = system, loggers
        self._rbuff = shell

        loggers.info(
            "log",
            f"HTST: LAMMPS version {info['lammps_version']}; Hessian scratch files "
            f"go to {info['hessian_dir']}; fd_step = {rc.fd_step:g} A.",
        )
        source = (
            "[RateConstant] rbuff"
            if rc.rbuff is not None
            else f"2 x force cutoff {info['force_cutoff']:g} A read from LAMMPS"
        )
        loggers.info(
            "log",
            "HTST: Hessian region: atoms within [RateConstant] rmov = "
            f"{rc.rmov:g} A of the moving atom's saddle position are "
            f"displaced, a frozen shell of {shell:g} A ({source}) is kept, atoms "
            "beyond are left out.",
        )
        if config.atomicenvironment.atom_coloring_mode == "grey" and len(species) > 1:
            loggers.warning(
                "log",
                f"HTST: atom_coloring_mode = grey with {len(species)} species: a "
                "reference nu0 is computed with the species of the event it was "
                "found on and is reused by the events on other species.",
            )

    def prefactor(self) -> float:
        """Return `[RateConstant] k0`, the prefactor of an event without a `nu0`.

        Returns
        -------
        float
            `k0` (ps^-1).

        """
        return self.config.k0

    def compute(
        self, requests: list[RateRequest]
    ) -> list[Result[RateConstantOutput, ErrorInfo]]:
        """Compute the rate constants of several events in one call.

        A request with a prefactor uses it. A request with its geometry (initial
        and saddle positions) gets its own nu0; requests sharing the same
        `saddle_positions` array are one Hessian job on a free session, and every
        job is submitted before any result is awaited. Any other request uses
        `k0`.

        Parameters
        ----------
        requests : list[RateRequest]
            The required data. Geometries are those of the system as it is now:
            the minimum the events were searched from.

        Returns
        -------
        list[Result[RateConstantOutput, ErrorInfo]]
            One result per request, in the same order: `Ok` with the rate
            constant and its prefactor, or `Err` with the reason there is no nu0
            and a backup computed with `k0`.

        Notes
        -----
        It never raises: a job that cannot be built, sent or collected is an
        `Err` for each of its requests.

        """
        results: list[Any] = [None] * len(requests)
        events: dict[int, list[int]] = {}  # saddle array -> its requests
        for i, request in enumerate(requests):
            if request.prefactor is not None:
                results[i] = self._ok(request, request.prefactor)
            elif request.initial_positions is None or request.saddle_positions is None:
                results[i] = self._ok(request, self.config.k0)
            else:
                events.setdefault(id(request.saddle_positions), []).append(i)
        if not events:
            return results

        groups = list(events.values())
        outcomes = self._jobs([[requests[i] for i in group] for group in groups])
        for group, per_event in zip(groups, outcomes, strict=True):
            for i, outcome in zip(group, per_event, strict=True):
                results[i] = self._result(requests[i], outcome)
        if self._loggers is not None:
            n = sum(len(group) for group in groups)
            n_ok = sum(isinstance(results[i], Ok) for group in groups for i in group)
            self._loggers.info(
                "log", f"HTST: nu0 computed for {n_ok} of {n} new event directions"
            )
        return results

    def _ok(self, request: RateRequest, prefactor: float) -> Ok[RateConstantOutput]:
        """Return the rate constant of `request` with a known prefactor."""
        k = prefactor * self._boltzmann_factor(request.dE)
        return Ok(RateConstantOutput(k=k, prefactor=prefactor))

    def _result(
        self, request: RateRequest, outcome: Result[float, ErrorInfo]
    ) -> Result[RateConstantOutput, ErrorInfo]:
        """Turn the result of a Hessian job into the rate constant of one request.

        Parameters
        ----------
        request : RateRequest
            The request the Hessian job was for.
        outcome : Result[float, ErrorInfo]
            `Ok` with nu0 (THz), or `Err` with the reason there is none.

        Returns
        -------
        Result[RateConstantOutput, ErrorInfo]
            `Ok` with the rate constant computed with nu0, or `Err` with the
            reason and a backup computed with `k0`.

        """
        if isinstance(outcome, Ok):
            nu0 = outcome.ok_value()
            if np.isfinite(nu0) and nu0 > 0:
                low, high = NU0_WARN_BAND_THZ
                if not low <= nu0 <= high and self._loggers is not None:
                    self._loggers.warning(
                        "log",
                        f"HTST: nu0 = {nu0:.4g} THz is outside 0.1-1000 THz "
                        f"(atom {request.central_atom}, barrier {request.dE:.4f} "
                        "eV); used as computed",
                    )
                return self._ok(request, float(nu0))
            error = ErrorInfo(
                type=PrefactorError.NONFINITE_PREFACTOR,
                message="nu0 is not a finite positive number "
                f"(value {float(nu0)!r} THz)",
            )
        else:
            error = outcome.err_value()
        backup = self._ok(request, self.config.k0).ok_value()
        return Err(
            ErrorInfo(
                type=error.type, message=error.message, variables={"backup": backup}
            )
        )

    def _frozen_ids(self) -> np.ndarray:
        """Return the indices of the frozen atoms (none without [frozen_atoms])."""
        region = self._full_config.frozen_atoms
        if region is None:
            return np.empty(0, dtype=np.int64)
        frozen = Region.create(
            region.region_type,
            config=region,
            types=region.types,
            indices=region.indices,
            invert=region.side == "outside",
        )
        mask = frozen.mask(self._system.positions, self._system.types)
        return np.flatnonzero(mask).astype(np.int64)

    def _jobs(
        self, events: list[list[RateRequest]]
    ) -> list[list[Result[float, ErrorInfo]]]:
        """Run the Hessian job of each event and return their results.

        Parameters
        ----------
        events : list[list[RateRequest]]
            For each event, the requests sharing its saddle: one per minimum.

        Returns
        -------
        list[list[Result[float, ErrorInfo]]]
            For each event, one result per request: `Ok` with nu0 (THz) or `Err`
            with the reason there is none.

        """

        def failure(code: PrefactorError, message: str, n: int) -> list:
            return [Err(ErrorInfo(type=code, message=message))] * n

        def well_formed(outcome: object) -> bool:
            if isinstance(outcome, Ok):
                return isinstance(outcome.ok_value(), (float, np.floating))
            return isinstance(outcome, Err) and isinstance(
                outcome.err_value(), ErrorInfo
            )

        try:
            if self._manager is None:
                raise RuntimeError("setup was not called")
            full_config, system = self._full_config, self._system
            rc = self.config
            cell = np.asarray(system.cell)
            pbc = tuple(bool(p) for p in system.pbc)
            types = np.asarray(system.types)
            species = sorted(set(system.types))
            frozen_ids = self._frozen_ids()
        except Exception as exc:
            message = f"region build failed: {type(exc).__name__}: {exc}"
            return [
                failure(PrefactorError.ENGINE_FAILURE, message, len(event))
                for event in events
            ]

        # Submit every job before waiting for any result.
        submitted = []
        for event in events:
            saddle = event[0].saddle_positions
            minima = [request.initial_positions for request in event]
            centre = event[0].central_atom
            n = len(event)
            try:
                region = build_region(
                    saddle,
                    cell,
                    pbc,
                    center_index=centre,
                    r_displaced=rc.rmov,
                    r_present=rc.rmov + self._rbuff,
                )
                # The search recentres its geometries with a periodic wrap.
                wrapped = not all(pbc) and wrapped_on_open_axis(
                    region, (saddle, *minima), system.positions, cell, pbc
                )
                n_frozen = frozen_overlap(region, frozen_ids)
            except Exception as exc:
                submitted.append(
                    failure(
                        PrefactorError.ENGINE_FAILURE,
                        f"region build failed: {type(exc).__name__}: {exc}",
                        n,
                    )
                )
                continue
            if wrapped:
                submitted.append(
                    failure(
                        PrefactorError.WRAPPED_OPEN_AXIS,
                        "recentring wrapped atoms across a non-periodic axis",
                        n,
                    )
                )
                continue
            if n_frozen > 0:
                submitted.append(
                    failure(
                        PrefactorError.FROZEN_OVERLAP,
                        "the frozen region overlapped the HTST calculation space "
                        f"({n_frozen} frozen atoms among the displaced atoms)",
                        n,
                    )
                )
                continue
            present = region.present
            try:
                submitted.append(
                    self._manager.compute_htst_prefactors(
                        config=full_config,
                        saddle=saddle[present],
                        minima=[minimum[present] for minimum in minima],
                        types=types[present].tolist(),
                        species=species,
                        cell=cell,
                        pbc=region.pbc,
                        displaced=region.displaced,
                        n_zero_modes=region.n_zero_modes,
                    )
                )
            except Exception as exc:
                submitted.append(
                    failure(
                        PrefactorError.ENGINE_FAILURE,
                        f"HTST job failed: {type(exc).__name__}: {exc}",
                        n,
                    )
                )

        results = []
        for event, job in zip(events, submitted, strict=True):
            n = len(event)
            if isinstance(job, list):  # not sent
                results.append(job)
                continue
            try:
                outcomes = job.result()
            except Exception as exc:
                results.append(
                    failure(
                        PrefactorError.ENGINE_FAILURE,
                        f"HTST job failed: {type(exc).__name__}: {exc}",
                        n,
                    )
                )
                continue
            if not (
                isinstance(outcomes, list)
                and len(outcomes) == n
                and all(well_formed(outcome) for outcome in outcomes)
            ):
                outcomes = failure(
                    PrefactorError.ENGINE_FAILURE,
                    "HTST job returned an unexpected result: "
                    f"{type(outcomes).__name__}",
                    n,
                )
            results.append(outcomes)
        return results
