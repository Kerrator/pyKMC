"""ARTn event search strategy, on pyKMC's own ARTn core.

The search runs as one `artn_search` operation on a worker
(:mod:`pykmc.engine.artn_search`), on the dimer's force path, and pARTn still
refines the events. Its parameters come from the `[pARTn]` section; its seeds
from the run-wide stream, as the dimer's.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from ...dimer import _minimum_image
from ...engine.artn_search import ArtnSearchExtension
from .base import EventSearchStrategy
from .dimer import DimerStrategy, _then, dimer_search_result

if TYPE_CHECKING:
    from ...dimer import DimerResult
    from ...engine import EngineExtension
    from ...manager import Manager
    from ...result import ErrorInfo, EventSearchOutput, Result
    from ..eventsearch import SearchRequest

__all__ = [
    "ArtnStrategy",
    "artn_options",
    "ignored_partn_parameters",
]

# `[pARTn]` parameters the ARTn core takes, under the same names.
_CORE_PARAMETERS = (
    "push_mode",
    "push_dist_thr",
    "push_step_size",
    "ninit",
    "lanczos_min_size",
    "lanczos_max_size",
    "lanczos_disp",
    "lanczos_eval_conv_thr",
    "eigval_thr",
    "eigen_step_size",
    "nsmooth",
    "alpha_mix_cr",
    "nnewchance",
    "nperp",
    "nperp_limitation",
    "forc_thr",
    "convergence_property",
    "push_over",
    "nevalf_max",
    "dmax",
)


class ArtnSearchConfig(Protocol):
    """Configuration read by `ArtnStrategy`."""

    control: Any
    activevolume: Any
    atomicenvironment: Any
    eventsearch: Any
    partn: Any
    frozen_atoms: Any


def artn_options(partn: Any) -> dict[str, Any]:
    """Return the ARTn core's keyword options from a `[pARTn]` section.

    The core builds pARTn's perpendicular step limits from ``nperp`` and
    ``nperp_limitation`` as pARTn does (:func:`pykmc.artn.perp_limits`).
    """
    return {name: getattr(partn, name) for name in _CORE_PARAMETERS}


def ignored_partn_parameters(partn: Any) -> dict[str, str]:
    """Return the `[pARTn]` settings the ARTn search does not follow, with why.

    Only settings that would change pARTn's search are listed: ``neigen``
    other than 1 and a set ``zseed``. At positive pARTn verbosity the debug
    log reports the saddle and minimum displacements and the number of atoms
    that moved farther than ``[pARTn] delr_thr``.

    Parameters
    ----------
    partn : PartnConfig
        The `[pARTn]` section.

    Returns
    -------
    dict[str, str]
        For each ignored parameter, its value and the reason, as text.

    """
    ignored = {}
    if partn.neigen != 1:
        ignored["neigen"] = f"{partn.neigen} (fixed at 1, as pARTn's setup does)"
    if partn.zseed != 0:
        ignored["zseed"] = (
            f"{partn.zseed} (ARTn searches are seeded from [Control] seed; "
            "pARTn's refinement still reads zseed)"
        )
    return ignored


class ArtnStrategy(EventSearchStrategy):
    """Event search with pyKMC's ARTn core, run by the `artn_search` engine extension.

    Each request carries a seed, drawn by the facade, and the movable atoms,
    chosen as for the dimer: those within `rmov` of the central atom with the
    active volume on, every atom with it off, and in both cases none of
    `[frozen_atoms]`. The `[pARTn]` settings the core does not follow are
    logged once, at the first submit.
    """

    name = "artn"
    uses_seed = True

    def __init__(self, config: ArtnSearchConfig) -> None:
        self.config = config
        self.options = artn_options(config.partn)
        self._ignored_logged = False

    # The same movable set as the dimer's, from the same config sections.
    movable_atoms = DimerStrategy.movable_atoms

    @property
    def engine_extensions(self) -> list[type[EngineExtension]]:
        """The `artn_search` operation's extension."""
        return [ArtnSearchExtension]

    def submit(self, requests: list[SearchRequest], manager: Manager) -> list[Future]:
        """Submit one `artn_search` op per request.

        Parameters
        ----------
        requests : list[SearchRequest]
            The searches to perform, each with a seed and a movable set.
        manager : Manager
            The manager that runs the searches on the engines.

        Returns
        -------
        list[Future]
            One future per request, in the same order, each resolving to the
            search's `Result`. An error raised by the operation, such as a
            LAMMPS error, is raised by the future.

        Raises
        ------
        ValueError
            If the active volume is on and its radius `ract` is not larger than
            the cutoff radius `rcut` (refinement still runs in the active
            volume).

        """
        if self.config.control.active_volume:
            if self.config.activevolume.ract <= self.config.atomicenvironment.rcut:
                raise ValueError(
                    "Active Volume radius is smaller than cutoff radius. Please increase ract or decrease rcut"
                )
        if not self._ignored_logged:
            self._ignored_logged = True
            ignored = ignored_partn_parameters(self.config.partn)
            if ignored:
                logging.getLogger("log").warning(
                    "[EventSearch] style = artn ignores these [pARTn] settings: "
                    + "; ".join(f"{name} = {why}" for name, why in ignored.items())
                )
        delr_threshold = self.config.eventsearch.delr_thr
        moved_threshold = self.config.partn.delr_thr
        logger = logging.getLogger("log")
        report_endpoints = self.config.partn.verbosity > 0 and logger.isEnabledFor(
            logging.DEBUG
        )

        def finish(
            raw: DimerResult, request: SearchRequest
        ) -> Result[EventSearchOutput, ErrorInfo]:
            if report_endpoints:
                for name, positions in (
                    ("saddle", raw.saddle_positions),
                    ("min1", raw.min1_positions),
                    ("min2", raw.min2_positions),
                ):
                    if positions is None:
                        continue
                    delta = _minimum_image(
                        positions - request.positions, request.cell, request.pbc
                    )
                    npart = np.count_nonzero(
                        np.linalg.norm(delta, axis=1) > moved_threshold
                    )
                    logger.debug(
                        "ARTn atom %d seed %d %s: delr=%.12g Å, npart=%d",
                        request.central_atom,
                        raw.seed,
                        name,
                        np.linalg.norm(delta),
                        npart,
                    )
            return dimer_search_result(raw, request, delr_threshold, backend="artn")

        futures = []
        for request in requests:
            future = manager.submit(
                "artn_search",
                positions=request.positions,
                types=request.types,
                cell=request.cell,
                pbc=request.pbc,
                movable=request.movable,
                central_atom=request.central_atom,
                seed=request.seed,
                options=self.options,
            )
            futures.append(
                _then(future, lambda raw, request=request: finish(raw, request))
            )
        return futures
