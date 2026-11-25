"""ARTn event search strategy, on pyKMC's own ARTn core.

The search runs as one `artn_search` operation on a worker
(:mod:`pykmc.engine.artn_search`), on the dimer's force path or, with
`[ActiveVolume] artn_forces = av`, on pARTn's active volume. The refiner
is `[EventSearch] refine_style`'s, pARTn by default. Its parameters come from the
`[pARTn]` section; its seeds from the run-wide stream, as the dimer's.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from ...artn import perp_limits
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
    "artn_refine_options",
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


# The ARTn core's keyword options in refinement, from the `[pARTn]` r_* fields
# pyKMC's partn_refine passes to pARTn (INVENTORY.md, "Refinement", "Keys").
# convergence_property and push_over keep the core's defaults: partn_refine sets
# neither, so pARTn uses "maxval", and push_over is unread at the saddle end.
_REFINE_PARAMETERS = {
    "push_mode": "r_push_mode",
    "push_dist_thr": "r_push_dist_thr",
    "push_step_size": "r_push_step_size",
    "ninit": "r_ninit",
    "lanczos_min_size": "r_lanczos_min_size",
    "lanczos_max_size": "r_lanczos_max_size",
    "lanczos_disp": "r_lanczos_disp",
    "lanczos_eval_conv_thr": "r_lanczos_eval_conv_thr",
    "eigval_thr": "r_eigval_thr",
    "eigen_step_size": "r_eigen_step_size",
    "nsmooth": "r_nsmooth",
    "alpha_mix_cr": "r_alpha_mix_cr",
    "nnewchance": "r_nnewchance",
    "nperp": "r_nperp",
    "nperp_limitation": "r_nperp_limitation",
    "forc_thr": "r_forc_thr",
    "nevalf_max": "r_nevalf_max",
    "dmax": "r_dmax",
}


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


def artn_refine_options(partn: Any) -> dict[str, Any]:
    """Return the ARTn core's keyword options in refinement, from `[pARTn]`.

    The ``r_*`` fields under the core's names. ``push_dist_thr`` is passed
    whatever ``r_push_mode`` is; the core reads it only with ``rad``.
    """
    return {name: getattr(partn, field) for name, field in _REFINE_PARAMETERS.items()}


def ignored_partn_parameters(
    partn: Any, style: str = "artn", refine_style: str = "partn"
) -> dict[str, str]:
    """Return the `[pARTn]` settings pyKMC's ARTn does not follow, with why.

    Only settings that would change pARTn's search or refinement are listed:
    ``neigen`` other than 1 with ``style = artn``, ``r_neigen`` other than 1
    with ``refine_style = artn``, and a set ``zseed`` with either. The
    ``zseed`` entry names what still reads it.

    Parameters
    ----------
    partn : PartnConfig
        The `[pARTn]` section.
    style, refine_style : str
        `[EventSearch] style` and `refine_style`.

    Returns
    -------
    dict[str, str]
        For each ignored parameter, its value and the reason, as text.

    """
    ignored = {}
    if style == "artn" and partn.neigen != 1:
        ignored["neigen"] = f"{partn.neigen} (fixed at 1, as pARTn's setup does)"
    if refine_style == "artn" and partn.r_neigen != 1:
        ignored["r_neigen"] = f"{partn.r_neigen} (fixed at 1, as pARTn's setup does)"
    seeded = [
        name
        for name, artn in (
            ("searches", style == "artn"),
            ("refinements", refine_style == "artn"),
        )
        if artn
    ]
    readers = [
        name
        for name, partn_style in (
            ("pARTn's search", style == "partn"),
            ("pARTn's refinement", refine_style == "partn"),
        )
        if partn_style
    ]
    if partn.zseed != 0 and seeded:
        still = (
            f"{' and '.join(readers)} still reads zseed"
            if readers
            else "nothing reads zseed"
        )
        ignored["zseed"] = (
            f"{partn.zseed} (ARTn {' and '.join(seeded)} are seeded from [Control] seed; "
            f"{still})"
        )
    return ignored


class ArtnStrategy(EventSearchStrategy):
    """Event search with pyKMC's ARTn core, run by the `artn_search` engine extension.

    Each request carries a seed, drawn by the facade, and the movable atoms,
    chosen as for the dimer: those within `rmov` of the central atom with the
    active volume on, every atom with it off, and in both cases none of
    `[frozen_atoms]`. With the active volume on and `[ActiveVolume]
    artn_forces = av`, the search runs on pARTn's active volume instead of the full
    system (the worker option ``av_radii``). The `[pARTn]` settings the core
    does not follow are logged once, at the first submit.
    """

    name = "artn"
    uses_seed = True

    def __init__(self, config: ArtnSearchConfig) -> None:
        self.config = config
        perp_limits(config.partn.nperp, config.partn.nperp_limitation)
        self.options = artn_options(config.partn)
        if config.control.active_volume and config.activevolume.artn_forces == "av":
            av = config.activevolume
            self.options["av_radii"] = (av.rmov, av.ract)
        self._ignored_logged = False

    # The same movable set as the dimer's, from the same config sections.
    movable_atoms = DimerStrategy.movable_atoms

    @property
    def engine_extensions(self) -> list[type[EngineExtension]]:
        """The `artn_search` operation's extension."""
        return [ArtnSearchExtension]

    def submit(self, requests: list[SearchRequest], manager: Manager) -> list[Future]:
        """Submit one `artn_search` op per request.

        At positive pARTn verbosity the debug log reports the saddle and
        minimum displacements and the number of atoms that moved farther than
        ``[pARTn] delr_thr``.

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
            ignored = ignored_partn_parameters(
                self.config.partn, "artn", self.config.eventsearch.refine_style
            )
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
