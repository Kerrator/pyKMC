"""On-the-fly ML potential retraining controller."""

from __future__ import annotations

from dataclasses import dataclass
import subprocess
from typing import TYPE_CHECKING, Callable
from .otfml_paths import ensure_otfml_dirs
from .result import ErrorType

if TYPE_CHECKING:
    from .kmc import KMC
    from .eventsearch import EventSearch
    from .refinement import Refinement


@dataclass(frozen=True)
class OTFExtrapolationFlags:
    """Latched extrapolation state for one completed operation."""

    extrapolated: bool = False
    extreme_extrapolated: bool = False


class OTFMLController:
    """Coordinate OTF retraining around the existing KMC workflow."""

    def __init__(self, kmc: KMC) -> None:
        self.kmc = kmc
        self.config = kmc.config.otfml
        self.enabled = bool(self.config and self.config.enabled)
        if self.enabled:
            ensure_otfml_dirs()

    def is_enabled_for_phase(self, phase: str) -> bool:
        """Return whether OTF handling is enabled for a phase."""
        return self.enabled and phase in self.config.enabled_phases

    def retry_extrapolating_searches(self, event_search: EventSearch) -> None:
        """Retry extrapolating event searches until the phase is stable."""
        if not self.is_enabled_for_phase("search"):
            return
        cycle = 0
        while True:
            retry_task_ids = self.collect_search_retry_task_ids(event_search)
            if not retry_task_ids:
                return
            self._log(
                "log",
                "\t :=> OTFML retry cycle {} for {} jobs in phase 'search'.".format(
                    cycle + 1, len(retry_task_ids)
                ),
            )
            self._retrain_reload_and_minimize()
            event_search.retry(retry_task_ids)
            cycle += 1

    def retry_extrapolating_refinements(self, refinement: Refinement) -> None:
        """Retry extrapolating refinements until the phase is stable."""
        if not self.is_enabled_for_phase("refine"):
            return
        cycle = 0
        while True:
            retry_task_ids = self.collect_refinement_retry_task_ids(refinement)
            if not retry_task_ids:
                return
            self._log(
                "log",
                "\t :=> OTFML retry cycle {} for {} jobs in phase 'refine'.".format(
                    cycle + 1, len(retry_task_ids)
                ),
            )
            self._retrain_reload_and_minimize()
            refinement.retry(retry_task_ids)
            cycle += 1

    def retry_extrapolating_minimization(
        self, minimize_once: Callable[[], None]
    ) -> None:
        """Retry minimization until no further extrapolation is detected."""
        if not self.is_enabled_for_phase("minimize"):
            minimize_once()
            return

        flags = self._coerce_flags(minimize_once())
        while True:
            if not flags.extrapolated:
                return
            self._log(
                "log",
                "\t :=> OTFML detected minimization extrapolation{}.".format(
                    " above gamma_max" if flags.extreme_extrapolated else ""
                ),
            )
            self._retrain_reload_and_minimize()
            flags = self._coerce_flags(minimize_once())

    def collect_search_retry_task_ids(
        self,
        event_search: EventSearch,
    ) -> list:
        """Return search task ids that must be rerun."""
        retry_task_ids = []
        for task_id, result in enumerate(event_search.results):
            if result is None:
                continue
            if result.is_ok():
                continue
            err = result.err_value()
            if err.type not in {
                ErrorType.EXTRAPOLATION,
                ErrorType.EXTREME_EXTRAPOLATION,
            }:
                continue
            variables = err.variables or {}
            atom_index = variables.get("central_atom_index")
            if atom_index is None:
                continue
            retry_task_ids.append(task_id)
        return retry_task_ids

    def collect_refinement_retry_task_ids(
        self,
        refinement: Refinement,
    ) -> list:
        """Return refinement task ids from extrapolation errors."""
        retry_task_ids = []
        for task_id, result in enumerate(refinement.results):
            if result is None:
                continue
            if result.is_ok():
                continue
            err = result.err_value()
            if err.type not in {
                ErrorType.EXTRAPOLATION,
                ErrorType.EXTREME_EXTRAPOLATION,
            }:
                continue
            retry_task_ids.append(task_id)
        return retry_task_ids

    def _retrain_reload_and_minimize(self) -> None:
        """Retrain the potential, reload it in all sessions, and minimize."""
        if not self.enabled:
            return

        self._log(
            "log",
            "\t :=> OTFML retraining command: {}".format(self.config.retrain_command),
        )
        subprocess.run(self.config.retrain_command, shell=True, check=True)

        self.kmc.manager.use_local()
        self.kmc.manager.reload_all_potentials(self.kmc.config)
        if self.kmc.manager.global_session is not None:
            self.kmc.manager.use_global()
            self.kmc.manager.global_reload_potential(self.kmc.config)

        self.kmc._minimize_system_once()
        self.kmc.manager.use_local()
        self.kmc.manager.set_all_positions(self.kmc.system.positions)

    def _coerce_flags(self, value) -> OTFExtrapolationFlags:
        if isinstance(value, OTFExtrapolationFlags):
            return value
        return OTFExtrapolationFlags()

    def _log(self, logger_name: str, message: str) -> None:
        if getattr(self.kmc, "loggers", None) is not None:
            self.kmc.loggers.info(logger_name, message)
