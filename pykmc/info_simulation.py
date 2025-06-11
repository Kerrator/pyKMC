"""Module with functions to construct information dataclass for the output's files."""

from .result import (
    AtomicEnvironmentInfo,
    EventSearchOutput,
    ErrorInfo,
    Result,
    ReferenceEventSearchInfo,
    ErrorType,
    ReferenceValidEventsInfo,
    RefinementsInfo,
    EventRefinementOutput,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .kmc import KMC
import pandas as pd


def info_atomic_environments(
    kmc: "KMC",
    new_environments: list[str | bytes],
) -> AtomicEnvironmentInfo:
    """Construct the dataclass containing information of the atomic environments.

    Parameters
    ----------
    kmc : KMC
        the kmc object.
    new_environments : list[str | bytes]
        list of new atomic environments at the current step.

    Returns
    -------
    AtomicEnvironmentInfo
        the dataclass containing information of the atomic environments.

    """
    atomic_environments_info = AtomicEnvironmentInfo(
        total_atomic_environments_encounter=len(kmc.visited_environments),
        n_current_atomic_environments=len(
            set(kmc.atomic_environment.atomic_environment_list)
        ),
        n_new_atomic_environments=len(new_environments),
    )
    if kmc.config.control.verbosity == 2:
        atom_group = {}
        for index, item in enumerate(kmc.atomic_environment.atomic_environment_list):
            if item != "crystal":
                if item not in atom_group:
                    atom_group[item] = []
                atom_group[item].append(index)
        atomic_environments_info.atoms_grouped_by_environment = list(
            atom_group.values()
        )
    return atomic_environments_info


def info_reference_event_searches(
    results_reference_event_searches: list[Result[EventSearchOutput, ErrorInfo]],
) -> ReferenceEventSearchInfo:
    """Construct the dataclass containing informations on the reference event searches.

    Parameters
    ----------
    results_reference_event_searches : list[Result[EventSearchOutput, ErrorInfo]]
        list of results of the reference event searches procedure.

    Returns
    -------
    ReferenceEventSearchInfo
        the dataclass containing information on the reference event searches.

    """
    total_event_searches = len(results_reference_event_searches)
    n_success = 0
    n_fails = {"no_event_found": 0, "minima_not_matching_positions": 0}
    for res in results_reference_event_searches:
        if res.is_ok():
            n_success += 1
        else:
            # n_fails += 1
            if res.err_value().type == ErrorType.EVENT_NOT_FOUND:
                n_fails["no_event_found"] += 1
            else:
                n_fails["minima_not_matching_positions"] += 1
    return ReferenceEventSearchInfo(total_event_searches, n_success, n_fails)


def info_is_valid_reference_events(
    results_is_valid_events: list[Result[pd.DataFrame, ErrorInfo]],
) -> ReferenceValidEventsInfo:
    """Construct the dataclass containing informations on whether or not an event has been added to the reference table.

    Parameters
    ----------
    results_is_valid_events : list[Result[pd.DataFrame, ErrorInfo]]
        list of results of the add event procedure.

    Returns
    -------
    ReferenceValidEventsInfo
        the dataclass containing informations on whether or not an event has been added to the reference table.

    """
    n_valid_events = 0
    invalid_events = {
        "dE > emax_event": 0,
        "dE < emin_event": 0,
        "dE inverse < emin_event": 0,
        "Event asymmetric": 0,
        "Event already in reference table": 0,
    }
    for res in results_is_valid_events:
        if res.is_ok():
            n_valid_events += 1
        else:
            match res.err_value().type:
                case ErrorType.EVENT_ENERGY_HIGHER_THAN_THRESHOLD:
                    invalid_events["dE > emax_event"] += 1
                case ErrorType.EVENT_ENERGY_LOWER_THAN_THRESHOLD:
                    invalid_events["dE < emin_event"] += 1
                case ErrorType.EVENT_BACKWARD_ENERGY_LOWER_THAN_THRESHOLD:
                    invalid_events["dE inverse < emin_event"] += 1
                case ErrorType.EVENT_ASYMMETRIC:
                    invalid_events["Event asymmetric"] += 1
                case ErrorType.EVENT_NOT_NEW:
                    invalid_events["Event already in reference table"] += 1
    return ReferenceValidEventsInfo(n_valid_events, invalid_events)


def info_refinements(
    results_refinements: list[Result[EventRefinementOutput, ErrorType]],
) -> RefinementsInfo:
    """Construct the dataclass containing information on the refinements procedure.

    Parameters
    ----------
    results_refinements : list[Result[EventRefinementOutput, ErrorType]]
        list of results from the refinement procedure.

    Returns
    -------
    RefinementsInfo
        the dataclass containing information on the refinements procedure.

    """
    n_attempts = len(results_refinements)
    n_successes = 0
    n_fails = {
        "psr": {
            "no match found": 0,
            "matching score > matching thr": 0,
            "n_symmetries": [],
        },
        "invalid dE": 0,
        "invalid min": 0,
        "event not found": 0,
    }

    for res in results_refinements:
        if res.is_ok():
            n_successes += 1
        else:
            match res.err_value().type:
                case ErrorType.PSR_NO_MATCH_FOUND:
                    n_fails["psr"]["no match found"] += 1
                    n_fails["psr"]["n_symmetries"].append(
                        res.err_value().variables["n_sym_associated"]
                    )
                case ErrorType.PSR_MATCHING_SCORE_ABOVE_ACCEPTANCE_THRESHOLD:
                    n_fails["psr"]["matching score > matching thr"] += 1
                    n_fails["psr"]["n_symmetries"].append(
                        res.err_value().variables["n_sym_associated"]
                    )
                case ErrorType.REFINEMENT_INVALID_ENERGY_BARRIER:
                    n_fails["invalid dE"] += 1
                case ErrorType.REFINEMENT_INVALID_MINIMA:
                    n_fails["invalid min"] += 1
                case ErrorType.EVENT_NOT_FOUND:
                    n_fails["event not found"] += 1

    return RefinementsInfo(n_attempts, n_successes, n_fails)
