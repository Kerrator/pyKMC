"""Module implementing event search and event refinement for lammps using pARTn."""

__all__= ["pARTn_search", "pARTn_refine_event"]
import pypARTn2
import numpy as np
from ..result import (
    Result,
    ErrorInfo,
    EventSearchOutput,
    Ok,
    Err,
    ErrorType,
    EventRefinementOutput,
)
from ..config import Config
from lammps import lammps


def pARTn_search(
    lmp: lammps, config: Config, central_atom_idx: int
) -> Result[EventSearchOutput, ErrorInfo]:
    """Perform an event search with pARTn using Lammps based on the config.

    Parameters
    ----------
    lmp : lammps
        The lammps instance.
    config : Config
        The configuration of the simulation.
    central_atom_idx : int
        The central atom index.

    Returns
    -------
    Result[EventSearchOutput, ErrorInfo]
        The result of the event search

    """
    # PARAMETERS :
    delr_threshold = config.partn.delr_thr
    # INITILIZE ARTN
    artn = pypARTn2.artn(engine="lmp")

    # LAMMPS COMMANDS
    lmp.command("plugin load {}".format(config.partn.path_artnso))
    lmp.command("fix 10 all artn dmax {}".format(config.partn.dmax))
    lmp.command("min_style fire")

    # SETUP ARTN
    artn.set("engine_units", "lammps/metal")
    artn.set("verbose", config.partn.verbosity)
    artn.set("struc_format_out", "none")
    artn.set("lpush_final", True)
    artn.set(
        "lmove_nextmin", False
    )  # if true fortran runtime error when event not found
    artn.set("ninit", config.partn.ninit)
    artn.set("forc_thr", config.partn.forc_thr)
    artn.set("push_mode", config.partn.push_mode)
    if config.partn.push_mode == "rad":
        artn.set("push_dist_thr", config.partn.push_dist_thr)
    artn.set("push_step_size", config.partn.push_step_size)
    artn.set("push_ids", [central_atom_idx + 1])
    artn.set("push_over", config.partn.push_over)
    artn.set("eigen_step_size", config.partn.eigen_step_size)
    artn.set("lanczos_disp", config.partn.lanczos_disp)
    artn.set("nsmooth", config.partn.nsmooth)
    artn.set("nperp", config.partn.nperp)

    # RUN
    lmp.command("minimize 1e-6 1e-8 1000 1000")
    # EXTRACT DATA
    err = artn.get_runparam("error_message")
    if not err:
        # Results
        delr1 = artn.extract("delr_min1")
        delr2 = artn.extract("delr_min2")
        # Checks if one minimum is close to the original configuration
        if delr1 < delr_threshold or delr2 < delr_threshold:
            E_sad = artn.extract("etot_sad")
            E_min1 = artn.extract("etot_min1")
            E_min2 = artn.extract("etot_min2")

            dE_forward = E_sad - E_min1
            dE_backward = E_sad - E_min2

            min1positions = artn.extract("tau_min1")
            min2positions = artn.extract("tau_min2")
            saddlepositions = artn.extract("tau_sad")

            # find atom that moves the most
            dist = (min1positions - saddlepositions) ** 2
            dist = dist.sum(axis=-1)
            dist = np.sqrt(dist)
            dist[dist > config.atomicenvironment.rcut] = (
                0  # if atom moves more that rcutevent, consider that it crosses the cell (happens with lammps), so distance = 0 to not consider it as the one that moves the most
            )
            index_move = np.argmax(dist)
            if delr1 < delr2:  # necessary for no reconstruction option
                return Ok(
                    EventSearchOutput(
                        central_atom_index=central_atom_idx,
                        dE_forward=dE_forward,
                        dE_backward=dE_backward,
                        min1_positions=min1positions,
                        saddle_positions=saddlepositions,
                        min2_positions=min2positions,
                        move_atom_index=index_move,
                    )
                )
            else:
                return Ok(
                    EventSearchOutput(
                        central_atom_index=central_atom_idx,
                        dE_forward=dE_backward,
                        dE_backward=dE_forward,
                        min1_positions=min2positions,
                        saddle_positions=saddlepositions,
                        min2_positions=min1positions,
                        move_atom_index=index_move,
                    )
                )
        else:
            return Err(
                ErrorInfo(
                    type=ErrorType.EVENT_MINIMA_NOT_MATCH_POSITIONS,
                    message="delr1 and delr2 > at {}".format(delr_threshold),
                    variables={"delr1": delr1, "delr2": delr2},
                )
            )
    else:
        return Err(
            ErrorInfo(
                type=ErrorType.EVENT_NOT_FOUND, message="No event found", details=err
            )
        )


def pARTn_refine_event(
    lmp: lammps, config: Config, central_atom_idx: int
) -> Result[EventSearchOutput, ErrorInfo]:
    """Perform an event refinement with lammps using pARTn based on the config.

    Parameters
    ----------
    lmp : lammps
        The lammps instance.
    config : Config
        The configuration of the simulation.
    central_atom_idx : int
        The central atom index.

    Returns
    -------
    Result[EventSearchOutput, ErrorInfo]
        The results of the event refinement.

    """
    # INITILIZE ARTN
    artn = pypARTn2.artn(engine="lmp")

    # LAMMPS COMMANDS
    lmp.command("plugin load {}".format(config.partn.path_artnso))
    lmp.command("fix 10 all artn dmax {}".format(config.partn.dmax))
    lmp.command("min_style fire")

    # SETUP ARTN
    artn.set("engine_units", "lammps/metal")
    artn.set("verbose", config.partn.verbosity)
    artn.set("struc_format_out", "none")
    artn.set("lpush_final", True)
    artn.set("ninit", 0)
    artn.set("forc_thr", config.partn.forc_thr)
    artn.set("push_mode", "list")
    artn.set("push_step_size", config.partn.push_step_size)
    artn.set("push_ids", [central_atom_idx + 1])
    artn.set("push_over", config.partn.push_over)
    artn.set("eigen_step_size", config.partn.eigen_step_size)
    artn.set("lanczos_disp", config.partn.lanczos_disp)
    artn.set("nsmooth", config.partn.nsmooth)
    artn.set("nperp", config.partn.nperp)

    # RUN
    lmp.command("minimize 1e-6 1e-8 1000 1000")

    # EXTRACT DATA
    err = artn.get_runparam("error_message")
    if not err:
        E_sad = artn.extract("etot_sad")
        E_min1 = artn.extract("etot_min1")
        E_min2 = artn.extract("etot_min2")

        dE_forward = E_sad - E_min1
        dE_backward = E_sad - E_min2

        min1positions = artn.extract("tau_min1")
        min2positions = artn.extract("tau_min2")
        saddlepositions = artn.extract("tau_sad")

        # TEST CHECK IF ATOM MOVE SAME AS CENTRAL ATOM IDX
        # TODO either putting rcut in function parameters or remove this
        dist = (min1positions - saddlepositions) ** 2
        dist = dist.sum(axis=-1)
        dist = np.sqrt(dist)
        dist[dist > config.atomicenvironment.rcut] = (
            0  # if atom moves more that rcutevent, consider that it crosses the cell (happens with lammps), so distance = 0 to not consider it as the one that moves the most
        )
        index_move = np.argmax(dist)

        return Ok(
            EventRefinementOutput(
                central_atom_index=central_atom_idx,
                min1_positions=min1positions,
                min2_positions=min2positions,
                saddle_positions=saddlepositions,
                dE_forward=dE_forward,
                dE_backward=dE_backward,
                move_atom_index=index_move,
            )
        )

    else:
        return Err(
            ErrorInfo(
                type=ErrorType.EVENT_NOT_FOUND, message="no event found", details=err
            )
        )
