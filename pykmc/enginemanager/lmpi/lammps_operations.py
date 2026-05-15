import numpy as np
from ase.data import atomic_numbers, atomic_masses
from mpi4py import MPI
import ctypes
import pypARTn
import os
from ...activevolume.active_volume import reset, redefine_atoms, partn_search_AV, partn_refine_AV, position_results_AV
from ...otfml_paths import (
    OTFML_EXTRAPOLATION_TOLERANCE,
    OTFML_MAX_GAMMA,
    OTFML_MAX_FLAG_INTERNAL,
    OTFML_MAX_FLAG_VARIABLE,
    OTFML_TOL_FLAG_INTERNAL,
    OTFML_TOL_FLAG_VARIABLE,
    ensure_otfml_dirs,
    session_dump_path,
)
from ...otfml import OTFExtrapolationFlags

from ...result import  (
    Result,
    ErrorInfo,
    EventSearchOutput,
    Ok,
    Err,
    ErrorType,
    EventRefinementOutput,
)


def initialize_parameters(engine) :
    engine.command("units metal")
    engine.command("atom_style atomic")
    engine.command("dimension 3")
    engine.command("boundary p p p")
    engine.command("atom_modify map array") #! necessary for scatter atoms
    engine.command("atom_modify sort 0 0.0") #! necessary for partn

def initialize_system(engine, system) :

        #system parameters
        natoms = len(system.types)
        cell = system.cell
        types = system.types
        x = system.positions.flatten()  # Lammps format

        xhi, yhi, zhi = cell[0][0], cell[1, 1], cell[2, 2]

        ind = np.linspace(0, natoms - 1, natoms).astype(int)
        ind += 1  # Lammps id start at 1
        # map type to int alphabetic order create a dictionary with atom id and mass, eg {'H' : {'ref': 1, 'mass' : 1.00}, 'Ni': {'ref' : 2, 'mass' : 58.69} }
        map_type = {
            atom_type: {"ref": i + 1, "mass": atomic_masses[atomic_numbers[atom_type]]}
            for i, atom_type in enumerate(sorted(set(types)))
        }
        types = [map_type[element]["ref"] for element in types]  # map to integer

        # lammps create system
        engine.command(
            "region box block 0.0 {} 0.0 {} 0.0 {}".format(xhi, yhi, zhi)
        )
        engine.command("create_box {} box".format(len(map_type)))
        engine.lmp.create_atoms(natoms, ind, types, x)
        # Set masses
        for key in map_type.keys():
            engine.command(
                "mass {} {}".format(map_type[key]["ref"], map_type[key]["mass"])
            )
        # Label atoms name to type :
        engine.command(
            "labelmap atom "
            + " ".join(f"{int(e['ref'])} {key}" for key, e in map_type.items())
        )

def initialize_potential(engine, config) :
    pair_style = config.lammps.pair_style
    pair_coeff = config.lammps.pair_coeff
    engine.command("pair_style {}".format(pair_style))
    engine.command("pair_coeff {}".format(pair_coeff))
    for cmd in config.lammps.setup_commands or []:
        engine.command(cmd)
    if config.otfml and config.otfml.enabled:
        ensure_otfml_dirs()
        dump_path = session_dump_path(engine.engine_id).as_posix()
        engine.command(f"variable {OTFML_TOL_FLAG_INTERNAL} internal 0")
        engine.command(f"variable {OTFML_MAX_FLAG_INTERNAL} internal 0")
        engine.command(f"variable {OTFML_TOL_FLAG_VARIABLE} equal v_{OTFML_TOL_FLAG_INTERNAL}")
        engine.command(f"variable {OTFML_MAX_FLAG_VARIABLE} equal v_{OTFML_MAX_FLAG_INTERNAL}")
        engine.command(f"fix extrapolation_grade all pair 1 mtp/extrapolation extrapolation 1")
        engine.command(f"compute max_grade all pair mtp/extrapolation")
        engine.command(f"variable max_grade equal c_max_grade[1]")
        engine.command(
            f'variable dump_skip equal "v_max_grade < {OTFML_EXTRAPOLATION_TOLERANCE:.4f}"'
        )
        engine.command(
            f"dump extrapolative_structures_dump all custom 1 {dump_path} id type x y z f_extrapolation_grade"
        )
        engine.command(f"dump_modify extrapolative_structures_dump append yes")
        engine.command(f"dump_modify extrapolative_structures_dump skip v_dump_skip")
        engine.command(
            f"fix extreme_extrapolation all halt 3 v_max_grade > {OTFML_MAX_GAMMA:.4f} error continue"
        )
        engine.command(f"thermo 1")
        engine.command(f"thermo_style custom step pe v_max_grade")


def reload_potential(engine, config):
    """Reload an updated potential without rebuilding the LAMMPS system."""
    engine.command("pair_style {}".format(config.lammps.pair_style))
    engine.command("run 0")


def reset_otf_flags(engine) -> None:
    """Reset the latched OTF extrapolation flags on the current engine."""
    if not hasattr(engine.lmp, "set_internal_variable"):
        raise RuntimeError(
            "LAMMPS Python module does not expose set_internal_variable(), "
            "which is required for OTFML flag handling."
        )
    if engine.lmp.set_internal_variable(OTFML_TOL_FLAG_INTERNAL, 0.0) != 0:
        raise RuntimeError(
            f"Failed to reset OTFML variable '{OTFML_TOL_FLAG_INTERNAL}'."
        )
    if engine.lmp.set_internal_variable(OTFML_MAX_FLAG_INTERNAL, 0.0) != 0:
        raise RuntimeError(
            f"Failed to reset OTFML variable '{OTFML_MAX_FLAG_INTERNAL}'."
        )


def get_otf_flags(engine) -> OTFExtrapolationFlags:
    """Read the current latched OTF extrapolation flags from the engine."""

    def extract_scalar(name: str) -> float:
        try:
            value = engine.lmp.extract_variable(name, None, 0)
        except TypeError:
            value = engine.lmp.extract_variable(name)
        return float(value)

    return OTFExtrapolationFlags(
        extrapolated=bool(extract_scalar(OTFML_TOL_FLAG_VARIABLE)),
        extreme_extrapolated=bool(extract_scalar(OTFML_MAX_FLAG_VARIABLE)),
    )


def _build_extrapolation_error(
    flags: OTFExtrapolationFlags,
    *,
    phase: str,
    message: str,
    variables: dict,
):
    if flags.extreme_extrapolated:
        return Err(
            ErrorInfo(
                type=ErrorType.EXTREME_EXTRAPOLATION,
                message=message,
                variables={"phase": phase, **variables},
            )
        )
    if flags.extrapolated:
        return Err(
            ErrorInfo(
                type=ErrorType.EXTRAPOLATION,
                message=message,
                variables={"phase": phase, **variables},
            )
        )
    return None


def minimize(engine, config, positions=None) :
    if positions is not None :
        set_positions(engine=engine, positions=positions)
    engine.command("min_style {}".format(config.lammps.min_style))
    engine.command("minimize {}".format(config.lammps.minimize))

def get_total_energy(engine, positions=None) :
    if positions is not None :
        set_positions(engine=engine, positions=positions)
    #Get total energy
    engine.command("run 0")
    result = engine.lmp.get_thermo("etotal")
    if engine.rank == 0 :
        return result

def get_potential_energy(engine, positions = None) :
    if positions is not None :
        set_positions(engine=engine, positions=positions)
    #get potential energy
    engine.command("compute c1 all pe")
    engine.command("run 0")
    result = engine.lmp.extract_compute("c1", 0, 0)
    engine.command("uncompute c1")
    return result

def get_positions(engine) :
    result = engine.lmp.gather_atoms("x", 1, 3)
    if engine.rank == 0:
        # convert ctype positions into a numpy array
        result = np.ctypeslib.as_array(result)
        result = np.reshape(result, (-1, 3))
        return result

def set_positions(engine, positions) :
    positions = positions.flatten().astype(np.float64)
    positions = np.ascontiguousarray(positions)
    c_array = (ctypes.c_double * len(positions))(*positions)
    engine.lmp.scatter_atoms("x", 1, 3, c_array)

def minimize_with_results(engine, config, positions=None) :
    """
    Minimize and return the minimized positions and the total energy.
    """
    if positions is not None :
        set_positions(engine=engine, positions=positions)
    minimize(engine, config)
    new_positions = get_positions(engine)
    total_energy = get_total_energy(engine)
    if engine.rank == 0 :
        return new_positions, total_energy

def minimize_freeze_core(engine, central_atom_positions: np.ndarray, rcut: float, maxiter:int = 10) :
    """
    Minimize with fix atom around central atom up to rcut
    """

    #define core region and group
    engine.command(f"region sphere_region sphere {central_atom_positions[0]} {central_atom_positions[1]} {central_atom_positions[2]} {rcut}")
    engine.command("group frozen_group region sphere_region")

    #freeze core region
    engine.command("fix freeze frozen_group setforce 0.0 0.0 0.0")

    #minimization
    engine.command(f"min_style cg")
    engine.command(f"minimize 1e-6 1e-8 {maxiter} {maxiter}")

    #unfreeze/delte
    engine.command("unfix freeze")
    engine.command("group frozen_group delete")
    engine.command("region sphere_region delete")

def partn_search(
    engine,
    config,
    central_atom_idx: int,
    positions=None,
    cell=None,
    type=None,
):
    original_stdout_fd = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    # Redirect stdout (fd 1) to /dev/null, only way to deal with pARTn error write
    os.dup2(devnull, 1)

    print('Central Atom', central_atom_idx)
    #Check to see if system is in AV mode:
    if config.control.active_volume == True:
        atom_map, central_lammps_id=partn_search_AV(engine, config, central_atom_idx, positions, cell, type)

    else:
        #Set positions
        atom_map = None
        central_lammps_id=[central_atom_idx+1]
        if positions is not None :
            set_positions(engine=engine, positions=positions)

    # PARAMETERS :
    delr_threshold = config.eventsearch.delr_thr

    # LAMMPS COMMANDS
    engine.command("plugin load {}".format(config.partn.path_artnso))
    engine.command("fix 10 all artn dmax {}".format(config.partn.dmax))
    engine.command("min_style fire")

    # INITILIZE ARTN on all ranks
    artn = pypARTn.artn(engine="lmp")
    # SETUP ARTN
    artn.reset_input()
    # Control
    artn.set("filout", "artn.out."+str(engine.engine_id))
    artn.set("engine_units", "lammps/metal")
    artn.set("verbose", config.partn.verbosity)
    artn.set("struc_format_out", "none")
    artn.set("delr_thr", config.partn.delr_thr)

    #Exploration
    artn.set("lpush_final", True)
    artn.set(
        "lmove_nextmin", False
    )  # if true fortran runtime error when event not found
    artn.set("zseed", config.partn.zseed)

    #Initial push
    artn.set("push_mode", config.partn.push_mode)
    if config.partn.push_mode == "rad":
        artn.set("push_dist_thr", config.partn.push_dist_thr)
    artn.set("push_step_size", config.partn.push_step_size)
    artn.set("push_ids", central_lammps_id)
    artn.set("ninit", config.partn.ninit)

    #Lanczos
    artn.set("lanczos_min_size", config.partn.lanczos_min_size)
    artn.set("lanczos_max_size", config.partn.lanczos_max_size)
    artn.set("lanczos_disp", config.partn.lanczos_disp)
    artn.set("lanczos_eval_conv_thr", config.partn.lanczos_eval_conv_thr)

    #Eigenvector push
    artn.set("eigval_thr", config.partn.eigval_thr)
    artn.set("eigen_step_size", config.partn.eigen_step_size)
    artn.set("nsmooth", config.partn.nsmooth)
    artn.set("neigen", config.partn.neigen)
    artn.set("alpha_mix_cr", config.partn.alpha_mix_cr)
    artn.set("nnewchance", config.partn.nnewchance)

    #Perpendicular relaxation
    if config.partn.nperp is not None :
        artn.set("nperp", config.partn.nperp)
    if config.partn.nperp_limitation is not None :
        artn.set("nperp_limitation", np.array(config.partn.nperp_limitation))
    else :
        artn.set("lnperp_limitation", False)

    #Convergence
    artn.set("forc_thr", config.partn.forc_thr)

    #Final push
    artn.set("push_over", config.partn.push_over)

    # RUN
    engine.command(f"minimize 1e-6 1e-8 10000 {config.partn.evalf_max}")
    engine.command("unfix 10")

    # Restore original stdout (fd 1)
    os.dup2(original_stdout_fd, 1)
    os.close(original_stdout_fd)
    os.close(devnull)


    # EXTRACT DATA
    if engine.rank == 0 :
        extrapolation_error = _build_extrapolation_error(
            get_otf_flags(engine),
            phase="search",
            message="Search extrapolated and must be retried.",
            variables={
                "central_atom_index": central_atom_idx,
            },
        )
        if extrapolation_error is not None:
            return extrapolation_error

        err = artn.get_error()
        if err[0]==0:
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

                if config.control.active_volume==True:
                    min1positions, min2positions, saddlepositions, index_move= position_results_AV(config, artn, atom_map, positions)
                else:
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

def partn_refine(
    engine,
    config,
    central_atom_idx: int,
    positions=None,
    cell=None,
    type=None,
    saddle_idx=None,
    saddle_positions=None,
    minimize_outter_atoms: bool = True,
    num_reference_event: int | None = None,
    symmetry_index: int | None = None,
):

    #Set positions
    if config.control.active_volume==True:
        E_init, atom_map, central_lammps_id = partn_refine_AV(engine, config, central_atom_idx, positions, cell, type, saddle_idx, saddle_positions)
    else:
        central_lammps_id=[central_atom_idx+1]
        E_init=0
        atom_map=None
        if positions is not None :
            set_positions(engine=engine, positions=positions)
        #small minimization with fix core atoms around central atom
            if minimize_outter_atoms :
                minimize_freeze_core(engine, positions[central_atom_idx], config.atomicenvironment.rcut, maxiter = 10)

    # INITILIZE ARTN
    artn = pypARTn.artn(engine="lmp")
    # LAMMPS COMMANDS
    engine.command("plugin load {}".format(config.partn.path_artnso))

    # SETUP ARTN
    artn.reset_input()
    #Control
    artn.set("filout", "artn.out."+str(engine.engine_id))
    artn.set("engine_units", "lammps/metal")
    artn.set("verbose", config.partn.verbosity)
    artn.set("struc_format_out", "none")
    artn.set("delr_thr", config.partn.delr_thr)

    #Exploration
    artn.set("lpush_final", False)
    artn.set(
        "lmove_nextmin", False
    )  # if true fortran runtime error when event not found
    artn.set("zseed", config.partn.zseed)

    #Initial push : Should not happen when refining
    artn.set("push_mode", config.partn.r_push_mode)
    if config.partn.push_mode == "rad":
        artn.set("push_dist_thr", config.partn.r_push_dist_thr)
    artn.set("push_step_size", config.partn.r_push_step_size)
    artn.set("push_ids", central_lammps_id) #fortran start at 1
    artn.set("ninit", config.partn.r_ninit)

    #Lanczos
    artn.set("lanczos_min_size", config.partn.r_lanczos_min_size)
    artn.set("lanczos_max_size", config.partn.r_lanczos_max_size)
    artn.set("lanczos_disp", config.partn.r_lanczos_disp)
    artn.set("lanczos_eval_conv_thr", config.partn.r_lanczos_eval_conv_thr)

    #Eigenvector push
    artn.set("eigval_thr", config.partn.r_eigval_thr)
    artn.set("eigen_step_size", config.partn.r_eigen_step_size)
    artn.set("nsmooth", config.partn.r_nsmooth)
    artn.set("neigen", config.partn.r_neigen)
    artn.set("alpha_mix_cr", config.partn.r_alpha_mix_cr)
    artn.set("nnewchance", config.partn.r_nnewchance)

       #Perpendicular relaxation
    if config.partn.r_nperp is not None :
        artn.set("nperp", config.partn.r_nperp)
    if config.partn.r_nperp_limitation is not None :
        artn.set("nperp_limitation", np.array(config.partn.r_nperp_limitation))
    else :
        artn.set("lnperp_limitation", False)

    #Convergence
    artn.set("forc_thr", config.partn.r_forc_thr)



    #MAX attempt based on delr_sad (from initial position)
    #Fix that sometime, we go back to the minimum, so saddle point found is the minimum
    #When using a different seed it solves the problem

    max_attempts = config.partn.r_max_attempts
    inner_attempt = 0

    while inner_attempt < max_attempts :
        exit_flag = False
        result = None #for rank > 0
        engine.command("fix 10 all artn dmax {}".format(config.partn.r_dmax))
        engine.command("min_style fire")
                # RUN
        engine.command(f"minimize 1e-6 1e-8 10000 {config.partn.r_evalf_max}")
        engine.command("unfix 10")

        # EXTRACT DATA
        if engine.rank == 0 :
            extrapolation_error = _build_extrapolation_error(
                get_otf_flags(engine),
                phase="refine",
                message="Refinement extrapolated and must be retried.",
                variables={
                    "central_atom_index": central_atom_idx,
                    "num_reference_event": num_reference_event,
                    "symmetry_index": symmetry_index,
                },
            )
            if extrapolation_error is not None:
                exit_flag = True
                result = extrapolation_error
            else:
                err = artn.get_error()
                if err[0]==0: #No error
                    #Check if went back to minimum
                    delr_sad = artn.extract("delr_sad")
                    if delr_sad < config.partn.r_delr_sad_thr : #Success

                        E_sad = artn.extract("etot_sad")
                        E_result=E_sad-E_init #If AV's are on, will return the activation energy of the event. If not, jsut saddle energy
                        saddlepositions = artn.extract("tau_sad")

                        if config.control.active_volume==True:
                            saddlepositions_results=positions.copy()
                            for i, atom_idx in enumerate(atom_map):
                                saddlepositions_results[atom_idx][0] = saddlepositions[i][0]
                                saddlepositions_results[atom_idx][1] = saddlepositions[i][1]
                                saddlepositions_results[atom_idx][2] = saddlepositions[i][2]
                        else:
                            saddlepositions_results=saddlepositions

                        exit_flag = True
                        result =  Ok(
                            EventRefinementOutput(
                                central_atom_index=central_atom_idx,
                                saddle_positions=saddlepositions_results,
                                E_saddle= E_result,
                                num_reference_event=num_reference_event,
                                symmetry_index=symmetry_index,
                                refined='T'
                            )
                        )
        # Synchronize all ranks
        exit_flag = engine.local_engine_comm.bcast(exit_flag, root=0)
        if exit_flag :
            return result

        inner_attempt +=1
        artn.set("zseed", config.partn.zseed)

    else: #fail after max attemps
        if engine.rank == 0 :
            err = artn.get_error()
            return Err(
                ErrorInfo(
                    type=ErrorType.EVENT_NOT_FOUND, message="no event found", details=err
                )
            )
        return None
