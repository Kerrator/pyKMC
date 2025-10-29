import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, atomic_masses
from mpi4py import MPI
import ctypes 
import pypARTn2
import os

from ...result import  (
    Result,
    ErrorInfo,
    EventSearchOutput,
    Ok,
    Err,
    ErrorType,
    EventRefinementOutput,
)
from ...system import (System)


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


def minimize(engine, config) : 
    engine.command("min_style {}".format(config.lammps.min_style))
    engine.command("minimize {}".format(config.lammps.minimize))

def get_total_energy(engine) : 
    #Get total energy
    result = engine.lmp.get_thermo("etotal")
    return result

def get_potential_energy(engine) : 
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


def partn_search(engine, config, central_atom_idx: int, system) :
    # original_stdout_fd = os.dup(1)
    # devnull = os.open(os.devnull, os.O_WRONLY)
    # # Redirect stdout (fd 1) to /dev/null, only way to deal with pARTn error write
    # os.dup2(devnull, 1)

    engine.command('plugin clear')
    engine.command('clear')

    #Define active volume:
    av, inner_ind, buffer_ind = define_active_volume(config, central_atom_idx, system)
    atom_map = make_active_volume(engine, config, av, inner_ind, buffer_ind)

    # PARAMETERS :
    delr_threshold = config.eventsearch.delr_thr

    # LAMMPS COMMANDS
    engine.command("plugin load {}".format(config.partn.path_artnso))
    engine.command("fix 10 all artn dmax {}".format(config.partn.dmax))
    engine.command("min_style fire")

    # INITILIZE ARTN on all ranks
    artn = pypARTn2.artn(engine="lmp")
    # SETUP ARTN
    artn.reset_input()
    # Control 
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

    artn.set("push_ids", (np.where(atom_map == central_atom_idx+1)[0]+1))
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
    artn.set("nperp", config.partn.nperp)

    #Convergence
    artn.set("forc_thr", config.partn.forc_thr)

    #Final push
    artn.set("push_over", config.partn.push_over)

    # RUN
    engine.command("minimize 1e-6 1e-8 1000 1000")

    engine.command("unfix f_buffer")
    engine.command("group inner_movable delete")
    engine.command("group buffer delete")

    # Restore original stdout (fd 1)
    #os.dup2(original_stdout_fd, 1)
    #os.close(original_stdout_fd)
    #os.close(devnull)


    # EXTRACT DATA
    if engine.rank == 0 :
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
                            central_atom_index=int(np.where(atom_map == central_atom_idx+1)[0]),,
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
                            central_atom_index=int(np.where(atom_map == central_atom_idx+1)[0]),,
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

def partn_refine(engine, config, central_atom_idx:int, system) :


    engine.command('plugin clear')
    engine.command('clear')

    # Define active volume:
    av, inner_ind, buffer_ind = define_active_volume(config, central_atom_idx, system)
    atom_map = make_active_volume(engine, config, av, inner_ind, buffer_ind)
    #map = make_active_volume(engine, av, inner_ind, buffer_ind)

    # LAMMPS COMMANDS
    engine.command("plugin load {}".format(config.partn.path_artnso))
    engine.command("fix 10 all artn dmax {}".format(config.partn.r_dmax))
    engine.command("min_style fire")

    # INITILIZE ARTN
    artn = pypARTn2.artn(engine="lmp")

    # SETUP ARTN
    artn.reset_input()
    #Control
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

    print('--------CENTRAL ATOM INDEX--------', central_atom_idx)
    print('-------Mapped Central Index',np.where(atom_map == central_atom_idx+1)[0]+1)

    artn.set("push_ids", (np.where(atom_map == central_atom_idx+1)[0]+1))
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
    artn.set("nperp", config.partn.r_nperp)
    artn.set("nperp_limitation", [200])

    #Convergence
    artn.set("forc_thr", config.partn.r_forc_thr)





    # RUN
    engine.command("minimize 1e-6 1e-8 1000 1000")
    engine.command("unfix 10") #Otherwise, if you use other minimization style for the system minimization error "min_style fire must be use with fix ART"


    engine.command("unfix f_buffer")
    engine.command("group inner_movable delete")
    engine.command("group buffer delete")

    if engine.rank == 0 :
        err = artn.get_runparam("error_message")
        if not err:
            E_sad = artn.extract("etot_sad")
            saddlepositions = artn.extract("tau_sad")
            return Ok(
                EventRefinementOutput(
                    central_atom_index=central_atom_idx,
                    saddle_positions=saddlepositions,
                    E_saddle= E_sad
                )
            )

    else:
        return Err(
            ErrorInfo(
                type=ErrorType.EVENT_NOT_FOUND, message="no event found", details=err
            )
        )


#----------------------------------NEW CODE-----------------------------------------

def define_active_volume(config, central_atom_idx: int, system):
    '''
    Want to make it so the radius and such are defined in the config file later
    Need to pass the ID of the atom getting checked. It will be in ORIGINAL system units, so it'll need to be mapped

        Parameters
    ----------
    radius: float
        Radius of active volume in latsize units (can change to angstrom)
    ratio: float
        Ratio of the active volume that is allowed to move
    central_atom_idx : int
        The central atom index.
    '''

    # Defining parameters
    # Radius of whole active volume in Ang
    r_a = config.partn.r_act # Ensure AV is larger than topology analysis
    # Defines the radius of atoms that can move.
    r_m = config.partn.r_mov

    #NEED TO ADD WARNING IF R_A<R_M

    positions = system.positions

    center = positions[central_atom_idx]

    inner_movable_indices = []
    buffer_indices = []
    total_active_indices = []
    non_active_indices = []

    for i, pos in enumerate(positions):
        diff = pos - center
        diff_mic, distance = find_mic(diff, system.cell, pbc=True)
        if np.abs(distance) <= r_m:
            inner_movable_indices.append(i)
            total_active_indices.append(i)  # Inner is also part of total active
        elif np.abs(distance) > r_m and np.abs(distance) <= r_a: #Can change to make it between r_m and r_a
            buffer_indices.append(i)
            total_active_indices.append(i)
        else:
            non_active_indices.append(i)

    print("Number of atoms in Active Volume: {}".format(len(total_active_indices)))

    #buffer_indices = np.array(list(set(total_active_indices) - set(inner_movable_indices)))
    inner_movable_indices = np.array(sorted(inner_movable_indices))  # Ensure sorted for consistent indexing
    buffer_indices = np.array(sorted(buffer_indices))
    non_active_indices = np.array(sorted(non_active_indices))
    av_indices = np.array(sorted(total_active_indices))
    #av_indices = np.sort(np.concatenate([inner_movable_indices, buffer_indices]))  # Sorts all the atoms

    cell = system.cell
    types = [system.types[i] for i in av_indices]
    positions = system.positions[av_indices]
    pbc = system.pbc

    av = System(types, positions, cell, pbc, av_indices)
    return av, inner_movable_indices, buffer_indices


def make_active_volume(engine, config, system, inner_movable_indices, buffer_indices) :
    '''
    Makes the active volume in lammps. This is where the map for ID's gets made
    '''

    # -------------------Setting up AV----------------------

    initialize_parameters(engine)
    initialize_system(engine, system)  # This encompasses making the av system
    initialize_potential(engine, config)

    # ------------------Now want refactored atoms-----------------------------
    engine.command('reset_atoms id')
    # ---------------------------Now Map-------------------------------
    engine.command("atom_modify sort 0 0.0")  # ! necessary for partn
    engine_positions = get_positions(engine)

    atom_map = map_atom_indices(system, engine_positions)  # This will make the map needed for pARTn

    engine_inner_movable_ids = []
    engine_buffer_ids = []
    # Convert target arrays to sets for efficient O(1) average time lookup

    inner_movable_set = set(inner_movable_indices+1)
    buffer_set = set(buffer_indices+1)

    for i in range(len(atom_map)):
        # self.map[i] represents an original ID
        # Check if this original ID is present in the set of inner_movable_indices
        if atom_map[i] in inner_movable_set:
            engine_inner_movable_ids.append(i + 1)  # i is the 0-based new_ID, +1 if LAMMPS IDs are 1-based

        # Check if this original ID is present in the set of buffer_indices
        if atom_map[i] in buffer_set:
            engine_buffer_ids.append(i + 1)  # i is the 0-based new_ID, +1 if LAMMPS IDs are 1-based
    #
    # #Defining groups of atoms
    #
    engine.command(f"group inner_movable id {' '.join(map(str, engine_inner_movable_ids))}")
    if len(engine_buffer_ids) > 0:
        engine.command(f"group buffer id {' '.join(map(str, engine_buffer_ids))}")
        engine.command("fix f_buffer buffer setforce 0.0 0.0 0.0")

    else:
        print('No buffer atoms defined')
    engine.command('run 0')
    engine.command('write_dump all custom dump.atom id type x y z fx fy fz')
    engine.command('write_dump buffer custom dump.buffer id type x y z fx fy fz')
    engine.command('write_dump inner_movable custom dump.active id type x y z fx fy fz')

    return atom_map


def map_atom_indices(system, engine_positions) :
    """
    Create a mapping from new IDs (after 'reset_ids all compress yes')
    back to original IDs (before reset) using positions.

    Parameters:
        combined_array_0 (ndarray): N x 4 array [old_ID, x, y, z] before reset
        atom_positions (ndarray): N x 3 array [x, y, z] after reset (ordered by new_ID)

    Returns:
        new_to_old (dict): {new_ID: old_ID}
    """
    old_IDs = system.index+1
    old_positions = system.positions

    # Round positions to avoid floating-point mismatch issues
    precision = 12
    old_positions_rounded = np.round(old_positions, precision)
    atom_positions_rounded = np.round(engine_positions, precision)

    # Convert positions to structured array for fast set lookups
    dtype = [('x', float), ('y', float), ('z', float)]

    old_struct = old_positions_rounded.copy(order='C').view(dtype).squeeze()
    new_struct = atom_positions_rounded.copy(order='C').view(dtype).squeeze()

    # Build a dictionary from position → old ID
    pos_to_old = {tuple(p): id_ for p, id_ in zip(old_struct, old_IDs)}

    # Map each new position to old ID
    new_to_old = np.array([pos_to_old[tuple(p)] for p in new_struct], dtype=int)

    return new_to_old


# def shift_atoms(engine) -> None:
#     positions= get_positions(engine)
#
#     for i in range(len(positions)):
#         self.og_system.positions[self.map[i]] = self.lmp_new_positions[i]