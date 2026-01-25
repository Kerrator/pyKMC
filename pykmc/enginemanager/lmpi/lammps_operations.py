import numpy as np
from ase.geometry import find_mic
from ase.data import atomic_numbers, atomic_masses
from mpi4py import MPI
import pypARTn
import ctypes
import os
import sys

from ...result import  (
    Result,
    ErrorInfo,
    EventSearchOutput,
    Ok,
    Err,
    ErrorType,
    EventRefinementOutput,
)

from ase import Atoms
from ase.io import write



def initialize_parameters(engine) : 
    engine.command("units metal")
    engine.command("atom_style atomic")
    engine.command("dimension 3")
    engine.command("boundary p p p")
    engine.command("atom_modify map array") #! necessary for scatter atoms
    #engine.command("atom_modify sort 0 0.0") #! necessary for partn
    #engine.command("processors 1 1 *")

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

# def get_positions(engine) :
#     result = engine.lmp.gather_atoms("x", 1, 3)
#     if engine.rank == 0:
#         # convert ctype positions into a numpy array
#         result = np.ctypeslib.as_array(result)
#         result = np.reshape(result, (-1, 3))
#         return result

def get_positions(engine) :
    x = engine.lmp.gather_atoms("x", 1, 3)
    #ids = engine.lmp.gather_atoms("id", 0, 1)
    if engine.rank == 0:
        # convert ctype positions into a numpy array
        x = np.ctypeslib.as_array(x)
        x = np.reshape(x, (-1, 3))
        return x
    
def set_positions(engine, positions) : 
    positions = positions.flatten().astype(np.float64)
    positions = np.ascontiguousarray(positions)
    c_array = (ctypes.c_double * len(positions))(*positions)
    engine.lmp.scatter_atoms("x", 1, 3, c_array)


def minimize_with_results(engine, config, positions=None, cell=None) :
    """ 
    Minimize and return the minimized positions and the total energy.
    """
    reset(engine, config, cell)
    if positions is not None :
        redefine_atoms(engine=engine, positions=positions)

    minimize(engine, config)
    new_positions = get_positions(engine)
    total_energy = get_total_energy(engine)
    if engine.rank == 0 :
        return new_positions, total_energy


def partn_search(engine, config, central_atom_idx: int, cell, positions=None) :

    # original_stdout_fd = os.dup(1)
    # devnull = os.open(os.devnull, os.O_WRONLY)
    # # Redirect stdout (fd 1) to /dev/null, only way to deal with pARTn error write
    # os.dup2(devnull, 1)

    #engine.command('plugin clear')
    reset(engine, config, cell)

    #Define active volume
    av_positions, av_idx, buffer_idx = define_active_volume(config, central_atom_idx, cell, positions)

    #-------------------------STEP 1: CREATE ATOMS------------------------
    # Update lammps to only have these atoms
    if av_positions is not None:
        redefine_atoms(engine=engine, positions=av_positions)

    #Define map based on past atom positions and the new index
    atom_map = make_active_volume(engine, av_positions, av_idx, buffer_idx) #Issues in atom_map

    # INITILIZE ARTN on all ranks
    artn = pypARTn.artn(engine="lmp")

    delr_threshold = config.eventsearch.delr_thr


    engine.command("fix 10 all artn dmax {}".format(config.partn.dmax))
    engine.command("min_style fire")

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
    artn.set("push_ids", (np.where(atom_map == central_atom_idx)[0]+1))
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

    engine.command('run 0')
    engine.command("minimize 1e-6 1e-8 1000 1000")

    #Prep for next run
    engine.command("unfix 10")
    engine.command("unfix f_buffer")

    # #Restore original stdout (fd 1)
    # os.dup2(original_stdout_fd, 1)
    # os.close(original_stdout_fd)
    # os.close(devnull)


    # EXTRACT DATA
    if engine.rank == 0 :
        ierr = artn.get_error()
        if ierr[0]==0:

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

                index_move_mapped=atom_map[index_move]

                min1positions_mapped = positions.copy()
                min2positions_mapped = positions.copy()
                saddlepositions_mapped = positions.copy()

                for i, atom_idx in enumerate(atom_map):
                    saddlepositions_mapped[atom_idx][0] = saddlepositions[i][0]
                    saddlepositions_mapped[atom_idx][1] = saddlepositions[i][1]
                    saddlepositions_mapped[atom_idx][2] = saddlepositions[i][2]

                    min1positions_mapped[atom_idx][0] = min1positions[i][0]
                    min1positions_mapped[atom_idx][1] = min1positions[i][1]
                    min1positions_mapped[atom_idx][2] = min1positions[i][2]

                    min2positions_mapped[atom_idx][0] = min2positions[i][0]
                    min2positions_mapped[atom_idx][1] = min2positions[i][1]
                    min2positions_mapped[atom_idx][2] = min2positions[i][2]

                if delr1 < delr2:  # necessary for no reconstruction option
                    return Ok(
                        EventSearchOutput(
                            central_atom_index=central_atom_idx,
                            dE_forward=dE_forward,
                            dE_backward=dE_backward,
                            min1_positions=min1positions_mapped,
                            saddle_positions=saddlepositions_mapped,
                            min2_positions=min2positions_mapped,
                            move_atom_index=index_move_mapped,
                        )
                    )
                else:
                    return Ok(
                        EventSearchOutput(
                            central_atom_index=central_atom_idx,
                            dE_forward=dE_backward,
                            dE_backward=dE_forward,
                            min1_positions=min2positions_mapped,
                            saddle_positions=saddlepositions_mapped,
                            min2_positions=min1positions_mapped,
                            move_atom_index=index_move_mapped,
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
                    type=ErrorType.EVENT_NOT_FOUND, message="No event found", details=ierr
                )
            )

def partn_refine(engine, config, central_atom_idx:int, cell, positions, saddle_positions, saddle_idx) :

    '''
    Receive the system with the central atom index, define an active volume around this atom, then update the positions
    with those for the saddle.

    This was added in order to get the activation energy for an event, as the traditional method does not work for
    Active Volumes.
    '''
    #engine.command('plugin clear')

    # if engine.rank==0:
    #     print('Neighbors', saddle_idx)
    reset(engine, config, cell)

    #Define active volume
    av_positions, av_idx, buffer_idx = define_active_volume(config, central_atom_idx, cell, positions)


    # Update lammps to only have these atoms
    if positions is not None:
        redefine_atoms(engine=engine, positions=av_positions)
    # Define map based on past atom positions and the new index
    atom_map = make_active_volume(engine, av_positions, av_idx, buffer_idx)

    engine.command('write_dump all atom pre_core.dump')

    #Now get Active Volume energy to store for later
    E_init=get_potential_energy(engine)

    #Now update positions of atoms in lammps, only those needed for the refinement
    core_idx=[]
    core_ids=[]


    for i, atom_idx in enumerate(saddle_idx):
        index=int(np.where(atom_map == atom_idx)[0]) #index in atom map where this value is true
        av_positions[index]=saddle_positions[i]
        core_idx.append(index) #Atom id
        core_ids.append(index+1)

    #Now we have the atoms at the saddle point
    set_positions(engine, av_positions)

    engine.command('fix 1 all setforce 0.0 0.0 0.0')
    engine.command('run 0')
    engine.command('unfix 1')

    engine.command('write_dump all atom post_core.dump')
    #Want to minimize initially to speed up refinement process
    engine.command(f"group core id {' '.join(map(str, core_ids))}")
    engine.command('fix f_core core setforce 0.0 0.0 0.0')
    engine.command("min_style {}".format(config.lammps.min_style))
    engine.command("minimize 1.0e-6 1.0e-8 10 10")
    engine.command('unfix f_core')

    # INITILIZE ARTN
    artn = pypARTn.artn(engine="lmp")

    engine.command("plugin load {}".format(config.partn.path_artnso))
    # LAMMPS COMMANDS
    engine.command("fix 10 all artn dmax {}".format(config.partn.r_dmax))
    engine.command("min_style fire")


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

    artn.set("push_ids", (np.where(atom_map == central_atom_idx)[0]+1))
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
    engine.command("unfix 10")
    #Otherwise, if you use other minimization style for the system minimization error "min_style fire
    # must be use with fix ART"

    engine.command("unfix f_buffer")

    if engine.rank == 0 :
        ierr = artn.get_error()
        if ierr[0] == 0:
            E_sad = artn.extract("etot_sad")

            E_diff=E_sad-E_init

            saddlepositions = artn.extract("tau_sad")
            saddlepositions_mapped = positions.copy()
            for i, atom_idx in enumerate(atom_map):
                saddlepositions_mapped[atom_idx][0] = saddlepositions[i][0]
                saddlepositions_mapped[atom_idx][1] = saddlepositions[i][1]
                saddlepositions_mapped[atom_idx][2] = saddlepositions[i][2]

            return Ok(
                EventRefinementOutput(
                    central_atom_index=central_atom_idx,
                    saddle_positions=saddlepositions_mapped,
                    E_saddle=E_diff,
                )
            )
        else:
            return Err(
                ErrorInfo(
                    type=ErrorType.EVENT_NOT_FOUND, message="no event found", details=ierr,
                )
            )


#----------------------------------NEW CODE-----------------------------------------

def define_active_volume(config, central_atom_idx: int, cell, positions):
    '''
    Defines which atoms are within the Active Volume and which ones can move. Does this independent of lammps

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

    center = positions[central_atom_idx]

    inner_movable_idx = []
    buffer_idx = []
    total_active_idx = []
    non_active_idx = []



    for i, pos in enumerate(positions):
        diff = pos - center
        diff_mic, distance = find_mic(diff, cell, pbc=True)
        if np.abs(distance) <= r_m:
            inner_movable_idx.append(i)
            total_active_idx.append(i)  # Inner is also part of total active
        elif np.abs(distance) > r_m and np.abs(distance) <= r_a: #Can change to make it between r_m and r_a
            buffer_idx.append(i)
            total_active_idx.append(i)
        else:
            non_active_idx.append(i)


    buffer_idx = np.array(sorted(buffer_idx))
    av_idx = np.array(sorted(total_active_idx))

    av_positions=positions[av_idx]

    return av_positions, av_idx, buffer_idx


def make_active_volume(engine, av_positions, av_indices, buffer_indices):
    #engine.command('reset_atoms id')

    # Since we used engine.lmp.create_atoms(len(positions), ... x=av_positions)
    # The LAMMPS ID 'i+1' is exactly the atom at av_positions[i].
    # Therefore, the map is simply the indices we used to slice the original system.
    atom_map = np.array(av_indices, dtype=int)

    # Define the buffer group based on the new LAMMPS IDs
    # We need to find which index in 'av_indices' corresponds to 'buffer_indices'
    engine_buffer_ids = []
    buffer_set = set(buffer_indices)
    for i, original_id in enumerate(av_indices):
        if original_id in buffer_set:
            engine_buffer_ids.append(i + 1)  # LAMMPS IDs are 1-based

    if engine_buffer_ids:
        engine.command(f"group buffer id {' '.join(map(str, engine_buffer_ids))}")
        engine.command("fix f_buffer buffer setforce 0.0 0.0 0.0")
    else:
        engine.command(f"group buffer empty")
        engine.command("fix f_buffer buffer setforce 0.0 0.0 0.0")
        print('No buffer atoms defined')

    engine.command('run 0 post no')

    return atom_map


# def make_active_volume(engine, av_positions, av_indices, buffer_indices) :
#     '''
#     Makes active volume within lammps based on what is defined
#
#     Parameters
#         -----------
#         engine: engine
#             What engine instance is being used
#         av_positions: np.array
#             Array of the positions defined within the active volume with PBC
#         av_indices: np.array
#             Indices of all the atoms defined in the active volume in old system indexing
#         buffer_indices: np.array
#             Atoms that are on the outskirts of the active volume which are unable to move
#     '''
#     # ------------------Now want refactored atoms-----------------------------
#     engine.command('reset_atoms id')
#     engine.command("atom_modify sort 0 0.0")  # ! necessary for partn
#     # ---------------------------Now Map-------------------------------
#
#     engine_positions = get_positions(engine) #This line is the issues
#     #print('Broadcaasting positions')
#     # engine_positions= engine.engine_comm.bcast(engine_positions, root=0)
#     #print(f"[rank {engine.rank}] positions = {engine_positions}", flush=True)
#     if engine.rank == 0 :
#         atom_map = map_atom_indices(av_positions, av_indices, engine_positions)  # This will make the map needed for pARTn
#         atom_map = np.asarray(atom_map, dtype=int)
#
#         if atom_map.ndim != 1:
#             raise RuntimeError(f"atom_map has unexpected shape: {atom_map.shape}")
#         # No zero or negative IDs allowed
#         if np.any(atom_map <= 0):
#             raise RuntimeError(f"Invalid IDs in atom_map (<=0): {atom_map}")
#     else:
#         atom_map = None
#
#     atom_map= engine.engine_comm.bcast(atom_map, root=0)
#     engine_av_ids = []
#     engine_buffer_ids = []
#     # Convert target arrays to sets for efficient O(1) average time lookup
#
#     # atom_map is an array of old IDs (1-based). Make a 1-based set for comparison.
#     print('Making atom map: 3')
#     av_set = set(av_indices)          # now 1-based
#     buffer_set = set(buffer_indices)  # now 1-based
#
#     for new_id_zero_based, old_id in enumerate(atom_map):
#         if old_id in av_set:
#             engine_av_ids.append(new_id_zero_based + 1)   # new LAMMPS IDs are 1-based
#         if old_id in buffer_set:
#             engine_buffer_ids.append(new_id_zero_based + 1)
#
#     if len(engine_buffer_ids) > 0:
#         engine.command(f"group buffer id {' '.join(map(str, engine_buffer_ids))}")
#         engine.command("fix f_buffer buffer setforce 0.0 0.0 0.0")
#     else:
#         engine.command(f"group buffer empty")
#         engine.command("fix f_buffer buffer setforce 0.0 0.0 0.0")
#         print('No buffer atoms defined')
#
#     engine.command('run 0 post no')
#     #print('atoms defined')
#     return atom_map


def map_atom_indices(positions, idx, engine_positions) :
    """
    Create a mapping from new IDs to old system id's using the positions

    Parameters
        ----------
        positions : np.array
            Positions of all the atoms in the active volume
        idx : np.array
            Array of index's from the old system which are in the active volume
        engine_positions: list
            Array of all the positions currently loaded in lammps, with ID from 1->N atoms

    Returns:
        ---------
        new_to_old (dict): {new_ID: old_ID}
    """
    old_IDs = idx
    old_positions = positions

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

def redefine_atoms(engine, positions) -> None:
    '''
    Check to see if current lammps system has enough atoms
    If not, deletes all atoms then redefines them
    '''

    # if hasattr(engine, 'engine_comm'):
    #     engine.engine_comm.Barrier()

    #num_atoms = engine.lmp.get_natoms()
    # if len(positions) == num_atoms: # Enough atoms in system
    #     set_positions(engine, positions)
    #     engine.command('fix 1 all setforce 0.0 0.0 0.0')
    #     engine.command('run 0')
    #     engine.command('unfix 1')
    # else: #Uneven number of atoms
        # if num_atoms!=0:
        #     engine.command('delete_atoms group all')
    type=[1]*len(positions)
    #print('Creating atoms for refinement atoms')
    new_positions = positions.flatten().astype(np.float64)
    ids = np.arange(1, len(positions)+1, dtype=np.int32)
    engine.lmp.create_atoms(len(positions),ids,type,x=new_positions)
    engine.command("comm_style tiled")
    engine.command("balance 1.1 rcb")
    engine.command("neigh_modify every 1 delay 0 check yes")
    engine.command('fix 1 all setforce 0.0 0.0 0.0')
    engine.command('run 0')
    engine.command('unfix 1')

    # if hasattr(engine, 'engine_comm'):
    #     engine.engine_comm.Barrier()

def reset(engine, config, cell) -> None:
    """
    Clear lammps instance, preps it for the new sim:
    """

    engine.command('clear')
    initialize_parameters(engine)
    # Create cell
    xhi, yhi, zhi = cell[0][0], cell[1, 1], cell[2, 2]
    engine.command(
        "region box block 0.0 {} 0.0 {} 0.0 {}".format(xhi, yhi, zhi)
    )
    engine.command('create_box 1 box')  # NEEDS TO BE UPDATED FOR ALLOYS
    initialize_potential(engine, config)

def clear(engine):
    '''
    Clears lammps instance
    '''
    engine.command('clear')

# def redefine_atoms(engine, positions) -> None:
#     '''
#     Redistibutes atoms in current system. Adds/removes needed atoms to match what's being sent, then defines the system
#     '''
#     num_atoms = engine.lmp.get_natoms()
#     print('Number of atoms in system before check', num_atoms)
#     if len(positions) < num_atoms: #Too many atoms in system
#         print('Too many atoms, deleting')
#         engine.command('group delete_group id {}:{}'.format(len(positions)+1, num_atoms))
#         engine.command('delete_atoms group delete_group')
#     elif len(positions) > num_atoms: #Not enough atoms in system
#         print('Not enough atoms, making')
#         engine.command('create_atoms 1 random {} 1234 NULL'.format(len(positions)-num_atoms))
#     engine.command('reset_atoms id')
#     num_atoms = engine.lmp.get_natoms()
#     #print('Number of atoms in Lammps', num_atoms)
#     set_positions(engine, positions)
#     engine.command('fix 1 all setforce 0.0 0.0 0.0')
#     #engine.command('neigh_modify every 1 delay 0 check yes')
#     engine.command('run 0')
#     #niegh_modify (
#     engine.command('unfix 1')

