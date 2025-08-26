import numpy as np
from ase.data import atomic_numbers, atomic_masses
from mpi4py import MPI
import ctypes 
import pypARTn2

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


def minimize(engine, config) : 
    engine.command("min_style {}".format(config.lammps.min_style))
    engine.command("minimize {}".format(config.lammps.minimize))

def get_total_energy(engine) : 
    #Get total energy
    result = engine.lmp.get_thermo("etotal")
    return result

def get_positions(engine) : 
    result = engine.lmp.gather_atoms("x", 1, 3)
    if engine.rank == 0:
        # convert ctype positions into a numpy array
        result = np.ctypeslib.as_array(result)
        print("GET POS")
        print(result)
        result = np.reshape(result, (-1, 3))
        return result
    
def set_positions(engine, positions) : 
    positions = positions.flatten().astype(np.float64)
    positions = np.ascontiguousarray(positions)
    c_array = (ctypes.c_double * len(positions))(*positions)
    engine.lmp.scatter_atoms("x", 1, 3, c_array)


def partn_search(engine, config, central_atom_idx: int) : 
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
    artn.set("push_ids", [central_atom_idx + 1])
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

    