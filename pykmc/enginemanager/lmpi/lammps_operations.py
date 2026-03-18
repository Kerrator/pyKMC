import logging
import numpy as np
from ase.data import atomic_numbers, atomic_masses
from mpi4py import MPI
import ctypes 
import pypARTn
import os
from ...activevolume.active_volume import reset, redefine_atoms, partn_search_AV, partn_refine_AV, position_results_AV

from ...result import  (
    Result,
    ErrorInfo,
    EventSearchOutput,
    Ok,
    Err,
    ErrorType,
    EventRefinementOutput,
)

logger = logging.getLogger("log")


def initialize_parameters(engine, boundary="p p p") :
    engine.command("units metal")
    engine.command("atom_style atomic")
    engine.command("dimension 3")
    engine.command(f"boundary {boundary}")
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


def reinitialize_system(engine, config, system):
    """Clear and reinitialize LAMMPS with a new system.

    Used after atom removal to rebuild LAMMPS state with updated atom count.
    """
    from ...config import pbc_to_lammps_boundary
    if config.lammps.boundary is not None:
        boundary = config.lammps.boundary
    else:
        boundary = pbc_to_lammps_boundary(system.pbc) if system.pbc is not None else "p p p"
    engine.lmp.command("clear")
    initialize_parameters(engine, boundary=boundary)
    initialize_system(engine, system)
    initialize_potential(engine, config)

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

def partn_search(engine, config, central_atom_idx: int, positions = None, cell = None, type=None) :
    original_stdout_fd = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    # Redirect stdout (fd 1) to /dev/null, only way to deal with pARTn error write
    os.dup2(devnull, 1)

    logger.debug("[pARTn] Searching from central atom %d", central_atom_idx)
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
    engine.command("minimize 1e-6 1e-8 10000 10000")
    engine.command("unfix 10")
    
    # Restore original stdout (fd 1)
    os.dup2(original_stdout_fd, 1)
    os.close(original_stdout_fd)
    os.close(devnull)


    # EXTRACT DATA
    if engine.rank == 0 :

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
                            types=type,
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
                            types=type,
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

def basin_reconstruct(engine, config, from_positions, from_types, cell, pbc,
                      ref_initial_positions, ref_saddle_positions, ref_final_positions,
                      ref_initial_types, sym_matrices, sym_perms, central_atom, sym_idx,
                      neighbor_indices, matching_score_thr, kmax_factor,
                      atom_coloring_mode):
    """Perform full basin reconstruction on engine ranks: PSR + 2x minimize.

    Rank 0 does PSR (IRA point set registration) and position manipulation.
    All ranks participate in LAMMPS minimization.

    Returns
    -------
    dict or None
        On rank 0: {"ok": True, "min2_positions": ndarray, "min2_etot": float}
        or {"ok": False, "error_type": str, "message": str}.
        None on non-root ranks.
    """
    import ira_mod
    import ase.geometry
    from ...utils.geometry import transform_positions, push_towards, compute_delr

    proceed = None  # control signal broadcast to all ranks

    if engine.rank == 0:
        # --- PSR phase (rank 0 only) ---
        coords1 = np.array(from_positions[neighbor_indices], copy=True)

        if atom_coloring_mode == "full":
            typ1 = list(np.array(from_types)[neighbor_indices])
            typ2 = list(ref_initial_types) if ref_initial_types is not None else typ1
        else:
            typ1 = ['X'] * len(coords1)
            typ2 = typ1

        # Unwrap coords near cell boundaries
        pbc_arr = pbc if pbc is not None else np.array([True, True, True])
        cell_lengths = [cell[d][d] for d in range(3)]
        central_pos = from_positions[central_atom]
        for i in range(len(coords1)):
            for dim in range(3):
                if pbc_arr[dim]:
                    diff = coords1[i][dim] - central_pos[dim]
                    if abs(diff) > cell_lengths[dim] / 2:
                        coords1[i][dim] += np.sign(-diff) * cell_lengths[dim]

        coords2 = np.array(ref_initial_positions)
        nat1 = len(coords1)
        nat2 = len(coords2)

        ira = ira_mod.IRA()
        try:
            rmat, tr, perm, dh = ira.match(nat1, typ1, coords1, nat2, typ2, coords2, kmax_factor)
        except Exception:
            proceed = {"ok": False, "error_type": "PSR_NO_MATCH_FOUND",
                       "message": "IRA did not find a match"}

        if proceed is None and dh > matching_score_thr:
            proceed = {"ok": False,
                       "error_type": "PSR_MATCHING_SCORE_ABOVE_ACCEPTANCE_THRESHOLD",
                       "message": f"PSR matching score {dh} above threshold {matching_score_thr}"}

        if proceed is None:
            # Apply symmetry + PSR transforms
            supposed_initial = np.array(ref_initial_positions, copy=True)
            supposed_final = np.array(ref_final_positions, copy=True)
            saddle = np.array(ref_saddle_positions, copy=True)

            if sym_idx != 0:
                sym_matrix = sym_matrices[sym_idx]
                sym_perm = sym_perms[sym_idx]
                supposed_initial = transform_positions(supposed_initial, sym_matrix, 0, sym_perm)
                saddle = transform_positions(saddle, sym_matrix, 0, sym_perm)
                supposed_final = transform_positions(supposed_final, sym_matrix, 0, sym_perm)

            supposed_initial = transform_positions(supposed_initial, rmat, tr, perm)
            saddle = transform_positions(saddle, rmat, tr, perm)
            supposed_final = transform_positions(supposed_final, rmat, tr, perm)

            # Build new system positions with saddle applied
            new_positions = np.array(from_positions, copy=True)
            new_positions[neighbor_indices] = saddle

            # Push toward min1
            saddle_toward_min1 = push_towards(
                new_positions[neighbor_indices], supposed_initial, fraction=0.15, cell=cell, pbc=pbc)
            tmp_positions = np.array(new_positions, copy=True)
            tmp_positions[neighbor_indices] = saddle_toward_min1
            proceed = {"step": "min1", "positions": tmp_positions,
                       "supposed_initial": supposed_initial, "supposed_final": supposed_final,
                       "saddle_positions": new_positions, "neighbor_indices": neighbor_indices}

    # Broadcast control signal to all ranks
    proceed = engine.engine_comm.bcast(proceed, root=0)

    if proceed is None or (isinstance(proceed, dict) and proceed.get("ok") is False):
        if engine.rank == 0:
            return proceed
        return None

    # --- Minimize toward min1 (all ranks) ---
    set_positions(engine=engine, positions=proceed["positions"])
    minimize(engine, config)
    min1_pos = get_positions(engine)

    # Validate min1 on rank 0
    proceed2 = None
    if engine.rank == 0:
        import ase.geometry
        from ...utils.geometry import push_towards, compute_delr

        supposed_initial = proceed["supposed_initial"]
        supposed_final = proceed["supposed_final"]
        saddle_positions = proceed["saddle_positions"]
        nbr_indices = proceed["neighbor_indices"]

        t1 = ase.geometry.wrap_positions(positions=min1_pos, cell=cell, pbc=pbc)
        delr1 = compute_delr(supposed_initial, t1[nbr_indices], cell, pbc=pbc)
        if delr1 > matching_score_thr:
            proceed2 = {"ok": False, "error_type": "RECONSTRUCTION_INVALID_MIN1",
                        "message": f"did not retrieve initial minimum: delr1 = {delr1}"}
        else:
            # Push toward min2
            saddle_toward_min2 = push_towards(
                saddle_positions[nbr_indices], supposed_final, fraction=0.15, cell=cell, pbc=pbc)
            tmp_positions2 = np.array(saddle_positions, copy=True)
            tmp_positions2[nbr_indices] = saddle_toward_min2
            proceed2 = {"step": "min2", "positions": tmp_positions2,
                        "supposed_final": supposed_final, "neighbor_indices": nbr_indices}

    proceed2 = engine.engine_comm.bcast(proceed2, root=0)

    if proceed2 is None or (isinstance(proceed2, dict) and proceed2.get("ok") is False):
        if engine.rank == 0:
            return proceed2
        return None

    # --- Minimize toward min2 (all ranks) ---
    set_positions(engine=engine, positions=proceed2["positions"])
    minimize(engine, config)
    min2_pos = get_positions(engine)
    min2_etot = get_total_energy(engine)

    if engine.rank == 0:
        import ase.geometry
        from ...utils.geometry import compute_delr

        supposed_final = proceed2["supposed_final"]
        nbr_indices = proceed2["neighbor_indices"]

        t2 = ase.geometry.wrap_positions(positions=min2_pos, cell=cell, pbc=pbc)
        delr2 = compute_delr(supposed_final, t2[nbr_indices], cell, pbc=pbc)
        if delr2 > matching_score_thr:
            return {"ok": False, "error_type": "RECONSTRUCTION_INVALID_MIN2",
                    "message": f"did not retrieve expected final minimum: delr2 = {delr2}"}

        return {"ok": True, "min2_positions": min2_pos, "min2_etot": min2_etot}


def basin_explore(engine, config_dict, reference_table_data, state_positions, state_types,
                  state_cell, state_pbc, state_index, start_index):
    """Perform basin exploration on engine rank 0. Other ranks idle.

    Parameters
    ----------
    config_dict : dict
        Subset of config fields needed for exploration.
    reference_table_data : bytes
        Pickled ReferenceEventTable.table DataFrame.
    state_positions, state_types, state_cell, state_pbc : array-like
        State data to reconstruct System + NeighborsList + AtomicEnvironment.
    state_index : int
        Index of the state being explored.
    start_index : int
        Starting index for new state connections.

    Returns
    -------
    list[dict] or None
        Connectivity rows on rank 0, None on other ranks.
    """
    if engine.rank == 0:
        import pickle
        from ...system import System
        from ...neighbors_list import NeighborsList
        from ...atomic_environment import AtomicEnvironment
        from ...basins.exploration import BasinGenericEventExplorer
        from ...basins.connectivity import BasinStatesConnectivity

        # Reconstruct the reference table DataFrame
        ref_table_df = pickle.loads(reference_table_data)

        # Proxy for ReferenceEventTable (only needs .table and .has_id_subset_table)
        class _RefTableProxy:
            def __init__(self, table_df):
                self.table = table_df
            def has_id_subset_table(self, ids):
                mask = self.table["event_id"].isin(ids)
                return self.table[mask]

        ref_table = _RefTableProxy(ref_table_df)

        # Reconstruct state
        system = System(positions=np.array(state_positions), types=list(state_types),
                        cell=np.array(state_cell), pbc=state_pbc,
                        index=np.arange(len(state_types)))
        neighbors_list = NeighborsList(system, config_dict["rnei"], config_dict["rcut"], graph_cutoff=config_dict.get("graph_cutoff"))
        types_for_env = system.types if config_dict["atom_coloring_mode"] == "full" else None
        environment = AtomicEnvironment(
            config_dict["ae_style"],
            neighbors_list.neighbors_list['rnei'],
            neighbors_list.neighbors_list['rcut'],
            config_dict["neighbors_add"],
            types=types_for_env,
            coordination_threshold=config_dict.get("coordination_threshold"))

        # Proxy for StateData
        class _StateProxy:
            def __init__(self, sys, env, nl):
                self.system = sys
                self.environment = env
                self.neighbors_list = nl
            def ensure_full_state(self, config):
                pass

        state = _StateProxy(system, environment, neighbors_list)

        # Proxy for Config (explorer only needs config.basin.energy_thr)
        class _ConfigProxy:
            def __init__(self, energy_thr):
                self.basin = type('obj', (object,), {'energy_thr': energy_thr})()

        config_proxy = _ConfigProxy(config_dict["energy_thr"])

        explorer = BasinGenericEventExplorer(config=config_proxy, reference_table=ref_table)
        explorer.explore(state=state, state_index=state_index, start_index=start_index)

        # Return connectivity rows as list of dicts
        if explorer.connectivity_table._rows:
            return explorer.connectivity_table._rows
        if explorer.connectivity_table._df is not None and not explorer.connectivity_table._df.empty:
            return explorer.connectivity_table._df.to_dict("records")
        return []

    return None


def partn_refine(engine, config, central_atom_idx:int , positions = None, cell = None, type=None, saddle_idx = None, saddle_positions = None, minimize_outter_atoms: bool = True) :

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
    attempt = 0

    while attempt < max_attempts :
        exit_flag = False
        result = None #for rank > 0
        engine.command("fix 10 all artn dmax {}".format(config.partn.r_dmax))
        engine.command("min_style fire")
                # RUN
        engine.command("minimize 1e-6 1e-8 10000 10000")
        engine.command("unfix 10")

        # EXTRACT DATA
        if engine.rank == 0 : 
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
                            refined='T'
                        )
                    )
        # Synchronize all ranks
        exit_flag = engine.local_engine_comm.bcast(exit_flag, root=0)
        if exit_flag : 
            return result 
                
        attempt +=1
        artn.set("zseed", config.partn.zseed)
        logger.debug("[pARTn] Retrying search with refreshed seed")

    else: #fail after max attemps
        if engine.rank == 0 : 
            err = artn.get_error()
            return Err(
                ErrorInfo(
                    type=ErrorType.EVENT_NOT_FOUND, message="no event found", details=err
                )
            )
        return None
