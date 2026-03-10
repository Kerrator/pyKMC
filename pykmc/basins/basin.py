from .detection import Detector
from .exploration import Explorer, BasinGenericEventExplorer
from .connectivity import BasinStatesConnectivity
from .selection import FPTASelector
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pykmc import System, Config, NeighborsList, AtomicEnvironment, ReferenceEventTable, PointSetRegistration, check_match, Reconstruction
from typing import Optional
from ..utils import geometry
from ..rate_constant import compute_rate_Eyring
import pandas as pd
import copy
import numpy as np
from scipy.spatial import cKDTree
from pykmc.result import Ok, BasinOutput
import logging

logger = logging.getLogger("log")

#TODO: StateDate is here to handle state informations, when State Object will be creates, need to remove
#TODO: For the moment Basin uses EnergyThresholdDetector, BasinGenericEventExplorer, FPTASelector, need to deal with possible multiple implementation with builder.
#TODO: Think about parallized exploration 
#TODO: Could think of refining transient -> absorbing event when exploring
#TODO : Exit if state 0 leads to all absorbing states because all unknown environments, here FTPA fails but because only have 1 transient state (0), should be a different ERROR.TYPE
#TODO should also check if we apply same event to different central atoms but same saddle position meaning that it s a duplicate event, so remove.

@dataclass
class StateData:
    system: Optional[System]
    environment: Optional[AtomicEnvironment]
    neighbors_list: Optional[NeighborsList] 
    transient: bool = False
    visited: bool = False

    def release_heavy_objects(self) -> None : 
        """Release heavy objects"""
        self.neighbors_list = None 
        self.environment = None
    
    def ensure_full_state(self, config: Config) -> None : 
        if self.system is not None : 
            if self.neighbors_list is None : 
                self.neighbors_list = NeighborsList(self.system, config.atomicenvironment.rnei, config.atomicenvironment.rcut)  
            if self.environment is None :
                types = self.system.types if config.atomicenvironment.atom_coloring_mode == "full" else None
                self.environment = AtomicEnvironment(config.atomicenvironment.style, self.neighbors_list.neighbors_list['rnei'], self.neighbors_list.neighbors_list['rcut'], config.atomicenvironment.neighbors_add, types=types, coordination_threshold=config.atomicenvironment.coordination_threshold)


class BasinsGenericEvents() : 

    def __init__(self, config: Config, reference_table,known_environments, manager ) -> None :  
        self.config = config #Config object with basins parameters
        self.explorer = None #object to explore a state in the basin 
        self.reference_table = reference_table #Object with reference generic events
        self.manager = manager #object to do external task (minimize, refine)

        self.connectivity_table = None #Dataframe of basin connexion state
        self.selected_event = None #The selected event after basin exploration
        self.current_state = None #Current state where we're at
        self.states_to_explore = None #List of state to explore
        self.explored_states = None #List of state that we already explored
        self.states: dict[int, StateData] = {}  #Dictionnary of StateDate
        self._state_fingerprints: dict[int, np.ndarray] = {}  # Fast dedup rejection cache
        self.known_environments = known_environments
        self.absorbing_saddle_positions: dict[int, np.ndarray] = {}
        self._next_state_index = 1  # Monotonic counter for state indices (0 is the initial state)
        self._use_session_pool = False  # Set True only for parallel strategies that call use_local()

    def detection(self, params) -> bool : 
        """Utility method."""
        return self.detector.detection(**params) 
    
    def execute(self, system) : 
        """ 
        run the basin exploration and select an event from a system, corresponding to the first state in the basin, it is assumed that this state is transient.
        """
        #initialize the basin
        self._initialize(system)
        #explore the basin
        strategy = self.config.basin.strategy
        if strategy != "serial":
            result = self.construct_connexion_table_parallel()
        else:
            result = self.construct_connexion_table()
        if not result.is_ok() :
            return result

        # Sanity check: every state in the connectivity table must be in self.states
        table_states = set(self.connectivity_table.df["state"]) | set(self.connectivity_table.df["state_connexion"])
        missing_from_states = table_states - set(self.states.keys())
        if missing_from_states:
            raise RuntimeError(f"[Basin] BUG: {len(missing_from_states)} states in connectivity table but not in self.states: {sorted(missing_from_states)[:10]}...")

        #reorder states index
        mapping = self.connectivity_table.reorder_states_index()
        self.states = {mapping[old]: val for old, val in self.states.items()}

        # Fix transient flags: after reordering, state classification is by index range.
        # change_state_index() during BFS can merge transient transitions to absorbing-range
        # states, leaving stale transient=True flags. Normalize based on reordered indices.
        n_transient = len(set(self.connectivity_table.df['state']))
        self.connectivity_table.df['transient'] = self.connectivity_table.df['state_connexion'].apply(lambda x: x < n_transient)

        all_states_set = set(self.connectivity_table.df["state"]) | set(self.connectivity_table.df["state_connexion"])
        n_absorbing_states = len(all_states_set) - n_transient
        n_absorbing_rows = len(self.connectivity_table.df[self.connectivity_table.df["transient"] == False])
        logger.info("[Basin] Reordered: %d transient + %d absorbing states | %d absorbing rows to refine",
                    n_transient, n_absorbing_states, n_absorbing_rows)

        #Refine absorbing states
        self.manager.use_local()
        result = self.refine_absorbing(system)
        if not result.is_ok() :
            return result

        logger.info("[Basin] Refined %d absorbing states", len(self.absorbing_saddle_positions))

        #apply selector algorithm to find t_exit and exit_state
        result = self.selector.select_from_connectivity(self.connectivity_table)
        if not result.is_ok() :
            return result
        #Construct output KMC needs
        t_exit = result.ok_value().t_exit
        exit_state = result.ok_value().exit_state
        logger.info("[Basin] FPTA selected: exit_state=%d, t_exit=%.6e", exit_state, t_exit)

        from_state, event_idx, central_atom, sym_idx, is_transient = self.connectivity_table.get_transition_to_state(target_state=exit_state)
        #Ensure from_state is state are full 
        self.states[from_state].ensure_full_state(self.config)

        neighbors = self.states[from_state].neighbors_list.get_neighbors("rcut", central_atom)
        return Ok(BasinOutput(initial_system_positions=self.states[from_state].system.positions, 
                              central_atom=central_atom, 
                              saddle_positions=self.absorbing_saddle_positions[exit_state], 
                              final_positions=self.states[exit_state].system.positions[neighbors], 
                              neighbors=neighbors,
                              energy_barrier= self.connectivity_table.df[(self.connectivity_table.df["state"] == from_state) & (self.connectivity_table.df["state_connexion"] == exit_state)].iloc[0]["dE_forward"], 
                              k_tot = self.connectivity_table.df.loc[self.connectivity_table.df["transient"] == False, "k_forward"].sum(),
                              t_exit = t_exit,
                              exit_state = exit_state, 
                              from_state = from_state,
                              num_reference_event= event_idx))
        

    def _initialize(self, system) -> None: 
        """ 
        Initialize necessary component after entering in basin. We always enter in state == 0.
        """
        self.current_state = 0
        self.states_to_explore = [0]
        self.explored_states = []
        self._next_state_index = 1  # State 0 is already assigned
        self.connectivity_table = BasinStatesConnectivity()
        self.explorer = BasinGenericEventExplorer(config=self.config, reference_table=self.reference_table)
        self.selector = FPTASelector()
        new_system = System(positions=system.positions.copy(), types=system.types.copy(), cell=system.cell.copy(), pbc=system.pbc.copy(), index=np.arange(len(system.types)))
        self._add_state(state_index=0, system=new_system)  #add current state 0 to self.states


    def construct_connexion_table(self) :
        """
        explore the basin and construct the connextion table
        """
        import time
        t_start = time.perf_counter()
        n_explored = 0
        n_duplicates = 0
        n_absorbing = 0
        n_processed = 0

        # Profiling accumulators (per-phase wall time in seconds)
        prof = {"reconstruct": 0.0, "psr": 0.0, "minimize": 0.0,
                "dedup": 0.0, "ensure_state": 0.0, "explore": 0.0,
                "merge": 0.0, "other": 0.0}

        # Switch to session pool for basin reconstruction (parallel minimization)
        self.manager.use_local()

        #Loop over state to explore
        while len(self.states_to_explore) != 0 :
            #next state to explore :
            to_explore = self.states_to_explore[0]

            if to_explore not in self.states : #always true except at the start (to_explore = 0)
                n_processed += 1
                #We need to create the state
                    #find a state and an event from which we go to the state that we want to create
                from_state, event_idx, central_atom, sym_idx, is_transient = self.connectivity_table.get_transition_to_state(target_state=to_explore)

                    #Create new system by applying (reconstruction) the generic event to the from_state
                t0 = time.perf_counter()
                result = self.system_from_state(from_state, event_idx, central_atom, sym_idx)
                prof["reconstruct"] += time.perf_counter() - t0
                if not result.is_ok() :
                    return result
                new_system = result.ok_value()

                    #Check if it is a new_system or already in states
                t0 = time.perf_counter()
                is_new_state = self.is_new_state(new_system)
                prof["dedup"] += time.perf_counter() - t0
                if is_new_state != -1 : #It already exists
                    #update table
                    self.connectivity_table.change_state_index(current_index=to_explore, new_index=is_new_state)
                    self.explored_states.append(to_explore)
                    self.states_to_explore.remove(to_explore)
                    n_duplicates += 1

                    if n_duplicates % 20 == 0:
                        elapsed = time.perf_counter() - t_start
                        logger.debug("[Basin] processed=%d | duplicates=%d | absorbing=%d | explored=%d | to_explore=%d | %.1fs",
                                     n_processed, n_duplicates, n_absorbing, n_explored, len(self.states_to_explore), elapsed)

                    #Cleaning
                    self.states[from_state].release_heavy_objects()
                    continue #Skip the rest

                #add state
                self._add_state(state_index=to_explore, system=new_system, transient=is_transient)

                #Ensure full state to explore
                t0 = time.perf_counter()
                self.states[to_explore].ensure_full_state(self.config)
                prof["ensure_state"] += time.perf_counter() - t0
                #Check if unknown atomic environments
                if self.is_states_has_unknown_environments(self.states[to_explore]) :
                    #We consider that this state is an absorbing one because we need to search new events (in main KMC loop)
                    #Need to update the connectivity table
                    self.connectivity_table.change_state_to_absorbing(to_explore)
                    self.states[to_explore].transient = False
                    is_transient = False

                if not is_transient :
                    self.states_to_explore.remove(to_explore)
                    self.explored_states.append(to_explore)
                    n_absorbing += 1

                    if n_absorbing % 20 == 0:
                        elapsed = time.perf_counter() - t_start
                        logger.debug("[Basin] processed=%d | absorbing=%d | duplicates=%d | explored=%d | to_explore=%d | %.1fs",
                                     n_processed, n_absorbing, n_duplicates, n_explored, len(self.states_to_explore), elapsed)

                    #Cleaning
                    self.states[from_state].release_heavy_objects()
                    self.states[to_explore].release_heavy_objects()

                    continue #We dont explore/skip the rest

                #Release heavy objet memory
                self.states[from_state].release_heavy_objects()


            #Explore state via MPI engine
            self.current_state = to_explore
            last_state_connectivity = self.get_last_state_index()

            t0 = time.perf_counter()
            self._explore_states_parallel([to_explore], n_workers=1)
            prof["explore"] += time.perf_counter() - t0

            #to_explore has been explored :
            self.states_to_explore.remove(to_explore)
            self.explored_states.append(to_explore)

            t0 = time.perf_counter()
            self.update_to_explore()
            prof["merge"] += time.perf_counter() - t0
            #Clean heavy state object :
            self.states[to_explore].release_heavy_objects()

            # Progress tracking
            n_explored += 1
            elapsed = time.perf_counter() - t_start
            logger.debug("[Basin] explored=%d | to_explore=%d | unique_states=%d | duplicates=%d | absorbing=%d | conn_rows=%d | %.1fs",
                         n_explored, len(self.states_to_explore), len(self.states), n_duplicates, n_absorbing, len(self.connectivity_table.df), elapsed)

        # Basin exploration complete — switch back to global mode
        self.manager.use_global()

        elapsed = time.perf_counter() - t_start
        n_transient = len(set(self.connectivity_table.df['state']))
        all_states = set(self.connectivity_table.df["state"]) | set(self.connectivity_table.df["state_connexion"])
        n_absorbing_final = len(all_states) - n_transient
        logger.info("[Basin] COMPLETE: %d transient + %d absorbing states | %d connectivity rows | processed=%d | duplicates=%d | %.1fs",
                    n_transient, n_absorbing_final, len(self.connectivity_table.df), n_processed, n_duplicates, elapsed)

        # Profiling summary — psr and minimize are sub-components of reconstruct,
        # so exclude them from the top-level sum to avoid double-counting.
        top_level = {k: v for k, v in prof.items() if k not in ("other", "psr", "minimize")}
        prof["other"] = elapsed - sum(top_level.values())
        logger.info("[Basin] PROFILING: reconstruct=%.2fs (psr=%.2fs + min=%.2fs) | dedup=%.2fs | explore=%.2fs | ensure_state=%.2fs | merge=%.2fs | other=%.2fs | total=%.2fs",
                    prof["reconstruct"], prof["psr"], prof["minimize"], prof["dedup"], prof["explore"],
                    prof["ensure_state"], prof["merge"], prof["other"], elapsed)
        for phase, t in sorted(top_level.items(), key=lambda x: -x[1]):
            pct = 100.0 * t / elapsed if elapsed > 0 else 0
            logger.info("[Basin] PROFILING:   %-15s %8.2fs  %5.1f%%", phase, t, pct)

        # Write timing checkpoint for compare_scaling.py
        self._write_timing_checkpoint(prof, elapsed, n_transient, n_absorbing_final, n_duplicates, n_processed)

        return Ok(None)

    def select_event(self) :
        """ 
        select an event base on the selector algorithm
        """
        pass

    def get_seletec_event(self) : 
        """ 
        convinient method
        """
        pass

    def get_last_state_index(self) :
        """Return the next available state index (monotonically increasing).

        Using a monotonic counter prevents index reuse when change_state_index()
        remaps high-valued indices to lower ones, which would cause the table max
        to drop and subsequent explorations to reuse indices already in explored_states.
        """
        return self._next_state_index
    
    def update_to_explore(self) : 
        #Find all state index in the connexion table : 
        unique_states = set(self.connectivity_table.get_table()['state']).union(set(self.connectivity_table.get_table()['state_connexion']))
        self.states_to_explore =  list(unique_states.difference(set(self.explored_states)))


    def _prepare_reconstruct_kwargs(self, from_state, event_idx, central_atom, sym_idx):
        """Prepare keyword arguments for manager.basin_reconstruct().

        Gathers all data needed by the engine to perform PSR + minimize.
        """
        ref_event = self.reference_table.table[self.reference_table.table["idx_ref"] == event_idx]
        if ref_event.empty:
            raise ValueError(f"idx_ref={event_idx} not found in reference table")
        ref_event = ref_event.iloc[0]

        self.states[from_state].ensure_full_state(self.config)
        neighbor_indices = self.states[from_state].neighbors_list.get_neighbors('rcut', central_atom)

        return {
            "config": self.config,
            "from_positions": self.states[from_state].system.positions.copy(),
            "from_types": list(self.states[from_state].system.types),
            "cell": self.states[from_state].system.cell.copy(),
            "pbc": self.states[from_state].system.pbc,
            "ref_initial_positions": np.array(ref_event["initial_positions"], copy=True),
            "ref_saddle_positions": np.array(ref_event["saddle_positions"], copy=True),
            "ref_final_positions": np.array(ref_event["final_positions"], copy=True),
            "ref_initial_types": ref_event.get("initial_types"),
            "sym_matrices": ref_event["sym_matrix"],
            "sym_perms": ref_event["sym_perm"],
            "central_atom": central_atom,
            "sym_idx": sym_idx,
            "neighbor_indices": neighbor_indices,
            "matching_score_thr": self.config.psr.matching_score_thr,
            "kmax_factor": self.config.ira.kmax_factor,
            "atom_coloring_mode": self.config.atomicenvironment.atom_coloring_mode,
        }

    def _result_from_mpi(self, mpi_result, from_state):
        """Convert MPI basin_reconstruct result dict to Ok(System) or Err(ErrorInfo)."""
        from pykmc.result import Err, ErrorInfo, ErrorType

        if mpi_result is None or not mpi_result.get("ok"):
            error_type_str = mpi_result.get("error_type", "UNKNOWN") if mpi_result else "UNKNOWN"
            message = mpi_result.get("message", "Unknown error") if mpi_result else "No result from engine"
            error_type = getattr(ErrorType, error_type_str, ErrorType.RECONSTRUCTION_INVALID_MIN2)
            return Err(ErrorInfo(type=error_type, message=message))

        import ase.geometry
        cell = self.states[from_state].system.cell
        pbc = self.states[from_state].system.pbc
        positions = ase.geometry.wrap_positions(
            positions=mpi_result["min2_positions"], cell=cell, pbc=pbc)
        new_system = System(
            positions=positions,
            types=self.states[from_state].system.types,
            cell=cell,
            pbc=pbc,
            index=np.arange(len(self.states[from_state].system.types)))
        return Ok(new_system)

    def system_from_state(self, from_state, event_idx, central_atom, sym_idx):
        """Reconstruct a new state via MPI engine (PSR + minimize).

        Submits the reconstruction task to an engine rank and blocks until complete.
        """
        kwargs = self._prepare_reconstruct_kwargs(from_state, event_idx, central_atom, sym_idx)
        future = self.manager.basin_reconstruct(**kwargs)
        mpi_result = future.result()
        return self._result_from_mpi(mpi_result, from_state)

    def refine_absorbing(self, system) :
        """When connectivity table is build, and that we have dict of states, we refine the energy barrier and k_forward of the transient -> absorbing event"""
        #compute the energy of the state 
        #for all row in connectivity table where we need to refine
        futures_context = {} #idx → { "min": f_min, "saddle": f_sad }
        for idx, row in self.connectivity_table.df.iterrows() : 
            if row['transient']  == False : #need to refine
                #tmp_system = copy.deepcopy(self.states[row["state"]].system)
                tmp_system = System(positions=self.states[row["state"]].system.positions.copy(), types=self.states[row["state"]].system.types, cell=self.states[row["state"]].system.cell, pbc=self.states[row["state"]].system.pbc, index=np.arange(len(self.states[row["state"]].system.types)))
                #get tmp_system energy 
                future1 = self.manager.get_total_energy(positions=tmp_system.positions.copy()) #Send copy not reference
                #move to generic saddle positions 
                ref_event = self.reference_table.table[self.reference_table.table["idx_ref"] == row["event_connexion"]] 
                if ref_event.empty:
                    raise ValueError(f"idx_ref={row['event_connexion']} not found in reference table")
                ref_event = ref_event.iloc[0].copy()
                #ref_event = self.reference_table.table.iloc[row["event_connexion"]].copy()
                saddle_positions = ref_event['saddle_positions'].copy()
                #Apply PSR between event initial position and environment positions of the central_atoms


                #ENSURE "STATE" FULL 
                self.states[row["state"]].ensure_full_state(self.config)

                result = PointSetRegistration(self.config, tmp_system, ref_event , self.states[row["state"]].neighbors_list, row["central_atom"]).match()
                if not result.is_ok(): #PSR Err
                    return result
                    # Check if PointSetRegistration match is valid 
                result = check_match(result, self.config.psr.matching_score_thr)
                if not result.is_ok() : #PSR matching score not valid : 
                    return result
                else : 
                    psr_output = result.ok_value() #get psr results

                # Apply symmetry matrix if sym != 0
                if row["sym"] != 0 :
                    sym_matrices = ref_event['sym_matrix']
                    sym_matrix = sym_matrices[row["sym"]]
                    saddle_positions = geometry.transform_positions(saddle_positions, sym_matrix,0, ref_event["sym_perm"][row["sym"]])
                saddle_positions = geometry.transform_positions(saddle_positions, psr_output.rotation_matrix, psr_output.translation_matrix, psr_output.permutation_matrix)
                neighbors = self.states[row["state"]].neighbors_list.get_neighbors('rcut', row["central_atom"])

                if self.config.control.active_volume==True:
                    # add a job to manager queue
                    future2 = self.manager.partn_refine(self.config, row["central_atom"],
                                                  tmp_system.positions.copy(),
                                                  tmp_system.cell,
                                                  tmp_system.types.copy(),
                                                  neighbors.copy(),
                                                  saddle_positions.copy())
                # Move system do saddle positions
                else:
                    tmp_system.update_positions(saddle_positions, atom_idx = neighbors)
                    #refine
                    future2 = self.manager.partn_refine(self.config, row["central_atom"], tmp_system.positions.copy()) #send copy not reference !
                
                #save future in context : 
                futures_context[idx] = {
            "min": future1,
            "saddle": future2, 
            "neighbors": neighbors}
                
                #RELEASE MEMORY : 
                self.states[row["state"]].release_heavy_objects()

        #modify connectivity table entry future1 hold min energy, future2 holds E_saddle
        for idx, ctx in futures_context.items():
            E_min    = ctx["min"].result()
            result_sad = ctx["saddle"].result()
            if not result_sad.is_ok() : 
                return result_sad
            E_sad = result_sad.ok_value().E_saddle
            if self.config.control.active_volume==True:
                dE = E_sad
            else:
                dE = E_sad - E_min
            k = compute_rate_Eyring(dE, self.config)

            #also save saddle positions refined 
            idx_state = self.connectivity_table.df.loc[idx].at['state_connexion']
            central_atom = self.connectivity_table.df.loc[idx].at['central_atom']
            #self.absorbing_saddle_positions[idx_state] = result.ok_value().saddle_positions[self.states[idx_state].neighbors_list.get_neighbors("rcut", central_atom)]
            self.absorbing_saddle_positions[idx_state] = result_sad.ok_value().saddle_positions[ctx["neighbors"]]
            # update connectivity table row
            self.connectivity_table.df.loc[idx, "dE_forward"] = dE
            self.connectivity_table.df.loc[idx, "k_forward"] = k
        return Ok(None)


    def is_new_state(self, system) :
        #Loop over all other system in self.states to see if system is already known
        fp_new = self._compute_fingerprint(system.positions, system.cell, system.pbc)

        # Vectorized fingerprint rejection: compare against all states at once
        fp_items = [
            (si, fp)
            for si, fp in self._state_fingerprints.items()
            if len(fp) == len(fp_new)
        ]
        if fp_items:
            indices, fps = zip(*fp_items)
            fp_matrix = np.vstack(fps)  # (N_states, N_atoms)
            max_diffs = np.max(np.abs(fp_matrix - fp_new[np.newaxis, :]), axis=1)
            candidates = [indices[i] for i in np.where(max_diffs <= 0.5)[0]]
        else:
            candidates = list(self.states.keys())

        for state_index in candidates:
            state_data = self.states[state_index]
            if state_data.system is None:
                continue
            are_equivalent = self.are_structures_equivalent(system.positions, state_data.system.positions, cell = system.cell, pbc=system.pbc)
            if are_equivalent :
                return state_index
        return -1


    def are_structures_equivalent(self, pos1, pos2, cell, pbc=None, tol=0.3):

        if len(pos1) != len(pos2):
            return False

        if pbc is None or np.all(pbc):
            # Fully periodic: use boxsize (existing fast path)
            box = np.diag(cell).tolist()
            tree2 = cKDTree(pos2, boxsize=box)
            distances, _ = tree2.query(pos1, k=1)
        else:
            # Mixed PBC: manual minimum-image distance
            box = np.diag(cell)
            distances = np.zeros(len(pos1))
            for i, p in enumerate(pos1):
                diffs = pos2 - p
                for dim in range(3):
                    if pbc[dim]:
                        diffs[:, dim] -= np.round(diffs[:, dim] / box[dim]) * box[dim]
                distances[i] = np.min(np.linalg.norm(diffs, axis=1))

        return np.max(distances) < tol

    def is_states_has_unknown_environments(self, state: StateData) : 
        if set(state.environment.atomic_environment_list).difference(self.known_environments) != set() :
            return True 
        else : 
            return False

    @staticmethod
    def _compute_fingerprint(positions: np.ndarray, cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
        """Compute a cheap structural fingerprint for fast inequality rejection.

        Returns sorted per-atom distances from center of mass. Rotationally and
        permutationally invariant — ideal for quickly ruling out non-equivalent
        structures before the expensive cKDTree comparison.
        """
        box = np.diag(cell).astype(np.float64)
        pbc_array = np.asarray(pbc, dtype=bool) if pbc is not None else np.array([True, True, True])
        pos = np.array(positions, dtype=np.float64, copy=True)
        for dim in range(3):
            if pbc_array[dim] and box[dim] > 0:
                pos[:, dim] = np.mod(pos[:, dim], box[dim])
        com = pos.mean(axis=0)
        diffs = pos - com
        for dim in range(3):
            if pbc_array[dim] and box[dim] > 0:
                diffs[:, dim] -= np.round(diffs[:, dim] / box[dim]) * box[dim]
        return np.sort(np.linalg.norm(diffs, axis=1))

    def _add_state(self, state_index, system=None, transient=True, applicable_events=None, visited=False, full=False ) :
        """Add a new state in the `self.states` dictionnary."""
        #to fit typing
        neighbors_list  = []
        atomic_environment = []

        if full == True :
            neighbors_list = NeighborsList(system, self.config.atomicenvironment.rnei, self.config.atomicenvironment.rcut)
            types = system.types if self.config.atomicenvironment.atom_coloring_mode == "full" else None
            atomic_environment = AtomicEnvironment(self.config.atomicenvironment.style, neighbors_list.neighbors_list['rnei'], neighbors_list.neighbors_list['rcut'], self.config.atomicenvironment.neighbors_add, types=types, coordination_threshold=self.config.atomicenvironment.coordination_threshold)
        else :
            neighbors_list = None
            atomic_environment = None
        new_state =  StateData(system=system, environment=atomic_environment, neighbors_list=neighbors_list, transient=transient,  visited=visited)

        self.states[state_index]= new_state
        if system is not None:
            self._state_fingerprints[state_index] = self._compute_fingerprint(
                system.positions, system.cell, system.pbc
            )

    def _write_timing_checkpoint(self, prof, elapsed, n_transient, n_absorbing, n_duplicates, n_processed):
        """Write a timing summary file for compare_scaling.py."""
        import os
        strategy = getattr(self.config.basin, 'strategy', 'serial')
        n_workers = getattr(self.config.basin, 'n_workers', 1)
        n_conn = len(self.connectivity_table.df) if not self.connectivity_table.df.empty else 0

        # Write as level_complete checkpoint (L0 = single-level basin)
        ckpt_path = f"basin_timing_{strategy}.txt"
        with open(ckpt_path, "w") as f:
            f.write(f"# Basin timing checkpoint\n")
            f.write(f"strategy = {strategy}\n")
            f.write(f"n_workers = {n_workers}\n")
            f.write(f"wall_time_s = {elapsed:.3f}\n")
            f.write(f"states_transient = {n_transient}\n")
            f.write(f"states_absorbing = {n_absorbing}\n")
            f.write(f"states_total = {n_transient + n_absorbing}\n")
            f.write(f"connectivity_rows = {n_conn}\n")
            f.write(f"n_duplicates = {n_duplicates}\n")
            f.write(f"n_processed = {n_processed}\n")
            for phase, t in sorted(prof.items(), key=lambda x: -x[1]):
                pct = 100.0 * t / elapsed if elapsed > 0 else 0
                f.write(f"prof_{phase} = {t:.3f}\n")
                f.write(f"pct_{phase} = {pct:.1f}\n")
        logger.info("[Basin] Timing checkpoint written to %s", ckpt_path)

        # Also write as level_complete format for compare_scaling.py compatibility
        level_path = f"basin_connectivity_0_L0_level_complete.txt"
        with open(level_path, "w") as f:
            f.write(f"# Basin level complete checkpoint\n")
            f.write(f"level = 0\n")
            f.write(f"wall_time_s = {elapsed:.3f}\n")
            f.write(f"level_wall_time_s = {elapsed:.3f}\n")
            f.write(f"states_total = {n_transient + n_absorbing}\n")
            f.write(f"connectivity_rows = {n_conn}\n")

    # ──────────────────────────────────────────────────────────────────
    # Parallel basin exploration strategies
    # ──────────────────────────────────────────────────────────────────

    def _estimate_max_transitions_per_state(self):
        """Upper-bound on transitions one explore() call can produce.

        Used to pre-allocate non-overlapping index ranges for parallel workers.
        """
        table = self.reference_table.table
        if table.empty:
            return 100  # safe fallback
        # Each event can apply to multiple atoms × symmetries
        max_syms = max(len(s) for s in table["sym_matrix"])
        # Rough upper bound: n_events * max_atoms * max_syms
        # Use generous estimate since unused indices are harmless
        return len(table) * 50 * max_syms

    def _prepare_explore_kwargs(self, state_idx, start_index):
        """Prepare keyword arguments for manager.basin_explore()."""
        import pickle

        self.states[state_idx].ensure_full_state(self.config)
        state = self.states[state_idx]

        config_dict = {
            "rnei": self.config.atomicenvironment.rnei,
            "rcut": self.config.atomicenvironment.rcut,
            "neighbors_add": self.config.atomicenvironment.neighbors_add,
            "ae_style": self.config.atomicenvironment.style,
            "atom_coloring_mode": self.config.atomicenvironment.atom_coloring_mode,
            "coordination_threshold": self.config.atomicenvironment.coordination_threshold,
            "energy_thr": self.config.basin.energy_thr,
        }

        return {
            "config_dict": config_dict,
            "reference_table_data": pickle.dumps(self.reference_table.table),
            "state_positions": state.system.positions.copy(),
            "state_types": list(state.system.types),
            "state_cell": state.system.cell.copy(),
            "state_pbc": state.system.pbc,
            "state_index": state_idx,
            "start_index": start_index,
        }

    def _explore_states_parallel(self, states_batch, n_workers=4):
        """Explore multiple transient states in parallel via MPI engines.

        Each engine rank runs its own BasinGenericEventExplorer with a
        non-overlapping state index range. Connectivity rows are merged
        on rank 0 after all engines complete.
        """
        if not states_batch:
            return

        gap = self._estimate_max_transitions_per_state()
        base = self._next_state_index

        # Submit all exploration tasks to MPI engines
        futures = {}
        for i, state_idx in enumerate(states_batch):
            start = base + i * gap
            kwargs = self._prepare_explore_kwargs(state_idx, start)
            futures[state_idx] = self.manager.basin_explore(**kwargs)

        # Collect results and merge connectivity rows
        max_new_index = base
        for state_idx, future in futures.items():
            rows = future.result()
            if rows:
                self.connectivity_table.add_connectivity_batch(rows)
                max_conn = max(r["state_connexion"] for r in rows)
                max_new_index = max(max_new_index, max_conn + 1)

        self._next_state_index = max_new_index

    def is_new_state_batch(self, new_systems):
        """Check multiple systems for duplicates at once.

        Parameters
        ----------
        new_systems : dict[int, System]
            Mapping state_idx -> System for newly reconstructed states.

        Returns
        -------
        dict[int, int]
            Mapping state_idx -> existing_state_idx for duplicates,
            state_idx -> -1 for genuinely new states.
        """
        results = {}

        # Pre-compute fingerprints for new systems
        new_fingerprints = {}
        for idx, system in new_systems.items():
            new_fingerprints[idx] = self._compute_fingerprint(system.positions, system.cell, system.pbc)

        # Build fingerprint-filtered cKDTree only for candidate existing states
        # (pre-filter: only build trees for states whose fingerprint is close)
        existing_trees = {}
        for idx, state_data in self.states.items():
            if state_data.system is not None:
                if state_data.system.pbc is None or np.all(state_data.system.pbc):
                    box = np.diag(state_data.system.cell).tolist()
                    existing_trees[idx] = cKDTree(state_data.system.positions, boxsize=box)
                else:
                    existing_trees[idx] = None  # fallback to manual comparison

        # Pre-compute fingerprint candidate sets for each new system vs existing states
        existing_fp_items = [
            (si, fp)
            for si, fp in self._state_fingerprints.items()
        ]

        for new_idx, system in new_systems.items():
            match = -1
            fp_new = new_fingerprints[new_idx]

            # Fingerprint pre-filter against existing states
            if existing_fp_items:
                candidate_indices = []
                for si, fp in existing_fp_items:
                    if len(fp) == len(fp_new) and np.max(np.abs(fp - fp_new)) <= 0.5:
                        candidate_indices.append(si)
            else:
                candidate_indices = list(existing_trees.keys())

            for existing_idx in candidate_indices:
                if existing_idx not in existing_trees:
                    continue
                tree = existing_trees[existing_idx]
                if tree is not None:
                    distances, _ = tree.query(system.positions, k=1)
                    if np.max(distances) < 0.3:
                        match = existing_idx
                        break
                else:
                    state_data = self.states[existing_idx]
                    if self.are_structures_equivalent(system.positions, state_data.system.positions,
                                                      cell=system.cell, pbc=system.pbc):
                        match = existing_idx
                        break

            # Cross-check within this batch (two new states may be duplicates of each other)
            if match == -1:
                for other_idx in list(results.keys()):
                    if results[other_idx] != -1:
                        continue  # this one is already a duplicate itself
                    if other_idx in new_systems:
                        fp_other = new_fingerprints[other_idx]
                        # Fingerprint pre-filter within batch
                        if len(fp_other) == len(fp_new) and np.max(np.abs(fp_other - fp_new)) > 0.5:
                            continue
                        if self.are_structures_equivalent(system.positions,
                                                          new_systems[other_idx].positions,
                                                          cell=system.cell, pbc=system.pbc):
                            match = other_idx
                            break

            results[new_idx] = match
        return results

    def construct_connexion_table_parallel(self):
        """Wavefront-parallel BFS: processes batches of states instead of one at a time.

        Phases per wavefront:
            A. Batch reconstruction (PSR + minimize)
            B. Batch deduplication
            C. Parallel exploration of new transient states
            D. Merge and update queue
        """
        import time

        strategy = self.config.basin.strategy
        n_workers = self.config.basin.n_workers

        t_start = time.perf_counter()
        n_explored = 0
        n_duplicates = 0
        n_absorbing = 0
        n_processed = 0
        prof = {"reconstruct": 0.0, "psr": 0.0, "minimize": 0.0,
                "dedup": 0.0, "ensure_state": 0.0, "explore": 0.0,
                "merge": 0.0, "other": 0.0}

        # Switch to session pool for basin reconstruction (parallel minimization)
        self.manager.use_local()

        while len(self.states_to_explore) != 0:
            batch = list(self.states_to_explore)

            # Separate: states that need reconstruction vs state 0 (already exists)
            to_reconstruct = [s for s in batch if s not in self.states]
            already_exist = [s for s in batch if s in self.states]

            # ── Phase A: Batch reconstruction ──
            reconstructed = {}  # state_idx -> System
            transition_info = {}  # state_idx -> (from_state, event_idx, central_atom, sym_idx, is_transient)

            if to_reconstruct:
                # Gather transition info first (main thread, fast)
                for state_idx in to_reconstruct:
                    transition_info[state_idx] = self.connectivity_table.get_transition_to_state(target_state=state_idx)

                # Submit all reconstruction tasks to MPI engines
                t0 = time.perf_counter()
                futures = {}
                for state_idx in to_reconstruct:
                    from_state, event_idx, central_atom, sym_idx, is_transient = transition_info[state_idx]
                    kwargs = self._prepare_reconstruct_kwargs(from_state, event_idx, central_atom, sym_idx)
                    futures[state_idx] = (from_state, self.manager.basin_reconstruct(**kwargs))

                for state_idx, (from_state, future) in futures.items():
                    mpi_result = future.result()
                    result = self._result_from_mpi(mpi_result, from_state)
                    n_processed += 1
                    if result.is_ok():
                        reconstructed[state_idx] = result.ok_value()
                    else:
                        logger.warning("[Basin] Reconstruction failed for state %d: %s", state_idx, result.err_value())
                prof["reconstruct"] += time.perf_counter() - t0

            # ── Phase B: Batch deduplication ──
            # Always use batch dedup in the wavefront loop to catch intra-batch
            # duplicates.  Serial is_new_state() only checks against self.states
            # (which doesn't include other batch members), so duplicates within
            # the same batch go undetected — leading to exponential blowup.
            t0 = time.perf_counter()
            if len(reconstructed) > 1:
                dedup_results = self.is_new_state_batch(reconstructed)
            elif len(reconstructed) == 1:
                dedup_results = {}
                for state_idx, system in reconstructed.items():
                    dedup_results[state_idx] = self.is_new_state(system)
            else:
                dedup_results = {}
            prof["dedup"] += time.perf_counter() - t0

            # Process dedup results
            new_transient = []
            for state_idx in to_reconstruct:
                if state_idx not in reconstructed:
                    # Reconstruction failed — remove from queue
                    self.states_to_explore.remove(state_idx)
                    self.explored_states.append(state_idx)
                    continue

                existing = dedup_results.get(state_idx, -1)
                if existing != -1:
                    # Duplicate
                    self.connectivity_table.change_state_index(current_index=state_idx, new_index=existing)
                    self.explored_states.append(state_idx)
                    self.states_to_explore.remove(state_idx)
                    n_duplicates += 1
                else:
                    # New state
                    is_transient = transition_info[state_idx][4]
                    self._add_state(state_index=state_idx, system=reconstructed[state_idx], transient=is_transient)

                    t0 = time.perf_counter()
                    self.states[state_idx].ensure_full_state(self.config)
                    prof["ensure_state"] += time.perf_counter() - t0

                    if self.is_states_has_unknown_environments(self.states[state_idx]):
                        self.connectivity_table.change_state_to_absorbing(state_idx)
                        self.states[state_idx].transient = False
                        is_transient = False

                    if not is_transient:
                        self.states_to_explore.remove(state_idx)
                        self.explored_states.append(state_idx)
                        self.states[state_idx].release_heavy_objects()
                        n_absorbing += 1
                    else:
                        new_transient.append(state_idx)

            # Also include pre-existing states that need exploration (e.g., state 0)
            for state_idx in already_exist:
                if state_idx in self.states_to_explore:
                    new_transient.append(state_idx)

            # ── Phase C: Exploration via MPI engines ──
            if new_transient:
                t0 = time.perf_counter()
                self._explore_states_parallel(new_transient, n_workers=n_workers)
                prof["explore"] += time.perf_counter() - t0

                # Mark explored
                for state_idx in new_transient:
                    if state_idx in self.states_to_explore:
                        self.states_to_explore.remove(state_idx)
                    if state_idx not in self.explored_states:
                        self.explored_states.append(state_idx)
                    self.states[state_idx].release_heavy_objects()
                    n_explored += 1

            # ── Phase D: Update queue ──
            t0 = time.perf_counter()
            self.update_to_explore()
            prof["merge"] += time.perf_counter() - t0

            elapsed = time.perf_counter() - t_start
            logger.debug("[Basin] wavefront done | explored=%d | to_explore=%d | states=%d | dup=%d | abs=%d | %.1fs",
                         n_explored, len(self.states_to_explore), len(self.states), n_duplicates, n_absorbing, elapsed)

        # Basin exploration complete — switch back to global mode
        self.manager.use_global()

        elapsed = time.perf_counter() - t_start
        n_transient = len(set(self.connectivity_table.df['state']))
        all_states = set(self.connectivity_table.df["state"]) | set(self.connectivity_table.df["state_connexion"])
        n_absorbing_final = len(all_states) - n_transient
        logger.info("[Basin] COMPLETE (%s): %d transient + %d absorbing states | %d connectivity rows | processed=%d | duplicates=%d | %.1fs",
                    strategy, n_transient, n_absorbing_final, len(self.connectivity_table.df), n_processed, n_duplicates, elapsed)

        # Profiling summary — psr and minimize are sub-components of reconstruct,
        # so exclude them from the top-level sum to avoid double-counting.
        top_level = {k: v for k, v in prof.items() if k not in ("other", "psr", "minimize")}
        prof["other"] = elapsed - sum(top_level.values())
        logger.info("[Basin] PROFILING: reconstruct=%.2fs (psr=%.2fs + min=%.2fs) | dedup=%.2fs | explore=%.2fs | ensure_state=%.2fs | merge=%.2fs | other=%.2fs | total=%.2fs",
                    prof["reconstruct"], prof["psr"], prof["minimize"], prof["dedup"], prof["explore"],
                    prof["ensure_state"], prof["merge"], prof["other"], elapsed)
        for phase, t in sorted(top_level.items(), key=lambda x: -x[1]):
            pct = 100.0 * t / elapsed if elapsed > 0 else 0
            logger.info("[Basin] PROFILING:   %-15s %8.2fs  %5.1f%%", phase, t, pct)

        # Write timing checkpoint for compare_scaling.py
        self._write_timing_checkpoint(prof, elapsed, n_transient, n_absorbing_final, n_duplicates, n_processed)

        return Ok(None)
