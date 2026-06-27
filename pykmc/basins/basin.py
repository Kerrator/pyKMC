import logging
import time

from .exploration import BasinGenericEventExplorer
from .connectivity import BasinStatesConnectivity
from .selection import FPTASelector
from . import fingerprinting
from dataclasses import dataclass
from pykmc import System, Config, NeighborsList, AtomicEnvironment, PointSetRegistration, check_match, Reconstruction
from typing import Optional
from ..utils import geometry
from ..rate_constant import create_rate_constant
import pandas as pd
import copy
import numpy as np
from scipy.spatial import cKDTree
from pykmc.result import Result, Ok, Err, ErrorInfo, ErrorType, BasinOutput

logger = logging.getLogger("log")

#Sentinel returned by is_new_state_batch for states whose deduplication was cut
#short by the wall-time deadline: not a duplicate (-1 means new), but deferred.
DEDUP_DEFERRED = -2

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
                self.environment = AtomicEnvironment(config.atomicenvironment.style, self.neighbors_list.neighbors_list['rnei'], self.neighbors_list.neighbors_list['rcut'], config.atomicenvironment.neighbors_add, types=self.system.types if config.atomicenvironment.atom_coloring_mode == "full" else None, coordination_threshold=config.atomicenvironment.coordination_threshold)


class BasinsGenericEvents() : 

    def __init__(self, config: Config, reference_table, known_environments, manager) -> None:
        self.config = config
        self.rate_constant = create_rate_constant(
            T=config.rateconstant.T,
            prefactor_backend_name=config.rateconstant.style,
            config=config.rateconstant,
        )
        self.explorer = None #object to explore a state in the basin 
        self.reference_table = reference_table #Object with reference generic events
        self.manager = manager #object to do external task (minimize, refine)

        self.connectivity_table = None #Dataframe of basin connexion state
        self.selected_event = None #The selected event after basin exploration
        self.current_state = None #Current state where we're at 
        self.states_to_explore = None #List of state to explore 
        self.explored_states = None #List of state that we already explored
        self.states: dict[int, StateData] = {}  #Dictionnary of StateDate
        self.known_environments = known_environments 
        self.absorbing_saddle_positions: dict[int, np.ndarray] = {}

    def detection(self, params) -> bool : 
        """Utility method."""
        return self.detector.detection(**params) 
    
    def execute(self, system) : 
        """ 
        Run the basin exploration and select an event from a system, corresponding to the first state in the basin, it is assumed that this state is transient.
        """
        #initialize the basin
        self._initialize(system)
        #explore the basin (strategy from [BASIN]: serial one-state-at-a-time BFS, or
        #wavefront per-level batching across the MPI session pool)
        strategy = self.config.basin.strategy if self.config.basin is not None else "serial"
        if strategy == "wavefront" :
            result = self.construct_connexion_table_parallel()
        else :
            result = self.construct_connexion_table()
        if not result.is_ok() :
            return result
        #A budget can in principle fire before state 0 was even explored, leaving an
        #empty table; fall back to the plain KMC event instead of crashing downstream.
        if len(self.connectivity_table.df) == 0:
            return Err(ErrorInfo(
                type=ErrorType.BASIN_NO_VIABLE_EXIT,
                message="basin exploration produced no connectivity rows"))
        #reorder states index
        mapping = self.connectivity_table.reorder_states_index()
        self.states = {mapping[old]: val for old, val in self.states.items()}
        #Remap the exit-machinery containers through the same mapping (deferred /
        #excluded states have connectivity rows, so they appear in the mapping).
        #The fingerprint cache MUST be remapped too: the lazy exit path runs
        #is_new_state() post-reorder, and stale keys would crash or silently
        #scramble the dedup pre-filter.
        self._exit_excluded_states = {mapping.get(s, s) for s in self._exit_excluded_states}
        self._deferred_states = {mapping.get(s, s) for s in self._deferred_states}
        self._deferred_systems = {mapping.get(s, s): v for s, v in self._deferred_systems.items()}
        self._state_fingerprints = {mapping.get(s, s): fp for s, fp in self._state_fingerprints.items()}
        #Normalize the transient flag by index range. reorder_states_index() numbers the
        #transient states first (0..n_transient-1) and absorbing states after, but the
        #per-row 'transient' flag set during exploration can be stale after change_state_index()
        #merges duplicates. Recompute it from the post-reorder index range.
        n_transient, _ = self._connectivity_state_counts()
        self.connectivity_table.df["transient"] = self.connectivity_table.df["state_connexion"].apply(lambda x: x < n_transient)
        #Refine absorbing states
        self.manager.use_local()
        result =self.refine_absorbing(system)
        if not result.is_ok() : 
            return result
        #apply selector algorithm to find t_exit and exit_state
        result = self.selector.select_from_connectivity(self.connectivity_table, excluded_states=self._exit_excluded_states)
        if not result.is_ok() :
            return result
        #Construct output KMC needs
        t_exit = result.ok_value().t_exit
        exit_state = result.ok_value().exit_state

        #Snapshot k_tot from the same flags the generator that produced t_exit saw:
        #lazy materialization below can merge states and shift the flags.
        k_tot = self.connectivity_table.df.loc[self.connectivity_table.df["transient"] == False, "k_forward"].sum()  # noqa: E712

        #Deferred exits (budget-capped frontier) are materialized lazily here; a
        #materialization failure excludes that exit and redraws another.
        result = self._resolve_exit_state(t_exit, exit_state)
        if not result.is_ok() :
            return result
        exit_state = result.ok_value()

        #Re-normalize the per-row transient flag: a lazy merge (change_state_index)
        #can leave rows pointing at indices whose flag no longer matches the
        #'absorbing iff index >= n_transient' invariant the persisted table assumes.
        n_transient, _ = self._connectivity_state_counts()
        self.connectivity_table.df["transient"] = self.connectivity_table.df["state_connexion"].apply(lambda x: x < n_transient)

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
                              k_tot = k_tot,
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
        self.connectivity_table = BasinStatesConnectivity()
        self.explorer = BasinGenericEventExplorer(config=self.config, reference_table=self.reference_table)
        self.selector = FPTASelector(solver=self.config.basin.solver if self.config.basin is not None else "auto")
        new_system = System(positions=system.positions.copy(), types=system.types.copy(), cell=system.cell.copy(), pbc=system.pbc.copy(), index=np.arange(len(system.types)))
        self._state_fingerprints = {}  #state_index -> fingerprint vector (dedup pre-filter cache)
        self._was_capped = False  #set when a budget converts the remaining frontier to absorbing
        self._exit_excluded_states: set[int] = set()  #absorbing states the exit draw must skip (failed reconstruction)
        self._deferred_states: set[int] = set()  #absorbing states never reconstructed (budget cap); materialized lazily if selected
        self._deferred_systems: dict[int, System] = {}  #reconstructed-but-not-deduped systems stashed for lazy materialization
        self._n_failed = 0  #failed state reconstructions (failure-budget numerator)
        self._n_reconstruction_attempts = 0  #attempted state reconstructions (failure-budget denominator)
        self._add_state(state_index=0, system=new_system)  #add current state 0 to self.states
        self._next_state_index = 1  #monotonic state-index counter (state 0 is the initial state)


    def construct_connexion_table(self) :
        """Explore the basin and construct the connextion table.

        Per-phase timings are accumulated like the wavefront strategy and written to
        the same timing checkpoints at the end, so serial and wavefront runs can be
        compared directly with toolkit/basin_testing/compare_scaling.py.
        """
        t_start = time.perf_counter()
        n_processed = 0
        n_duplicates = 0
        n_explored = 0
        prof = {"reconstruct": 0.0, "psr": 0.0, "minimize": 0.0,
                "dedup": 0.0, "ensure_state": 0.0, "explore": 0.0,
                "merge": 0.0, "other": 0.0}

        #Loop over state to explore
        while len(self.states_to_explore) != 0 :
            # Budget checks (max_states / max_total_states / max_basin_walltime_s);
            # serial dedup is per-state already, so loop-top granularity suffices.
            reason = self._budget_breach_reason(n_explored, t_start)
            if reason is not None:
                result = self._cap_remaining_as_absorbing(reason)
                if not result.is_ok():
                    return result
                break

            #next state to explore :
            to_explore = self.states_to_explore[0]

            if to_explore not in self.states : #always true except at the start (to_explore = 0)
                #We need to create the state
                    #find a state and an event from which we go to the state that we want to create
                from_state, event_idx, central_atom, sym_idx, is_transient = self.connectivity_table.get_transition_to_state(target_state=to_explore)

                    #Create new system by applying (reconstruction) the generic event to the from_state
                t0 = time.perf_counter()
                result = self.system_from_state(from_state, event_idx, central_atom, sym_idx)
                prof["reconstruct"] += time.perf_counter() - t0
                n_processed += 1
                self._n_reconstruction_attempts += 1
                if not result.is_ok() :
                    #Non-fatal: keep the row as a non-selectable absorbing state and
                    #move on (same rule as the wavefront path).
                    self._mark_failed_absorbing(to_explore, result.err_value())
                    self.states[from_state].release_heavy_objects()
                    continue
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

                    #Cleaning
                    self.states[from_state].release_heavy_objects()
                    continue #Skip the rest

                #add state
                self._add_state(state_index=to_explore, system=new_system, transient=is_transient)

                #ENSURE FULL STATE TO EXPLORE
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

                    #Cleaning
                    self.states[from_state].release_heavy_objects()
                    self.states[to_explore].release_heavy_objects()


                    continue #We dont explore/skip the rest

                #Release heavy objet memory
                self.states[from_state].release_heavy_objects()




            #Explore state
            self.current_state = to_explore
            last_state_connectivity = self.get_last_state_index()

            #Ensure full state to explore
            t0 = time.perf_counter()
            self.states[to_explore].ensure_full_state(self.config)
            prof["ensure_state"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            self.explorer.explore(state=self.states[to_explore], state_index=self.current_state, start_index=last_state_connectivity)
            prof["explore"] += time.perf_counter() - t0
            n_explored += 1

            #to_explore has been explored :
            self.states_to_explore.remove(to_explore)
            self.explored_states.append(to_explore)

            #Merge state connectivity table to basin connectivity table
            t0 = time.perf_counter()
            self.connectivity_table.merge(self.explorer.connectivity_table)
            #Clrean explorer connectivity table
            self.explorer.clear()
            self.update_to_explore()
            #Advance the monotonic index counter past every index now in the table.
            #It must never decrease: change_state_index() can remap a high index to a
            #lower one, dropping the table max, and reading that max would reuse an index
            #already in explored_states (silently truncating exploration).
            self._advance_next_state_index()
            prof["merge"] += time.perf_counter() - t0
            #Clean heaby state object :
            self.states[to_explore].release_heavy_objects()

        return self._finalize_exploration_run(
            t_start,
            prof,
            n_processed,
            n_duplicates,
            strategy_label="serial",
        )

    def select_event(self) : 
        """ 
        Select an event base on the selector algorithm
        """
        pass

    def get_seletec_event(self) : 
        """ 
        Convinient method
        """
        pass

    def get_last_state_index(self) :
        """Return the next available state index (monotonically increasing).

        A monotonic counter prevents index reuse: when change_state_index() remaps a
        high-valued index to a lower one the table max drops, and the old
        ``state_connexion.iloc[-1] + 1`` scheme would then hand back an index already
        present in explored_states, so update_to_explore() would skip it.
        """
        return self._next_state_index

    def _advance_next_state_index(self) -> None:
        """Move the monotonic counter past the largest index in the table (never back)."""
        table = self.connectivity_table.get_table()
        if len(table) == 0:
            return
        current_max = int(max(table["state"].max(), table["state_connexion"].max()))
        self._next_state_index = max(self._next_state_index, current_max + 1)

    def _connectivity_state_counts(self) -> tuple[int, int]:
        """Return (n_transient, n_absorbing) from the connectivity table.

        Transient states are the explored sources (the ``state`` column); absorbing
        states appear only as ``state_connexion`` targets.
        """
        table = self.connectivity_table.get_table()
        if len(table) == 0:
            return 0, 0
        transient_states = set(table["state"])
        all_states = transient_states | set(table["state_connexion"])
        return len(transient_states), len(all_states) - len(transient_states)
    
    def update_to_explore(self) : 
        #Find all state index in the connexion table : 
        unique_states = set(self.connectivity_table.get_table()["state"]).union(set(self.connectivity_table.get_table()["state_connexion"]))
        self.states_to_explore =  list(unique_states.difference(set(self.explored_states)))


    def system_from_state(self, from_state, event_idx, central_atom, sym_idx) : 
        """Reconstruct the generic event to generate new state from state
        """
        ref_event = self.reference_table.table[self.reference_table.table["idx_ref"] == event_idx] #event where event_idx == idx_ref
        if ref_event.empty:
            raise ValueError(f"idx_ref={event_idx} not found in reference table")
        ref_event = ref_event.iloc[0].copy()
#        ref_event = self.reference_table.table.iloc[event_idx].copy()

        #supposed_initial_positions = ref_event["initial_positions"].copy()
        #supposed_final_positions = ref_event["final_positions"].copy()
        #saddle_positions = ref_event['saddle_positions'].copy()

        supposed_initial_positions = np.array(ref_event["initial_positions"], copy=True)
        supposed_final_positions = np.array(ref_event["final_positions"], copy=True)
        saddle_positions = np.array(ref_event["saddle_positions"], copy=True)

        #Apply the generic event to the current state 

        #ENSURE FULL STATE FOR FROM STATE 
        self.states[from_state].ensure_full_state(self.config)

            #We start from the from_state
        new_system = System(positions=self.states[from_state].system.positions.copy(), types=self.states[from_state].system.types, cell=self.states[from_state].system.cell, pbc=True, index=np.arange(len(self.states[from_state].system.types)))
        #new_system = copy.deepcopy(self.states[from_state].system)

            #Apply PSR between event initial position and environment positions of the central_atoms
        result = PointSetRegistration(self.config, new_system, ref_event , self.states[from_state].neighbors_list, central_atom).match()
        if not result.is_ok(): #PSR Err
            return result
            # Check if PointSetRegistration match is valid 
        result = check_match(result, self.config.psr.matching_score_thr)
        if not result.is_ok() : #PSR matching score not valid : 
            return result
        else : 
            psr_output = result.ok_value() #get psr results
            
        # Apply PSR to generic event to move 
            
        # Apply symmetry matrix if sym != 0
        if sym_idx != 0 :
            sym_matrices = ref_event["sym_matrix"]
            sym_matrix = sym_matrices[sym_idx]
            supposed_initial_positions = geometry.transform_positions(supposed_initial_positions, sym_matrix,0, ref_event["sym_perm"][sym_idx])
            saddle_positions = geometry.transform_positions(saddle_positions, sym_matrix,0, ref_event["sym_perm"][sym_idx])
            supposed_final_positions = geometry.transform_positions(supposed_final_positions, sym_matrix,0, ref_event["sym_perm"][sym_idx])
        supposed_initial_positions = geometry.transform_positions(supposed_initial_positions, psr_output.rotation_matrix, psr_output.translation_matrix, psr_output.permutation_matrix)
        saddle_positions = geometry.transform_positions(saddle_positions, psr_output.rotation_matrix, psr_output.translation_matrix, psr_output.permutation_matrix)
        supposed_final_positions= geometry.transform_positions(supposed_final_positions, psr_output.rotation_matrix, psr_output.translation_matrix, psr_output.permutation_matrix)

        # Move system do saddle positions
        neighbors = self.states[from_state].neighbors_list.get_neighbors('rcut', central_atom)

        if self.config.basin.style == "global" : 
            new_system.update_positions(supposed_final_positions, atom_idx=neighbors)
            min2_pos, _ = self.manager.global_minimize_with_results(self.config, positions=new_system.positions.copy())
            new_system.update_positions(min2_pos)

        elif self.config.basin.style == "global/reconstruction" : 
            new_system.update_positions(saddle_positions, atom_idx = neighbors)

            #Reconstruct the event
            #future = self.manager.minimize_with_results(self.config, positions=new_system.positions)
            #min_pos, _ = future.result()

            result = Reconstruction(self.config, self.manager).reconstruct(supposed_initial_positions, supposed_final_positions, new_system.positions, new_system.cell, self.config.psr.matching_score_thr, neighbors, central_atom=central_atom)
            if not result.is_ok() :
                return result
            new_system.update_positions(result.ok_value().min2_positions)
        
        else : 
            raise ValueError(f"Unknown {self.config.basin.style} style parameter.")

        return Ok(new_system)

    def _absorbing_rate(self, dE: float) -> float:
        """Scalar transition rate for an absorbing event, for the connectivity table.

        ``RateConstant.compute_rate`` returns a ``RateComponents``; the connectivity
        table's ``k_forward`` column is a float that gets summed (see ``exit``), so we
        take ``.rate``.
        """
        return self.rate_constant.compute_rate(dE).rate

    def _skip_refinement(self, idx: int, ctx: "dict | None", reason: str, n_skipped: int) -> int :
        """Per-row refinement failure: keep the row's exploration barrier/rate.

        The generic dE_forward/k_forward already in the table are exactly what the
        FPTA generator was built from for transient edges, so the table stays
        self-consistent; the row only misses the refined values. When a host-side
        PSR-transformed saddle is available it is stored so the row remains a
        selectable exit (the exit event is re-reconstructed and validated by KMC
        after the basin anyway); without one the exit is excluded from the draw.
        """
        state_connexion = int(self.connectivity_table.df.loc[idx].at["state_connexion"])
        if ctx is not None and ctx.get("fallback_saddle") is not None:
            self.absorbing_saddle_positions[state_connexion] = ctx["fallback_saddle"]
        else:
            self._exit_excluded_states.add(state_connexion)
        logger.warning(
            "[Basin] refine_absorbing: row %d (exit state %d) kept unrefined barrier (%s)",
            idx, state_connexion, reason,
        )
        return n_skipped + 1

    def refine_absorbing(self, system) :
        """When connectivity table is build, and that we have dict of states, we refine the energy barrier and k_forward of the transient -> absorbing event.

        Per-row failures are non-fatal: a row whose PSR or engine refinement fails
        keeps the exploration barrier/rate already in the table (see
        _skip_refinement) instead of aborting the whole basin — previously one bad
        row discarded an arbitrarily expensive exploration.
        """
        #compute the energy of the state
        #for all row in connectivity table where we need to refine
        n_skipped = 0
        futures_context = {} #idx → { "min": f_min, "saddle": f_sad }
        for idx, row in self.connectivity_table.df.iterrows() :
            if row["transient"]  == False : #need to refine
                #tmp_system = copy.deepcopy(self.states[row["state"]].system)
                tmp_system = System(positions=self.states[row["state"]].system.positions.copy(), types=self.states[row["state"]].system.types, cell=self.states[row["state"]].system.cell, pbc=True, index=np.arange(len(self.states[row["state"]].system.types)))
                #get tmp_system energy 
                future1 = self.manager.get_total_energy(positions=tmp_system.positions.copy()) #Send copy not reference
                #move to generic saddle positions 
                ref_event = self.reference_table.table[self.reference_table.table["idx_ref"] == row["event_connexion"]] 
                if ref_event.empty:
                    raise ValueError(f"idx_ref={row['event_connexion']} not found in reference table")
                ref_event = ref_event.iloc[0].copy()
                #ref_event = self.reference_table.table.iloc[row["event_connexion"]].copy()
                saddle_positions = ref_event["saddle_positions"].copy()
                #Apply PSR between event initial position and environment positions of the central_atoms


                #ENSURE "STATE" FULL 
                self.states[row["state"]].ensure_full_state(self.config)

                result = PointSetRegistration(self.config, tmp_system, ref_event , self.states[row["state"]].neighbors_list, row["central_atom"]).match()
                if not result.is_ok(): #PSR Err — no transform, so no fallback saddle either
                    n_skipped = self._skip_refinement(idx, None, f"PSR failed: {result.err_value()}", n_skipped)
                    continue
                    # Check if PointSetRegistration match is valid
                result = check_match(result, self.config.psr.matching_score_thr)
                if not result.is_ok() : #PSR matching score not valid :
                    n_skipped = self._skip_refinement(idx, None, f"PSR score: {result.err_value()}", n_skipped)
                    continue
                else :
                    psr_output = result.ok_value() #get psr results

                # Apply symmetry matrix if sym != 0
                if row["sym"] != 0 :
                    sym_matrices = ref_event["sym_matrix"]
                    sym_matrix = sym_matrices[row["sym"]]
                    saddle_positions = geometry.transform_positions(saddle_positions, sym_matrix,0, ref_event["sym_perm"][row["sym"]])
                saddle_positions = geometry.transform_positions(saddle_positions, psr_output.rotation_matrix, psr_output.translation_matrix, psr_output.permutation_matrix)
                neighbors = self.states[row["state"]].neighbors_list.get_neighbors("rcut", row["central_atom"])

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
                
                #save future in context (fallback_saddle: host-side PSR-transformed
                #saddle, used to keep the row selectable if engine refinement fails)
                futures_context[idx] = {
            "min": future1,
            "saddle": future2,
            "neighbors": neighbors,
            "fallback_saddle": np.array(saddle_positions, copy=True)}

                #RELEASE MEMORY :
                self.states[row["state"]].release_heavy_objects()

        #modify connectivity table entry future1 hold min energy, future2 holds E_saddle
        for idx, ctx in futures_context.items():
            try:
                E_min    = ctx["min"].result()
                result_sad = ctx["saddle"].result()
            except Exception as exc:
                #Engine/session transport failure (e.g. a remote LAMMPS 'Lost atoms'
                #raised through the error reply) — previously this killed the run.
                n_skipped = self._skip_refinement(idx, ctx, f"transport: {exc}", n_skipped)
                continue
            if not result_sad.is_ok() :
                n_skipped = self._skip_refinement(idx, ctx, str(result_sad.err_value()), n_skipped)
                continue
            E_sad = result_sad.ok_value().E_saddle
            if self.config.control.active_volume==True:
                dE = E_sad
            else:
                dE = E_sad - E_min
            k = self._absorbing_rate(dE)

            #also save saddle positions refined 
            idx_state = self.connectivity_table.df.loc[idx].at["state_connexion"]
            central_atom = self.connectivity_table.df.loc[idx].at["central_atom"]
            #self.absorbing_saddle_positions[idx_state] = result.ok_value().saddle_positions[self.states[idx_state].neighbors_list.get_neighbors("rcut", central_atom)]
            self.absorbing_saddle_positions[idx_state] = result_sad.ok_value().saddle_positions[ctx["neighbors"]]
            # update connectivity table row
            self.connectivity_table.df.loc[idx, "dE_forward"] = dE
            self.connectivity_table.df.loc[idx, "k_forward"] = k

        if n_skipped > 0:
            n_total = int((self.connectivity_table.df["transient"] == False).sum())  # noqa: E712
            logger.warning(
                "[Basin] refine_absorbing: %d/%d rows kept unrefined barriers",
                n_skipped, n_total,
            )
        return Ok(None)


    def is_new_state(self, system) :
        """Return the index of an equivalent known state, or -1 if the state is new.

        A cheap Chebyshev fingerprint pre-filter (see :mod:`pykmc.basins.fingerprinting`)
        narrows the candidates before the full structural comparison, which dominates
        deduplication cost as the basin grows.
        """
        fp_new = fingerprinting.compute_fingerprint(self.config, system.positions, system.cell, system.pbc)
        fp_tol = fingerprinting.fingerprint_tolerance(self.config)

        if fp_new is None:
            # fingerprint_mode = 'off': no pre-filter, structurally compare every state
            candidates = list(self.states.keys())
        else:
            # Vectorized fingerprint rejection: compare against all cached fingerprints at once
            fp_items = [
                (si, fp)
                for si, fp in self._state_fingerprints.items()
                if len(fp) == len(fp_new)
            ]
            if fp_items:
                indices, fps = zip(*fp_items)
                fp_matrix = np.vstack(fps)
                max_diffs = np.max(np.abs(fp_matrix - fp_new[np.newaxis, :]), axis=1)
                candidates = [indices[i] for i in np.where(max_diffs <= fp_tol)[0]]
            else:
                candidates = list(self.states.keys())

        for state_index in candidates:
            #.get(): defense-in-depth — a cache/states keyspace desync must degrade
            #to a missed pre-filter hit, not kill a multi-day run with KeyError.
            state_data = self.states.get(state_index)
            if state_data is None or state_data.system is None:
                continue
            are_equivalent = self.are_structures_equivalent(system.positions, state_data.system.positions, cell = system.cell)
            if are_equivalent :
                return state_index
        return -1

    @staticmethod
    def _wrap_positions(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
        """Wrap positions into [0, box) for cKDTree periodic queries."""
        box = np.diag(cell)
        return np.mod(positions, box)

    def are_structures_equivalent(self, pos1, pos2, cell, tol=0.3):

        if len(pos1) != len(pos2):
            return False

        # Wrap into the box before the boxsize-aware kd-tree query (correct periodic
        # nearest-neighbour even when minimized positions drift outside [0, box)).
        box = np.diag(cell).tolist()
        wrapped1 = self._wrap_positions(pos1, cell)
        wrapped2 = self._wrap_positions(pos2, cell)
        tree2 = cKDTree(wrapped2, boxsize=box)
        distances, _ = tree2.query(wrapped1, k=1)

        return np.max(distances) < tol

    def is_states_has_unknown_environments(self, state: StateData) : 
        if set(state.environment.atomic_environment_list).difference(self.known_environments) != set() :
            return True 
        else : 
            return False

    def _add_state(self, state_index, system=None, transient=True, applicable_events=None, visited=False, full=False ) :
        """Add a new state in the `self.states` dictionnary."""
        #to fit typing 
        neighbors_list  = []
        atomic_environment = []

        if full == True : 
            neighbors_list = NeighborsList(system, self.config.atomicenvironment.rnei, self.config.atomicenvironment.rcut)
            atomic_environment = AtomicEnvironment(self.config.atomicenvironment.style, neighbors_list.neighbors_list['rnei'], neighbors_list.neighbors_list['rcut'], self.config.atomicenvironment.neighbors_add, types=system.types if self.config.atomicenvironment.atom_coloring_mode == "full" else None, coordination_threshold=self.config.atomicenvironment.coordination_threshold)
        else : 
            neighbors_list = None 
            atomic_environment = None 
        new_state =  StateData(system=system, environment=atomic_environment, neighbors_list=neighbors_list, transient=transient,  visited=visited)

        self.states[state_index]= new_state
        #Cache the dedup fingerprint for this state (pre-filter for is_new_state).
        #None (fingerprint_mode = 'off') is not cached: with no pre-filter there is
        #nothing to compare against.
        if system is not None :
            fp = fingerprinting.compute_fingerprint(
                self.config, system.positions, system.cell, system.pbc
            )
            if fp is not None :
                self._state_fingerprints[state_index] = fp

    # ──────────────────────────────────────────────────────────────────
    # Wavefront-parallel basin exploration (strategy = 'wavefront')
    # ──────────────────────────────────────────────────────────────────

    def _transport_error(self, operation: str, exc: Exception) :
        """Wrap a session/engine transport failure as Err(MPI_REMOTE_ERROR)."""
        return Err(
            ErrorInfo(
                type=ErrorType.MPI_REMOTE_ERROR,
                message=f"{operation} failed: {exc}",
            )
        )

    def _prepare_reconstruct_kwargs(self, from_state: int, event_idx: int, central_atom: int, sym_idx: int) -> dict :
        """Prepare keyword arguments for manager.basin_reconstruct().

        Gathers all data needed by the engine to perform PSR + minimize.
        """
        ref_event = self.reference_table.table[self.reference_table.table["idx_ref"] == event_idx]
        if ref_event.empty:
            raise ValueError(f"idx_ref={event_idx} not found in reference table")
        ref_event = ref_event.iloc[0].copy()

        self.states[from_state].ensure_full_state(self.config)
        neighbor_indices = self.states[from_state].neighbors_list.get_neighbors("rcut", central_atom)

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
            #getattr: multi-element coloring ships on a separate branch; without it the
            #engine uses the uncolored ("grey") IRA matching path.
            "atom_coloring_mode": getattr(self.config.atomicenvironment, "atom_coloring_mode", "grey"),
        }

    def _result_from_mpi(self, mpi_result: "dict | None", from_state: int) -> "Result[System, ErrorInfo]" :
        """Convert an engine basin_reconstruct payload to Ok(System) or Err(ErrorInfo)."""
        if mpi_result is None or not mpi_result.get("ok"):
            error_type_str = mpi_result.get("error_type", "UNKNOWN") if mpi_result else "UNKNOWN"
            message = mpi_result.get("message", "Unknown error") if mpi_result else "No result from engine"
            error_type = getattr(ErrorType, error_type_str, ErrorType.MPI_REMOTE_ERROR)
            return Err(ErrorInfo(type=error_type, message=message))

        from_system = self.states[from_state].system
        # Route the engine's minimized positions through System.update_positions,
        # exactly like the serial path (system_from_state). It wraps into the cell
        # AND clamps tiny negative coordinates to zero - both needed: LAMMPS minimize
        # can leave coordinates slightly outside the box (it only re-wraps on
        # reneighbouring), and ase-style wrapping shifts boundary atoms (a slab layer
        # at exactly z = 0) slightly negative. Either case kills the periodic cKDTree
        # in NeighborsList.
        new_system = System(
            positions=from_system.positions.copy(),
            types=from_system.types,
            cell=from_system.cell,
            pbc=from_system.pbc,
            index=np.arange(len(from_system.types)))
        new_system.update_positions(np.array(mpi_result["min2_positions"], copy=True))
        return Ok(new_system)

    def _reconstruct_state_mpi(self, from_state: int, event_idx: int, central_atom: int, sym_idx: int) -> "Result[System, ErrorInfo]" :
        """Reconstruct a state via a session-pool engine (PSR + minimize); blocks on the future.

        The MPI counterpart of system_from_state() for use while the manager is in
        local (session-pool) mode.
        """
        kwargs = self._prepare_reconstruct_kwargs(from_state, event_idx, central_atom, sym_idx)
        future = self.manager.basin_reconstruct(**kwargs)
        try:
            mpi_result = future.result()
        except Exception as exc:
            return self._transport_error("basin_reconstruct", exc)
        return self._result_from_mpi(mpi_result, from_state)

    def _materialize_frontier_state(self, state_idx: int) -> "Result[int, ErrorInfo]" :
        """Reconstruct a frontier state so it can be treated as an absorbing exit."""
        if state_idx in self.states:
            return Ok(state_idx)

        from_state, event_idx, central_atom, sym_idx, is_transient = self.connectivity_table.get_transition_to_state(target_state=state_idx)
        result = self._reconstruct_state_mpi(from_state, event_idx, central_atom, sym_idx)
        if not result.is_ok():
            return result

        new_system = result.ok_value()
        existing_state = self.is_new_state(new_system)
        if existing_state != -1:
            self.connectivity_table.change_state_index(current_index=state_idx, new_index=existing_state)
            return Ok(existing_state)

        self._add_state(state_index=state_idx, system=new_system, transient=is_transient)
        self.states[from_state].release_heavy_objects()
        return Ok(state_idx)

    def _budget_breach_reason(self, n_explored: int, t_start: float) -> "str | None" :
        """Return the breached [BASIN] budget as a label, or None when within budget.

        Never fires before the entry state was explored (n_explored == 0): a basin
        with zero explored states has no connectivity rows and no possible exit, so
        capping it would only produce a degenerate table.
        """
        bc = self.config.basin
        if bc is None or n_explored == 0:
            return None
        if bc.max_states is not None and n_explored >= bc.max_states:
            return f"max_states={bc.max_states}"
        if bc.max_total_states is not None and \
                len(self.states) + len(self._deferred_states) + self._n_failed >= bc.max_total_states:
            return f"max_total_states={bc.max_total_states}"
        if bc.max_basin_walltime_s is not None and \
                time.perf_counter() - t_start >= bc.max_basin_walltime_s:
            return f"max_basin_walltime_s={bc.max_basin_walltime_s}"
        return None

    def _cap_remaining_as_absorbing(self, reason: str = "max_states") -> "Result[None, ErrorInfo]" :
        """Convert all remaining frontier states to absorbing WITHOUT reconstructing them.

        Unmaterialized frontier states become 'deferred': they stay selectable as
        exits (their rows keep the exploration barriers) and are materialized lazily
        only if the exit draw picks one. The previous implementation reconstructed
        and deduplicated the whole frontier here, re-entering the very bottleneck
        the cap exists to avoid. Deferred states are not deduplicated against each
        other, so duplicates split exit probability among themselves — an accepted
        bias of an explicitly budget-breached basin (the selected exit is deduped at
        materialization time).
        """
        self._was_capped = True
        capped = list(self.states_to_explore)
        for state_idx in capped:
            self.connectivity_table.change_state_to_absorbing(state_idx)
            if state_idx in self.states:
                self.states[state_idx].transient = False
                self.states[state_idx].release_heavy_objects()
            else:
                self._deferred_states.add(state_idx)
            if state_idx not in self.explored_states:
                self.explored_states.append(state_idx)
        self.states_to_explore.clear()
        logger.warning(
            "[Basin] Budget breach (%s): capped %d frontier states as deferred absorbing.",
            reason,
            len(capped),
        )
        return Ok(None)

    def _mark_failed_absorbing(self, state_idx: int, err: "ErrorInfo") -> None :
        """Shared rule for a state whose reconstruction failed: keep the connectivity
        row (the channel physically exists and its exploration barrier is known), flip
        it absorbing, and exclude it from the exit draw. The parent's total escape
        rate — and with it t_exit and k_tot — stays exact; only the landing draw is
        conditioned on 'not a failed state'."""
        self.connectivity_table.change_state_to_absorbing(state_idx)
        self._exit_excluded_states.add(state_idx)
        self._n_failed += 1
        if state_idx in self.states_to_explore:
            self.states_to_explore.remove(state_idx)
        if state_idx not in self.explored_states:
            self.explored_states.append(state_idx)
        logger.warning(
            "[Basin] state %d kept as non-selectable absorbing (reconstruction failed: %s)",
            state_idx,
            err,
        )

    def _materialize_exit_state(self, exit_state: int) -> "Result[int, ErrorInfo]" :
        """Ensure the selected exit state has a System, materializing lazily if needed.

        Returns the (possibly merged) state index: a deferred state can turn out to
        be a duplicate of an already-known state once reconstructed.
        """
        state_data = self.states.get(exit_state)
        if state_data is not None and state_data.system is not None:
            return Ok(exit_state)

        if exit_state in self._deferred_systems:
            #Reconstructed during exploration but never deduplicated (dedup deadline):
            #dedup now, against the full state set.
            new_system = self._deferred_systems.pop(exit_state)
            self._deferred_states.discard(exit_state)
            existing = self.is_new_state(new_system)
            if existing != -1:
                self.connectivity_table.change_state_index(current_index=exit_state, new_index=existing)
                #Never transplant a saddle onto an excluded state: its rows belong
                #to a different transition geometry.
                if existing not in self.absorbing_saddle_positions \
                        and existing not in self._exit_excluded_states \
                        and exit_state in self.absorbing_saddle_positions:
                    self.absorbing_saddle_positions[existing] = self.absorbing_saddle_positions[exit_state]
                return Ok(existing)
            self._add_state(state_index=exit_state, system=new_system, transient=False)
            return Ok(exit_state)

        #Never reconstructed (budget cap): one engine reconstruction + dedup.
        self._deferred_states.discard(exit_state)
        result = self._materialize_frontier_state(exit_state)
        if not result.is_ok():
            return result
        materialized = result.ok_value()
        if materialized != exit_state:
            if materialized not in self.absorbing_saddle_positions \
                    and materialized not in self._exit_excluded_states \
                    and exit_state in self.absorbing_saddle_positions:
                self.absorbing_saddle_positions[materialized] = self.absorbing_saddle_positions[exit_state]
        else:
            self.connectivity_table.change_state_to_absorbing(exit_state)
            if exit_state in self.states:
                self.states[exit_state].transient = False
        return Ok(materialized)

    def _resolve_exit_state(self, t_exit: float, exit_state: int) -> "Result[int, ErrorInfo]" :
        """Materialize the drawn exit; on failure exclude it and redraw another.

        t_exit is not recomputed: excluding an exit only conditions the landing draw,
        not the generator, so only select_absorbing_state() re-runs.
        """
        while True:
            result = self._materialize_exit_state(exit_state)
            if result.is_ok():
                resolved = result.ok_value()
                n_transient, _ = self._connectivity_state_counts()
                if resolved >= n_transient and resolved not in self._exit_excluded_states:
                    return Ok(resolved)
                #Merged into a transient or excluded state: not a viable exit;
                #exclude and redraw.
                logger.warning(
                    "[Basin] exit state %d resolved to transient or excluded state %d; redrawing exit",
                    exit_state, resolved,
                )
            else:
                logger.warning(
                    "[Basin] exit state %d failed materialization (%s); redrawing exit",
                    exit_state, result.err_value(),
                )
            self._exit_excluded_states.add(exit_state)
            exit_state = self.selector.select_absorbing_state(
                t_exit, excluded_states=self._exit_excluded_states)
            if exit_state is None:
                return Err(ErrorInfo(
                    type=ErrorType.BASIN_NO_VIABLE_EXIT,
                    message="all absorbing exits failed materialization or were excluded"))

    def _prepare_explore_kwargs(self, state_idx: int, start_index: int) -> dict :
        """Prepare keyword arguments for manager.basin_explore()."""
        import pickle

        self.states[state_idx].ensure_full_state(self.config)
        state = self.states[state_idx]

        config_dict = {
            "rnei": self.config.atomicenvironment.rnei,
            "rcut": self.config.atomicenvironment.rcut,
            "neighbors_add": self.config.atomicenvironment.neighbors_add,
            "ae_style": self.config.atomicenvironment.style,
            #getattr: multi-element coloring / coordination styles ship on a separate
            #branch; the engine builds its AtomicEnvironment kwargs from these only
            #when they carry non-default values.
            "atom_coloring_mode": getattr(self.config.atomicenvironment, "atom_coloring_mode", "grey"),
            "coordination_threshold": getattr(self.config.atomicenvironment, "coordination_threshold", None),
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

    def _explore_states_parallel(self, states_batch: list[int], n_workers: int = 4) -> "Result[None, ErrorInfo]" :
        """Explore multiple transient states in parallel via the MPI session pool.

        Each engine explores with local indices starting from 0. After collection,
        rows are remapped to contiguous global indices by adding an offset equal to
        ``_next_state_index`` at merge time (no index gaps, no index reuse).
        """
        if not states_batch:
            return Ok(None)

        # Submit all exploration tasks with local (zero-based) indices
        futures = {}
        for state_idx in states_batch:
            kwargs = self._prepare_explore_kwargs(state_idx, start_index=0)
            futures[state_idx] = self.manager.basin_explore(**kwargs)

        # Collect results sequentially; remap local -> global indices
        for _state_idx, future in futures.items():
            try:
                rows = future.result()
            except Exception as exc:
                return self._transport_error("basin_explore", exc)
            if isinstance(rows, dict) and not rows.get("ok", True):
                return Err(ErrorInfo(
                    type=ErrorType.MPI_REMOTE_ERROR,
                    message=f"basin_explore failed on engine: {rows.get('message', 'unknown')}"))
            if rows:
                offset = self._next_state_index
                for row in rows:
                    row["state_connexion"] += offset
                local_max = max(r["state_connexion"] for r in rows)
                self._next_state_index = local_max + 1
                self.connectivity_table.add_connectivity_batch(rows)

        return Ok(None)

    def is_new_state_batch(self, new_systems: "dict[int, System]", deadline: "float | None" = None) -> dict[int, int] :
        """Check multiple reconstructed systems for duplicates at once.

        Unlike serial is_new_state(), this also cross-checks the new states against
        each other: two states reconstructed in the same wavefront can be duplicates
        of each other, and missing that leads to exponential blowup of the basin.

        Parameters
        ----------
        new_systems : dict[int, System]
            Mapping state_idx -> System for newly reconstructed states.
        deadline : float, optional
            time.perf_counter() value after which remaining states are not compared
            at all and get DEDUP_DEFERRED instead (deduplication is the dominant
            basin cost; this bounds a single batch's dedup to the wall-time budget
            with per-state granularity).

        Returns
        -------
        dict[int, int]
            Mapping state_idx -> existing_state_idx for duplicates,
            state_idx -> -1 for genuinely new states,
            state_idx -> DEDUP_DEFERRED for states cut off by the deadline.

        """
        results = {}

        # Early-out: the deadline can already be spent before any comparison —
        # skip the per-state pre-work (fingerprints, tree builds) entirely.
        if deadline is not None and time.perf_counter() >= deadline:
            return {idx: DEDUP_DEFERRED for idx in new_systems}

        # Pre-compute fingerprints for the new systems
        new_fingerprints = {}
        for idx, system in new_systems.items():
            new_fingerprints[idx] = fingerprinting.compute_fingerprint(
                self.config, system.positions, system.cell, system.pbc)

        # Build kd-trees for existing states (None marks mixed-PBC fallback)
        existing_trees = {}
        for idx, state_data in self.states.items():
            if state_data.system is not None:
                if state_data.system.pbc is None or np.all(state_data.system.pbc):
                    box = np.diag(state_data.system.cell).tolist()
                    wrapped = self._wrap_positions(state_data.system.positions, state_data.system.cell)
                    existing_trees[idx] = cKDTree(wrapped, boxsize=box)
                else:
                    existing_trees[idx] = None  # fallback to manual comparison

        existing_fp_items = list(self._state_fingerprints.items())
        fp_tol = fingerprinting.fingerprint_tolerance(self.config)

        for new_idx, system in new_systems.items():
            if deadline is not None and time.perf_counter() >= deadline:
                results[new_idx] = DEDUP_DEFERRED
                continue

            match = -1
            fp_new = new_fingerprints[new_idx]

            # Fingerprint pre-filter against existing states
            # (fp_new None = fingerprint_mode 'off': compare against every state)
            if fp_new is not None and existing_fp_items:
                candidate_indices = []
                for si, fp in existing_fp_items:
                    if len(fp) == len(fp_new) and np.max(np.abs(fp - fp_new)) <= fp_tol:
                        candidate_indices.append(si)
            else:
                candidate_indices = list(existing_trees.keys())

            for existing_idx in candidate_indices:
                if existing_idx not in existing_trees:
                    continue
                tree = existing_trees[existing_idx]
                if tree is not None:
                    wrapped_query = self._wrap_positions(system.positions, system.cell)
                    distances, _ = tree.query(wrapped_query, k=1)
                    if np.max(distances) < 0.3:
                        match = existing_idx
                        break
                else:
                    state_data = self.states[existing_idx]
                    if self.are_structures_equivalent(system.positions, state_data.system.positions,
                                                      cell=system.cell):
                        match = existing_idx
                        break

            # Cross-check within this batch (two new states may be duplicates of each other)
            if match == -1:
                for other_idx in list(results.keys()):
                    if results[other_idx] != -1:
                        continue  # this one is already a duplicate itself
                    if other_idx in new_systems:
                        fp_other = new_fingerprints[other_idx]
                        # Fingerprint pre-filter within the batch (skipped when 'off')
                        if (fp_new is not None and fp_other is not None
                                and len(fp_other) == len(fp_new)
                                and np.max(np.abs(fp_other - fp_new)) > fp_tol):
                            continue
                        if self.are_structures_equivalent(system.positions,
                                                          new_systems[other_idx].positions,
                                                          cell=system.cell):
                            match = other_idx
                            break

            results[new_idx] = match
        return results

    def _reconstruct_wavefront_batch(self, to_reconstruct: list[int]) -> "Result[tuple, ErrorInfo]" :
        """Phase A: reconstruct one wavefront's worth of states via the session pool.

        Per-state failures are non-fatal: failed states are reported in the
        ``failed`` mapping and handled by _register_wavefront_states (kept as
        non-selectable absorbing states). Only a whole-batch transport failure —
        the session pool itself is gone — aborts the basin.
        """
        reconstructed = {}
        transition_info = {}
        failed: dict[int, "ErrorInfo"] = {}
        if not to_reconstruct:
            return Ok((reconstructed, transition_info, failed))

        for state_idx in to_reconstruct:
            transition_info[state_idx] = self.connectivity_table.get_transition_to_state(target_state=state_idx)

        futures = {}
        for state_idx in to_reconstruct:
            from_state, event_idx, central_atom, sym_idx, is_transient = transition_info[state_idx]
            kwargs = self._prepare_reconstruct_kwargs(from_state, event_idx, central_atom, sym_idx)
            futures[state_idx] = (from_state, self.manager.basin_reconstruct(**kwargs))

        transport_failed: set[int] = set()
        for state_idx, (from_state, future) in futures.items():
            try:
                mpi_result = future.result()
            except Exception as exc:
                result = self._transport_error("basin_reconstruct", exc)
                logger.warning("[Basin] Reconstruction transport failed for state %d: %s", state_idx, result.err_value())
                failed[state_idx] = result.err_value()
                transport_failed.add(state_idx)
                continue
            result = self._result_from_mpi(mpi_result, from_state)
            if result.is_ok():
                reconstructed[state_idx] = result.ok_value()
                continue

            logger.warning("[Basin] Reconstruction failed for state %d: %s", state_idx, result.err_value())
            failed[state_idx] = result.err_value()

        #Dead-pool guard: only the future.result() exception path can evidence a
        #dead pool — an engine that delivered an {"ok": False} payload is alive by
        #construction, so engine-reported failures (whatever their error_type
        #string) stay per-state. Abort only when EVERY state failed at the
        #transport layer.
        if failed and len(failed) == len(to_reconstruct) and transport_failed == set(failed):
            return Err(next(iter(failed.values())))
        return Ok((reconstructed, transition_info, failed))

    def _register_wavefront_states(self, to_reconstruct: list[int], reconstructed: "dict[int, System]", transition_info: dict, prof: dict, failed: "dict[int, ErrorInfo] | None" = None, deadline: "float | None" = None) -> "tuple[list[int], int, int]" :
        """Phase B: dedup the reconstructed batch and register the genuinely new states."""
        #Single-state batches go through the batch path too: it is equivalent for
        #size 1 (the intra-batch cross-check is a no-op) and honors the deadline.
        dedup_results = self.is_new_state_batch(reconstructed, deadline=deadline) if reconstructed else {}

        new_transient = []
        n_duplicates = 0
        n_absorbing = 0
        for state_idx in to_reconstruct:
            if failed and state_idx in failed:
                self._mark_failed_absorbing(state_idx, failed[state_idx])
                n_absorbing += 1
                continue

            existing = dedup_results.get(state_idx, -1)
            if existing == DEDUP_DEFERRED:
                #Dedup deadline hit: keep the already-paid-for reconstruction for
                #lazy materialization, flip the row absorbing. The loop-top budget
                #check fires next and caps the rest of the frontier.
                self.connectivity_table.change_state_to_absorbing(state_idx)
                self._deferred_states.add(state_idx)
                self._deferred_systems[state_idx] = reconstructed[state_idx]
                if state_idx in self.states_to_explore:
                    self.states_to_explore.remove(state_idx)
                if state_idx not in self.explored_states:
                    self.explored_states.append(state_idx)
                n_absorbing += 1
                continue
            if existing != -1:
                self.connectivity_table.change_state_index(current_index=state_idx, new_index=existing)
                self.explored_states.append(state_idx)
                self.states_to_explore.remove(state_idx)
                n_duplicates += 1
                continue

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

        return new_transient, n_duplicates, n_absorbing

    def _write_timing_checkpoint(self, prof: dict, elapsed: float, n_transient: int, n_absorbing: int, n_duplicates: int, n_processed: int) -> None :
        """Write timing summary files readable by toolkit/basin_testing/compare_scaling.py."""
        strategy = self.config.basin.strategy if self.config.basin is not None else "serial"
        n_workers = self.config.basin.n_workers if self.config.basin is not None else 1
        n_conn = len(self.connectivity_table.df) if not self.connectivity_table.df.empty else 0

        ckpt_path = f"basin_timing_{strategy}.txt"
        with open(ckpt_path, "w") as f:
            f.write("# Basin timing checkpoint\n")
            f.write(f"strategy = {strategy}\n")
            f.write(f"n_workers = {n_workers}\n")
            f.write(f"wall_time_s = {elapsed:.3f}\n")
            f.write(f"states_transient = {n_transient}\n")
            f.write(f"states_absorbing = {n_absorbing}\n")
            f.write(f"states_total = {n_transient + n_absorbing}\n")
            f.write(f"connectivity_rows = {n_conn}\n")
            f.write(f"n_duplicates = {n_duplicates}\n")
            f.write(f"n_processed = {n_processed}\n")
            f.write(f"n_failed = {self._n_failed}\n")
            for phase, t in sorted(prof.items(), key=lambda x: -x[1]):
                pct = 100.0 * t / elapsed if elapsed > 0 else 0
                f.write(f"prof_{phase} = {t:.3f}\n")
                f.write(f"pct_{phase} = {pct:.1f}\n")
        logger.info("[Basin] Timing checkpoint written to %s", ckpt_path)

        # Also write the level_complete format for compare_scaling.py compatibility
        level_path = "basin_connectivity_0_L0_level_complete.txt"
        with open(level_path, "w") as f:
            f.write("# Basin level complete checkpoint\n")
            f.write("level = 0\n")
            f.write(f"wall_time_s = {elapsed:.3f}\n")
            f.write(f"level_wall_time_s = {elapsed:.3f}\n")
            f.write(f"states_total = {n_transient + n_absorbing}\n")
            f.write(f"connectivity_rows = {n_conn}\n")

    def _finalize_exploration_run(self, t_start: float, prof: dict, n_processed: int, n_duplicates: int, strategy_label: "str | None" = None) -> "Result[None, ErrorInfo]" :
        """Log the run summary + profiling breakdown and write timing checkpoints."""
        elapsed = time.perf_counter() - t_start
        n_transient, n_absorbing_final = self._connectivity_state_counts()
        label = f" ({strategy_label})" if strategy_label else ""
        logger.info(
            "[Basin] COMPLETE%s: %d transient + %d absorbing states | %d connectivity rows | processed=%d | duplicates=%d | %.1fs",
            label,
            n_transient,
            n_absorbing_final,
            len(self.connectivity_table.df),
            n_processed,
            n_duplicates,
            elapsed,
        )

        top_level = {k: v for k, v in prof.items() if k not in ("other", "psr", "minimize")}
        prof["other"] = elapsed - sum(top_level.values())
        for phase, t in sorted(top_level.items(), key=lambda x: -x[1]):
            pct = 100.0 * t / elapsed if elapsed > 0 else 0
            logger.info("[Basin] PROFILING:   %-15s %8.2fs  %5.1f%%", phase, t, pct)

        self._write_timing_checkpoint(prof, elapsed, n_transient, n_absorbing_final, n_duplicates, n_processed)

        #Failure budget: sparse failures only perturb individual channels (the rate
        #mass is preserved and the exit draw skips them); systematic failures mean
        #the basin cannot be trusted, so fall back to the plain KMC event.
        if self._n_reconstruction_attempts > 0:
            failed_fraction = self._n_failed / self._n_reconstruction_attempts
            max_failed = self.config.basin.max_failed_fraction if self.config.basin is not None else 0.2
            if failed_fraction > max_failed:
                return Err(ErrorInfo(
                    type=ErrorType.BASIN_TOO_MANY_FAILED_STATES,
                    message=f"{self._n_failed}/{self._n_reconstruction_attempts} state reconstructions "
                            f"failed (max_failed_fraction = {max_failed})"))
        return Ok(None)

    def construct_connexion_table_parallel(self) -> "Result[None, ErrorInfo]" :
        """Wavefront-parallel BFS: process whole frontiers instead of one state at a time.

        Phases per wavefront:
            A. Batch reconstruction (PSR + minimize) across the MPI session pool
            B. Batch deduplication (incl. intra-batch cross-check)
            C. Parallel exploration of the new transient states
            D. Merge and update the frontier queue
        """
        n_workers = self.config.basin.n_workers if self.config.basin is not None else 1
        max_frontier = self.config.basin.max_frontier_size if self.config.basin is not None else None
        max_walltime = self.config.basin.max_basin_walltime_s if self.config.basin is not None else None

        t_start = time.perf_counter()
        #Deadline threaded into the dedup hot loop: the observed pathology was a
        #single frontier whose deduplication alone consumed hours between budget
        #checks, so the check must live between states inside the batch.
        dedup_deadline = (t_start + max_walltime) if max_walltime is not None else None
        n_explored = 0
        n_duplicates = 0
        n_absorbing = 0
        n_processed = 0
        prof = {"reconstruct": 0.0, "psr": 0.0, "minimize": 0.0,
                "dedup": 0.0, "ensure_state": 0.0, "explore": 0.0,
                "merge": 0.0, "other": 0.0}

        # Switch to the session pool for basin reconstruction (parallel minimization)
        self.manager.use_local()

        t_last_log = t_start

        try:
            while len(self.states_to_explore) != 0:
                # Budget checks (max_states / max_total_states / max_basin_walltime_s)
                reason = self._budget_breach_reason(n_explored, t_start)
                if reason is not None:
                    result = self._cap_remaining_as_absorbing(reason)
                    if not result.is_ok():
                        return result
                    break

                batch = list(self.states_to_explore)
                # Frontier chunking: bounds per-level memory and gives the wall-time
                # check chunk granularity. Later chunks dedup against earlier-chunk
                # states through self.states; update_to_explore() re-derives the
                # frontier each iteration, so leftover states are picked up next pass.
                if max_frontier is not None and len(batch) > max_frontier:
                    batch = batch[:max_frontier]
                # Clamp the batch to the remaining max_total_states capacity so the
                # overshoot is ~0 instead of one whole frontier (leftover states are
                # capped/deferred at the next loop-top check).
                bc = self.config.basin
                if bc is not None and bc.max_total_states is not None:
                    remaining = bc.max_total_states - (
                        len(self.states) + len(self._deferred_states) + self._n_failed)
                    if len(batch) > remaining:
                        batch = batch[:max(remaining, 1)]

                # Separate: states that need reconstruction vs state 0 (already exists)
                to_reconstruct = [s for s in batch if s not in self.states]
                already_exist = [s for s in batch if s in self.states]

                # ── Phase A: Batch reconstruction ──
                reconstructed = {}
                transition_info = {}
                failed = {}
                if to_reconstruct:
                    t0 = time.perf_counter()
                    result = self._reconstruct_wavefront_batch(to_reconstruct)
                    prof["reconstruct"] += time.perf_counter() - t0
                    if not result.is_ok():
                        return result
                    reconstructed, transition_info, failed = result.ok_value()
                    n_processed += len(to_reconstruct)
                    self._n_reconstruction_attempts += len(to_reconstruct)

                # ── Phase B: Batch deduplication ──
                # Always batch dedup in the wavefront loop: serial is_new_state() only
                # checks against self.states (not the other batch members), so duplicates
                # within the same wavefront would go undetected.
                t0 = time.perf_counter()
                new_transient, duplicates_delta, absorbing_delta = self._register_wavefront_states(
                    to_reconstruct,
                    reconstructed,
                    transition_info,
                    prof,
                    failed=failed,
                    deadline=dedup_deadline,
                )
                prof["dedup"] += time.perf_counter() - t0
                n_duplicates += duplicates_delta
                n_absorbing += absorbing_delta

                # Also include pre-existing states that still need exploration (e.g. state 0)
                for state_idx in already_exist:
                    if state_idx in self.states_to_explore:
                        new_transient.append(state_idx)

                # ── Phase C: Exploration via MPI engines ──
                if new_transient:
                    t0 = time.perf_counter()
                    result = self._explore_states_parallel(new_transient, n_workers=n_workers)
                    prof["explore"] += time.perf_counter() - t0
                    if not result.is_ok():
                        return result

                    # Mark explored
                    for state_idx in new_transient:
                        if state_idx in self.states_to_explore:
                            self.states_to_explore.remove(state_idx)
                        if state_idx not in self.explored_states:
                            self.explored_states.append(state_idx)
                        self.states[state_idx].release_heavy_objects()
                        n_explored += 1

                # ── Phase D: Update the frontier queue ──
                t0 = time.perf_counter()
                self.update_to_explore()
                prof["merge"] += time.perf_counter() - t0

                elapsed = time.perf_counter() - t_start
                logger.debug("[Basin] wavefront done | explored=%d | to_explore=%d | states=%d | dup=%d | abs=%d | %.1fs",
                             n_explored, len(self.states_to_explore), len(self.states), n_duplicates, n_absorbing, elapsed)

                # Periodic progress log (every 30s)
                now = time.perf_counter()
                if now - t_last_log >= 30.0:
                    logger.info(
                        "[Basin] PROGRESS (wavefront): explored=%d | to_explore=%d | states=%d | duplicates=%d | absorbing=%d | %.0fs elapsed",
                        n_explored, len(self.states_to_explore), len(self.states), n_duplicates, n_absorbing, now - t_start,
                    )
                    t_last_log = now
        finally:
            self.manager.use_global()
        return self._finalize_exploration_run(
            t_start,
            prof,
            n_processed,
            n_duplicates,
            strategy_label="wavefront",
        )
