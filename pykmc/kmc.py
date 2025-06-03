from pykmc import System, Engine, NeighborsList, AtomicEnvironment, PointSetRegistration, LogKMC, LOGGING_CONFIG, ActiveEventTable, ReferenceEventTable
import random 
from .result import EventSearchOutput, KMCLoopInfo, Err, ErrorInfo, ErrorType, Result, AtomicEnvironmentInfo, ReferenceEventSearchInfo, ReferenceValidEventsInfo
import numpy as np
from .utils import geometry
from ase.io import write
from ase import Atoms
import ase.geometry
from .algorithms import * 
import sys
import pandas as pd
from .rate_constant import compute_rate_Eyring
from dataclasses import asdict
import pickle

#TODO fix reconstruction = False

class KMC() : 

    def __init__(self, config) : 
        self.config = config 
        self.loggers = None
        self.system = None 
        self.engine = None 
        self.neighbors_list = None 
        self.atomic_environment = None 
        self.reference_table = None
        self.visited_environments = None 
            
    def run(self) : 
        
        #Initialize the simulation, KMC attributes and minimize the system
        self._initialize()
        #Write initial step to file  
        self._append_snapshot_to_trajectory()

        #LOOP KMC PARAMETERS
        nkmc_steps = self.config.control.n_steps
        time = 0.0 #in seconds
        nsearch = self.config.eventsearch.nsearch

        #KMC LOOP
        for step in range(nkmc_steps) :
            #FIND NEW GENERIC EVENTS 
                ##=> Find new atomic environments that have not been visited
            new_environments = list(set(self.atomic_environment.atomic_environment_list).difference(self.visited_environments)) 

                ##=>List of atoms(central) on which we gonna perfom an event search
            central_atom_research_list = self.central_atoms_research(new_environments, nsearch)

                ##=>Perform event serach on each atom in central_atom_research_list
            results_reference_event_searches = self.reference_event_searches(central_atom_research_list)

                ##=>Construct informations event searches for output 
            reference_event_searches_info = self.info_reference_event_searches(results_reference_event_searches)

                ##=>Find only success event searches 
            results_reference_event_searches = [e.ok_value() for e in results_reference_event_searches if e.is_ok()]

                ##=>Center event to prevent pbc problem with psr
            results_reference_event_searches = [self._center_event_positions(e) for e in results_reference_event_searches]

            #ADD NEW GENERIC EVENTS TO REFERENCE EVENT TABLE
                ##=>Check if the event is valid, ie if not already present and has a valid energy barrier
            results_is_valid_events = self.is_valid_events(results_reference_event_searches)  

                ##=>Construct informations valid events for output 
            is_valid_events_info = self.info_is_valid_events(results_is_valid_events)
            
            

            #MAX_TRIES = 5 #Number of tentatives to prevent emtpy reference table : 
            #tries = 0 
            #while tries < MAX_TRIES : 
            #    fails = 0 #to count the number of event search fails

            #=> For all central atom index on which we want to perform an event search  
            for idx in central_atom_research_list : 

                #==> Do an event search 
                #results = self.engine.search_event(self.system, idx)
                event_search_output = self.engine.search_event(self.system, idx)
                #==> Check if event found
                #if results != None : 
                if event_search_output.is_ok() : 
                    event_search_output = event_search_output.ok_value()
                #add results in reference table 
                    if self.config.control.reconstruction : #if reconstruction, need to center event to prevent pbc problem
                        #results = (*self._center_event_positions(results[0], results[1], results[2], results[3]), *results[3:])
                        self._center_event_positions(event_search_output)
                    #is_new, in_e_bounds = self.reference_table.add_event(*results, self.neighbors_list.neighbors_list['rcut'], self.system.cell)
                    valid_dfevents = self.reference_table.is_valid_new_event(min1_positions = event_search_output.min1_positions, 
                                                                  saddle_positions = event_search_output.saddle_positions, 
                                                                  min2_positions = event_search_output.min2_positions, 
                                                                  move_atom_idx = event_search_output.move_atom_index, 
                                                                  dE_forward = event_search_output.dE_forward, 
                                                                  dE_backward = event_search_output.dE_backward, 
                                                                  cell= self.system.cell)
                    if valid_dfevents.is_ok() : 
                        self.reference_table.add_event(valid_dfevents.ok_value())

                #else : #failed 
                #    fails += 1
            #=> if reference event table is not emppty break while loop 
            #if len(self.reference_table.table) > 0 : 
            #    break #end while loop
            #else : 
            #    tries += 1 
            #    #self.logger.logger.debug('Empty reference table after {} searches, retrying'.format(len(central_atom_research_list)))
            #else : #if not breack encounter : 
            #    #self.logger.logger.debug('Emtpy reference table after {} tries, closing simulation'.format(MAX_TRIES))
            if len(self.reference_table.table) == 0 : 
                self._close()

            ################################
            #CONTRUCT CURRENT ACTIVE EVENTS#
            ################################
            active_table = self.refinement() 
            ###############################
            #SELECT EVENT IN ACTIVE TABLE # 
            ###############################
            idx_selected_event, delta_t = self._select_event(active_table)
            self._apply_event(idx_selected_event, active_table)
            time += delta_t*10**-12 #time is in seconds

            #MINIMIZE (could remove)
            new_positions = self.engine.minimize(self.system)
            self.system.update_positions(new_positions)
            self._append_snapshot_to_trajectory()

            #=>if no reconstruction, new reference table
            if not self.config.control.reconstruction: 
                self.reference_table = ReferenceEventTable(self.config)
            else : #update visited environments 
                l_ids = list(set(self.atomic_environment.atomic_environment_list)) 
                self.visited_environments.update(set(l_ids).difference(self.visited_environments))

                #self.logger.table_line_info_kmc(step, time, len(list(set(self.atomic_environment.atomic_environment_list))), len(self.catalog.catalog), idx_event_catalog, self.catalog.catalog.loc[idx_event_catalog].at['energy_barrier'], is_reconstruction )
                #self.logger.table_line_info_kmc(step, time, len(list(set(self.atomic_environment.atomic_environment_list))), len(self.reference_table.table),  active_table.table.loc[idx_selected_event].at['energy_barrier'], active_table.table.loc[idx_selected_event].at['k'] )
            #update neighborlist : 
            self.neighbors_list = NeighborsList(self.system, self.config.atomicenvironment.rnei, self.config.atomicenvironment.rcut) 
            #update atomic environment 
            self.atomic_environment = AtomicEnvironment(self.config.atomicenvironment.style, self.neighbors_list.neighbors_list['rnei'], self.neighbors_list.neighbors_list['rcut'], self.config.atomicenvironment.neighbors_add)

            if set(list(self.atomic_environment.atomic_environment_list)) == {"crystal"} : 
                #self.logger.logger.info(':=> Only atoms with cristalline environment')
                self._close()

            #Loop Informations : 
            atomic_environment_info = self.info_atomic_environments(new_environments)
            kmc_loop_info = KMCLoopInfo(step = step, 
                                        atomic_environment_info=atomic_environment_info,
                                        reference_event_searches_info=reference_event_searches_info
                                        ) 
            self.loggers.info('info', kmc_loop_info.output_msg())
            print(kmc_loop_info.output_msg())

            self._save()
        


    def central_atoms_research(self, new_environments: list[str | bytes], nsearch: int) -> list[int]: 
        """Generate list of central atoms on which we gonna perform generic event searches for the reference table

        For each new environment it adds nseach atoms having that environment to the list.

        Parameters
        ----------
        new_environment : list[str  |  bytes]
            List of atomic environment ID.
        nsearch : int
            Number of searches per atomic environment. 

        Returns
        -------
        list[int]
            List of central atoms

        Raises
        ------
        IndexError
            If no atoms are found for a given environment, random.choice will raise an IndexError.

        """
        central_atom_research_list = []
        #for each atomic environment hash in new_environment 
        for env in new_environments :
            #find all index having that hash
            tmp1 = [i for i,e in enumerate(self.atomic_environment.atomic_environment_list) if e == env] 
            #Randomly choose nsearch atoms that have that environment 
            tmp2 = [random.choice(tmp1) for _i in range(nsearch)]
            central_atom_research_list += tmp2
        return central_atom_research_list


    def reference_event_searches(self, central_atom_research_list: list[int] ) -> list[Result[EventSearchOutput, ErrorInfo]] : 
        results = [] 
        for at_idx in central_atom_research_list : 
            event_search_output = self.engine.search_event(self.system, at_idx)
            results.append(event_search_output)
        return results
    
    def is_valid_events(self, results_reference_event_searches: list[EventSearchOutput]) -> list[Result[pd.DataFrame, ErrorInfo]] : 
        results_is_valid_events = [] 
        for result in results_reference_event_searches : 
            results_is_valid_events.append(self.reference_table.is_valid_new_event(min1_positions = result.min1_positions, 
                                                    saddle_positions = result.saddle_positions, 
                                                    min2_positions = result.min2_positions, 
                                                    move_atom_idx = result.move_atom_index, 
                                                    dE_forward = result.dE_forward, 
                                                    dE_backward = result.dE_backward, 
                                                    cell= self.system.cell))
        return results_is_valid_events
        
            

    def analyse_reference_event_searches(self, results_reference_event_searches: list[Result[EventSearchOutput, ErrorInfo]]) : 
        pass

    def _select_event(self, active_table) : 
        """
        """
        #list of rate constant
        l_k = np.array([active_table.table.loc[i].at['k'] for i in range(len(active_table.table))])
        idx_selected_event, delta_t = rejection_free(l_k)
        return idx_selected_event, delta_t
    def _select_event_generic(self) : 
        """ 
        """
        #Find all possible event
        if self.config.control.reconstruction : 
            l_env = list(set(self.atomic_environment.atomic_environment_list))
            if l_env == ['crystal'] : 
                self._close()
            l_reference_table = [i for i in range(len(self.reference_table.table)) if self.reference_table.table.loc[i].at['event_id'] in l_env ]
        else  : # all events in reference events are possible 
            l_reference_table = [i for i in range(len(self.reference_table.table))]
        #Get constant rate of possible events
        l_k = np.array([self.reference_table.table.loc[l_reference_table[i]].at['k'] for i in range(len(l_reference_table))])
        #Apply algorithm select event : 
        idx_selected_event, delta_t = rejection_free(l_k)
        return l_reference_table[idx_selected_event], delta_t 

    def _select_central_atom_idx(self, idx_event_table) : 
        """ 
        """
        if self.config.control.reconstruction : 
            id_hash = self.reference_table.table.loc[idx_event_table].at['event_id'] 
            possible = [i for i,e in enumerate(self.atomic_environment.atomic_environment_list) if e == id_hash]
            return random.choice(possible) 
        else : 
            return self.reference_table.table.loc[idx_event_table].at['atom_index'] 

    def _apply_event(self, idx_selected_event, active_table ) : 
        """ 
        """
        new_positions = active_table.table.loc[idx_selected_event].at['final_positions'] 
        self.system.update_positions(new_positions)


    def _apply_event_generic(self, idx_atom_apply_event, idx_event_table) : 
        """ 
        """
        if self.config.control.reconstruction :
            rmat, tr, perm, dh = PointSetRegistration(self.config, self.system, self.reference_table, self.neighbors_list, idx_event_table, idx_atom_apply_event).run()
            if rmat is None or dh > self.config.psr.matching_score_thr : 
                return False 
            else :
                current_positions = self.system.positions.copy()
                #initial potential energy
                Eini = self.engine.compute_potential_energy(self.system)
                #go to saddle point 
                neighbors = self.neighbors_list.get_neighbors('rcut', idx_atom_apply_event)
                new_positions = np.zeros((len(self.reference_table.table.loc[idx_event_table].at['saddle_positions']), 3))
                for i in range(len(new_positions)) : 
                    new_positions[i] = np.matmul(rmat, self.reference_table.table.loc[idx_event_table].at['saddle_positions'][i]) + tr 
                new_positions[:] = new_positions[perm]
                self.system.update_positions(new_positions, atom_idx = neighbors)
                #saddle potential energy 
                Esad = self.engine.compute_potential_energy(self.system)
                #check if energy barrier consistent : 
                dE = Esad-Eini
                if abs(dE-self.reference_table.table.loc[idx_event_table]['energy_barrier']) < 0.5 : 

                    new_positions = np.zeros((len(self.reference_table.table.loc[idx_event_table].at['final_positions']), 3))
                    for i in range(len(new_positions)) : 
                        new_positions[i] = np.matmul(rmat, self.reference_table.table.loc[idx_event_table].at['final_positions'][i]) + tr 
                    new_positions[:] = new_positions[perm]
                    self.system.update_positions(new_positions, atom_idx = neighbors)
                    return True
                else : 
                    #back to current positions
                    self.system.update_positions(current_positions)
                    return False

        else : 
            #neigbors of central atoms : 
            neighbors = self.neighbors_list.get_neighbors('rcut', idx_atom_apply_event)
            final_positions = self.reference_table.table.loc[idx_event_table].at['final_positions'] 
            #updat positions : 
            self.system.update_positions(final_positions, atom_idx = neighbors)


    def _center_event_positions(self, event_search_output: EventSearchOutput) : 
        #Translate atoms so that the atom that moves the most is at the center of the cell at start event, prevent pbc problem with psr 
        cell = self.system.cell
        ax, ay, az = cell[0][0], cell[1][1], cell[2][2] 
        #displacement 
        move_atom_idx = event_search_output.move_atom_index        
        dx, dy, dz = ax/2 - event_search_output.min1_positions[move_atom_idx][0],  ay/2 - event_search_output.min1_positions[move_atom_idx][1], az/2 - event_search_output.min1_positions[move_atom_idx][2]
        displacement = np.array([dx, dy, dz])
        event_search_output.min1_positions = self._center_positions(event_search_output.min1_positions, displacement, cell)
        event_search_output.saddle_positions = self._center_positions(event_search_output.saddle_positions, displacement, cell)
        event_search_output.min2_positions = self._center_positions(event_search_output.min2_positions, displacement, cell)
        return event_search_output
        #return min1positions, saddlepositions, min2positions
    


    #def _center_event_positions(self, min1positions, saddlepositions, min2positions, move_atom_idx) : 
    #    #Translate atoms so that the atom that moves the most is at the center of the cell at start event, prevent pbc problem with psr 
    #    cell = self.system.cell
    #    ax, ay, az = cell[0][0], cell[1][1], cell[2][2] 
    #    dx, dy, dz = ax/2 - min1positions[move_atom_idx][0], ay/2 - min1positions[move_atom_idx][1], az/2 - min1positions[move_atom_idx][2]
    #    displacement = np.array([dx, dy, dz])
    #    min1positions = self._center_positions(min1positions, displacement, cell)
    #    saddlepositions = self._center_positions(saddlepositions, displacement, cell)
    #    min2positions = self._center_positions(min2positions, displacement, cell)
    #    return min1positions, saddlepositions, min2positions
    

    def _center_positions(self, positions, displacement, cell) : 
        positions += displacement 
        positions = ase.geometry.wrap_positions(positions=positions, cell=cell, pbc=True)
        positions[positions < 0 ] = 0
        return positions
    

    def refinement(self) : 
        #Initialize ActiveEventTable
        active_table = ActiveEventTable() 
        #Save_current_positions 
        current_positions = self.system.positions.copy()

        #Subset of reference_event_table with generic event that can be apply to the current step (ie event_id in atomic environment)
        subset_reference_event_table = self.reference_table.table[self.reference_table.table['event_id'].isin(self.atomic_environment.atomic_environment_list)] 

        #Loop on subset : 
        counts = 0
        success = 0
        counts_sym = 0 
        success_sym = 0 
        for idx, dfevent in subset_reference_event_table.iterrows() : 
            #For each dfevent Series, need to find all atoms on which we can perform the event 
            l_atoms_refine = [i for i,e in enumerate(self.atomic_environment.atomic_environment_list) if e == dfevent['event_id']]
            #for each atoms refine 
            for at_idx in l_atoms_refine : 
                #Need to go to saddle point applying PSR : 
                psr_output = PointSetRegistration(self.config, self.system, dfevent, self.neighbors_list, 0, at_idx).run()
                if psr_output.is_ok() : 
                    psr_output = psr_output.ok_value()
                    if psr_output.matching_score < self.config.psr.matching_score_thr : 
                        #Go saddle point : 
                            #Apply PSR to generic event
                        saddle_positions = geometry.transform_positions(dfevent.at['saddle_positions'], psr_output.rotation_matrix, psr_output.translation_matrix, psr_output.permutation_matrix) 
                            #Get atomic environment atoms
                        neighbors = self.neighbors_list.get_neighbors('rcut', at_idx)
                            #Move system to saddle point
                        self.system.update_positions(saddle_positions, atom_idx = neighbors)

                        #When at saddle positions refine with partn
                        refine_output = self.engine.refine_event(self.system, at_idx)
                        counts += 1
                        if refine_output.is_ok() :
                            refine_output = refine_output.ok_value()
                        #Generate dfevent series from refine event results 
                            dfactive = self._build_refined_event_series(current_positions, at_idx, refine_output.min1_positions, refine_output.min2_positions, refine_output.dE_forward, refine_output.dE_backward)
                            #Check if dE coherent 
                            if abs(dfactive.at['energy_barrier']-dfevent.at['energy_barrier']) < self.config.eventsearch.refined_energy_thr : 
                                active_table.add_event(dfactive)
                                success += 1
                            else : 
                                refine_output = ErrorInfo(type=ErrorType.REFINEMENT_INVALID_ENERGY_BARRIER, 
                                                      message = "refinement energy barrier does not match reference one", 
                                                      details = "Reference energy barrier = {}, refined one = {}, refine energy threshold = {}".format(dfevent.at['energy_barrier'], dfactive.at['energy_barrier'], self.config.eventsearch.refine_energy_threshold, 
                                                      variables = {'reference_event_index' : idx , 'atom_index' : at_idx , 'min1_positions' : refine_output.min1_positions, 'saddle_positions' : refine_output.saddle_positions, 'min2_positions' :refine_output.min2_positions })) 
                        else : 
                            refine_output = refine_output.err_value()
                            #print("refine FAILED no event found")
                        #Back to current positions :
                        self.system.update_positions(current_positions)
                        #Need to do the same for symetries : 
                        for sym_matrix, perm_matrix in zip(dfevent.at['sym_matrix'], dfevent.at['sym_perm'])  : 
                            #Displacement between current positions and saddle_positions : 
                            displacements = dfevent.at['saddle_positions']-dfevent.at['initial_positions']
                            #APPLY Symmetry to displacement 
                            new_displacements = geometry.transform_positions(displacements, sym_matrix, 0, perm_matrix)
                            #Apply displacement to generic initial positions 
                            new_saddle_positions = dfevent.at['initial_positions']+new_displacements
                            #Aplly PSR to the new saddle positions 
                            new_positions = geometry.transform_positions(new_saddle_positions, psr_output.rotation_matrix, psr_output.translation_matrix, psr_output.permutation_matrix)
                            #update system positions
                            self.system.update_positions(new_positions, atom_idx = neighbors)
                            #event refine
                            refine_output = self.engine.refine_event(self.system, at_idx)
                            counts_sym +=1
                            if refine_output.is_ok() :
                                refine_output = refine_output.ok_value()
                            #Generate dfevent series from refine event results 
                                dfactive = self._build_refined_event_series(current_positions, at_idx, refine_output.min1_positions, refine_output.min2_positions, refine_output.dE_forward, refine_output.dE_backward)
                                #Check if dE coherent 
                                if abs(dfactive.at['energy_barrier']-dfevent.at['energy_barrier']) < self.config.eventsearch.refined_energy_thr : 
                                    active_table.add_event(dfactive)
                                    success_sym +=1
                                else : 
                                    refine_output = ErrorInfo(type=ErrorType.REFINEMENT_INVALID_ENERGY_BARRIER, 
                                                          message = "refinement energy barrier does not match reference one", 
                                                          details = "Reference energy barrier = {}, refined one = {}, refine energy threshold = {}".format(dfevent.at['energy_barrier'], dfactive.at['energy_barrier'], self.config.eventsearch.refine_energy_threshold, 
                                                          variables = {'reference_event_index' : idx , 'atom_index' : at_idx , 'min1_positions' : refine_output.min1_positions, 'saddle_positions' : refine_output.saddle_positions, 'min2_positions' :refine_output.min2_positions })) 
                            else :
                                refine_output = refine_output.err_value() 
                                #print("refine FAILED no event found, SYM EVENT")
                            #Back to current positions :
                            self.system.update_positions(current_positions)
                    else : 
                        psr_output = Err(ErrorInfo(type = ErrorType.PSR_MATCHING_SCORE_ABOVE_ACCEPTANCE_THRESHOLD,
                                                   message = "PSR found a match but matching score is above acceptance threshold", 
                                                   details= "Hausdorff distance = {}, acceptance threshold = {} ".format(psr_output.matching_score, self.config.psr.matching_score)))
        #self.logger.logger.debug("{} refine attemps, {} success, {} direct and {} sym".format(counts+counts_sym, success+success_sym, success, success_sym))
        return active_table
    
    def _build_refined_event_series(self, current_positions, at_idx, min1positions, min2positions, dE_forward, dE_backwards) : 
        #Find if min1 or min2 is initial positions 
        #To deal with pbc problem and lammps slighlty over/under box positions 
        dr1_vec, _ = ase.geometry.find_mic(current_positions[at_idx] - min1positions[at_idx],cell=self.system.cell,pbc=self.system.pbc)
        dr2_vec, _ = ase.geometry.find_mic(current_positions[at_idx] - min2positions[at_idx],cell=self.system.cell,pbc=self.system.pbc)
        #compare only atom that move 
        dr1 = np.sum(np.abs(dr1_vec))
        dr2 = np.sum(np.abs(dr2_vec))
        if dr1 < dr2 : 
            final_positions = min2positions
            dE = dE_forward
        else : 
            final_positions = min1positions
            dE = dE_backwards
        #build event series 
        dfactive = pd.Series({'atom_index': at_idx, 
                              'final_positions' : final_positions,
                              'energy_barrier' : dE, 
                              'k' :compute_rate_Eyring(dE, self.config)})
        return dfactive
    
    def minimize_system(self) -> None : 
        """Minimize the system and update its positions"""
        self.loggers.info('log',':=> Minimizing the system')
        new_positions = self.engine.minimize(self.system)
        self.system.update_positions(new_positions)

    def info_atomic_environments(self, new_environments: list[str|bytes]) -> AtomicEnvironmentInfo : 
        atomic_environments_info = AtomicEnvironmentInfo(total_atomic_environments_encounter = len(self.visited_environments),
                n_current_atomic_environments = len(set(self.atomic_environment.atomic_environment_list)),
                n_new_atomic_environments = len(new_environments))  
        if self.config.control.verbosity == 2 : 
            atom_group = {}
            for index, item in enumerate(self.atomic_environment.atomic_environment_list):
                if item != 'crystal' : 
                    if item not in atom_group: 
                        atom_group[item] = []  
                    atom_group[item].append(index) 
            atomic_environments_info.atoms_grouped_by_environment = list(atom_group.values())
        return atomic_environments_info 

    def info_reference_event_searches(self, results_reference_event_searches: list[Result[EventSearchOutput, ErrorInfo]])  -> ReferenceEventSearchInfo : 
        total_event_searches = len(results_reference_event_searches)
        n_success = 0 
        n_fails = {'no_event_found' : 0, "minima_not_matching_positions" : 0}
        for res in results_reference_event_searches : 
            if res.is_ok() : 
                n_success +=1 
            else : 
                #n_fails += 1
                if res.err_value().type == ErrorType.EVENT_NOT_FOUND :
                    n_fails['no_event_found'] +=1 
                else : 
                    n_fails['minima_not_matching_positions'] +=1 
                    
        return ReferenceEventSearchInfo(total_event_searches, n_success, n_fails)
    
    def info_is_valid_events(self, results_is_valid_events: list[Result[pd.DataFrame, ErrorInfo]]) -> ReferenceValidEventsInfo : 
        n_valid_events = 0 
        invalid_events = {"dE > emax_event": 0 , 
                          "dE < emin_event": 0, 
                          "dE inverse < emin_event": 0, 
                          "Event asymmetric" : 0, 
                          "Event already in reference table" : 0}
        for res in results_is_valid_events : 
            if res.is_ok() : 
                n_valid_events +=1 
            else : 
                match res.err_value().type : 
                    case ErrorType.EVENT_ENERGY_HIGHER_THAN_THRESHOLD : 
                        invalid_events['dE > emax_event'] +=1 
                    case ErrorType.EVENT_ENERGY_LOWER_THAN_THRESHOLD : 
                        invalid_events['dE < emin_event'] +=1
                    case ErrorType.EVENT_BACKWARD_ENERGY_LOWER_THAN_THRESHOLD : 
                        invalid_events['dE inverse < emin_event'] +=1 
                    case ErrorType.EVENT_ASYMMETRIC : 
                        invalid_events['Event asymmetric'] +=1
                    case ErrorType.EVENT_NOT_NEW : 
                        invalid_events['Event already in reference table'] +=1
        return ReferenceValidEventsInfo(n_valid_events, invalid_events)
                        
    def _initialize(self) -> None: 
        """Initialize the entire simulation before starting."""
        self._initialize_loggers()
        self._initialize_system()
        self._initialize_engine() 
        self.minimize_system() 
        self._initialize_neighbors_list() 
        self._initialize_atomic_environments() 
        self._initialize_reference_table()  
        self._initialize_visited_environments()

        self.loggers.new_line('log')
        self.loggers.info('log', '===========================')
        self.loggers.info('log', '= Starting KMC simulation =')
        self.loggers.info('log', '===========================')

    def _initialize_loggers(self) -> None: 
        """Initialize the loggers and create their files."""
        self.loggers = LogKMC(LOGGING_CONFIG)
        self.loggers.title('log')
        self.loggers.write_parameters('log', self.config)
        self.loggers.output_file_header('output')

    def _initialize_system(self) -> None : 
        """Read and initialize the system from the intial configuration file."""
        self.loggers.info('log', ':=> Reading initial configuration file : {}'.format(self.config.control.initial_config))
        self.system = System.create_from_file(self.config.control.initial_config)
    
    def _initialize_engine(self) -> None : 
        """Initialize the engine based on the Config"""
        self.loggers.info('log', ':=> Initializing E/F {} Engine'.format(self.config.control.engine))
        self.engine = Engine(self.config)

    def _initialize_neighbors_list(self) -> None : 
        """Construct a new Neighbors List""" 
        self.loggers.info('log', ':=> Constructing Neighbors Lists')
        self.neighbors_list = NeighborsList(self.system, self.config.atomicenvironment.rnei, self.config.atomicenvironment.rcut)

    def _initialize_atomic_environments(self) -> None : 
        """Construct a new Atomic Environment"""
        self.loggers.info('log', ':=> Computing Atomic Environments')
        self.atomic_environment = AtomicEnvironment(self.config.atomicenvironment.style, self.neighbors_list.neighbors_list['rnei'], self.neighbors_list.neighbors_list['rcut'], self.config.atomicenvironment.neighbors_add)

    def _initialize_reference_table(self) -> None : 
        """Initialize the Reference Event Table"""
        if self.config.control.reference_table is not None : 
            self.loggers.info('log', ':=> Reading Reference table file {}'.format(self.config.control.reference_table))
        else : 
            self.loggers.info('log', ':=> Generate a empty reference table')
        self.reference_table = ReferenceEventTable(self.config)

    def _initialize_visited_environments(self) -> None : 
        """Initialize visited environment from file if specified, else initialize as {'crystal'}"""
        if self.config.control.visited_environments is not None : 
            self.loggers.info('log', ':=> Initiating visited environment from file {}'.format(self.config.control.visited_environments))
            try:
                with open(self.config.control.visited_environments, 'rb') as file:
                    loaded_set_environments = pickle.load(file)
                self.visited_environments = loaded_set_environments
            except Exception as e:
                raise Exception("Can't read visited environment file.")
        else : 
            self.visited_environments = set(['crystal'])
        if self.config.control.visited_environments and not self.config.control.reference_table : 
            self.loggers.warning('log', "Visited environments are read from file while no reference table was provided")

    def _append_snapshot_to_trajectory(self) : 
        output = self.config.control.trajectory_output
        atoms = Atoms(self.system.types, positions=self.system.positions, cell=self.system.cell, pbc=self.system.pbc)
        write(output, atoms, append=True)

    def _save(self) : 
        self.reference_table.save('reference_table.pickle')
        with open(self.config.control.visited_environments_output, 'wb') as file : 
            pickle.dump(self.visited_environments, file)

    def _close(self) : 
        self.loggers.info('log', ':=> End of simulation')
        sys.exit()