from pykmc import System, Engine, NeighborsList, AtomicEnvironment, PointSetRegistration, LogKMC, LOGGING_CONFIG, ActiveEventTable, ReferenceEventTable
import random 
from .result import EventSearchOutput, KMCLoopInfo, Err, ErrorInfo, ErrorType
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


class KMC() : 

    def __init__(self, config) : 
        self.config = config 
        self.loggers = None
        self.system = None 
        self.engine = None 
        self.neighbors_list = None 
        self.atomic_environment = None 
        self.reference_table = None
        self.visited_environment = set(['crystal'])
            
    def run(self) : 

        ################################################################# 
        #                  INITIALIZE ATTRIBUTES                        #
        ################################################################# 
        self._initialize()
        #Write initial step to file : 
        self._append_snapshot_to_trajectory()

        ################################################################# 
        #                    LOOP KMC PARAMETERS                        #
        ################################################################# 
        nkmc_steps = self.config.control.n_steps
        time = 0.0 #in seconds
        nsearch = self.config.eventsearch.nsearch

        ################################################################# 
        #                         KMC LOOP                              #
        ################################################################# 
        #self.logger.first_line_table() #write log head table

        for step in range(nkmc_steps) :
            #########################################################
            #FIND NEW GENERIC EVENT AND UPDATE REFERENCE EVENT TABLE#
            #########################################################
                #=> Find new atomic environments that have not been visited
            new_environment = list(set(self.atomic_environment.atomic_environment_list).difference(self.visited_environment)) 

                #=>List of atoms(central) on which we gonna perfom an event search
            central_atom_research_list = self.central_atoms_research(new_environment, nsearch)

            MAX_TRIES = 5 #Number of tentatives to prevent emtpy reference table : 
            tries = 0 
            while tries < MAX_TRIES : 
                fails = 0 #to count the number of event search fails

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

                    else : #failed 
                        fails += 1

                #=> if reference event table is not emppty break while loop 
                if len(self.reference_table.table) > 0 : 
                    break #end while loop
                else : 
                    tries += 1 
                    #self.logger.logger.debug('Empty reference table after {} searches, retrying'.format(len(central_atom_research_list)))
            else : #if not breack encounter : 
                #self.logger.logger.debug('Emtpy reference table after {} tries, closing simulation'.format(MAX_TRIES))
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
                self.visited_environment.update(set(l_ids).difference(self.visited_environment))

                #self.logger.table_line_info_kmc(step, time, len(list(set(self.atomic_environment.atomic_environment_list))), len(self.catalog.catalog), idx_event_catalog, self.catalog.catalog.loc[idx_event_catalog].at['energy_barrier'], is_reconstruction )
                #self.logger.table_line_info_kmc(step, time, len(list(set(self.atomic_environment.atomic_environment_list))), len(self.reference_table.table),  active_table.table.loc[idx_selected_event].at['energy_barrier'], active_table.table.loc[idx_selected_event].at['k'] )
            #update neighborlist : 
            self.neighbors_list = NeighborsList(self.system, self.config.atomicenvironment.rnei, self.config.atomicenvironment.rcut) 
            #update atomic environment 
            self.atomic_environment = AtomicEnvironment(self.config, self.neighbors_list.neighbors_list['rnei'], self.neighbors_list.neighbors_list['rcut'])

            if set(list(self.atomic_environment.atomic_environment_list)) == {"crystal"} : 
                #self.logger.logger.info(':=> Only atoms with cristalline environment')
                self._close()

            #Loop Informations : 
            kmc_loop_info = KMCLoopInfo(step = step, 
                                        time = time, 
                                        nb_visited_environments = len(self.visited_environment), 
                                        nb_current_atomic_environments = len(set(self.atomic_environment.atomic_environment_list)), 
                                        size_reference_event_table= len(self.reference_table.table)) 
            kmc_loop_info.print_informations()

            self._save()
        


    def central_atoms_research(self, new_environment, nsearch) : 
        """ 
        return list of central atom having new_environment * nsearch
        """
        central_atom_research_list = []
        #for each atomic environment hash in new_environment 
        for env in new_environment :
            #find all index have that hash
            tmp1 = [i for i,e in enumerate(self.atomic_environment.atomic_environment_list) if e == env] 
            #Randomly choose nsearch atoms that have that environment 
            tmp2 = [random.choice(tmp1) for i in range(nsearch)]
            central_atom_research_list += tmp2
        return central_atom_research_list

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

    def _initialize(self) : 
        self._initialize_loggers()

        self.loggers.info('log', ':=> Reading initial configuration file : {}'.format(self.config.control.initial_config))
        self.system = System.create_from_file(self.config.control.initial_config)
        
        self.loggers.info('log', ':=> Initializing E/F {} Engine'.format(self.config.control.engine))
        self.engine = Engine(self.config)
        
        self.loggers.info('log',':=> Minimizing the system')
        new_positions = self.engine.minimize(self.system)
        self.system.update_positions(new_positions)

        self.loggers.info('log', ':=> Constructing Neighbors Lists')
        self.neighbors_list = NeighborsList(self.system, self.config.atomicenvironment.rnei, self.config.atomicenvironment.rcut) 

        self.loggers.info('log', ':=> Computing Atomic Environments')
        self.atomic_environment = AtomicEnvironment(self.config, self.neighbors_list.neighbors_list['rnei'], self.neighbors_list.neighbors_list['rcut'])

        #if self.config['Control']['reference_table'] is not None : 
            #self.logger.logger.info('=> Reading Reference table file {}'.format(self.config['Control']['reference_table']))
            #pass
        #else : 
            #self.logger.logger.info('=> Initilizing Reference Table')
        self.loggers.info('log', ':=> Generate a empty reference table')
        self.reference_table = ReferenceEventTable(self.config)

        self.loggers.new_line('log')
        self.loggers.info('log', '===========================')
        self.loggers.info('log', '= Starting KMC simulation =')
        self.loggers.info('log', '===========================')

    def _initialize_loggers(self) : 
        self.loggers = LogKMC(LOGGING_CONFIG)
        self.loggers.title('log')
        self.loggers.write_parameters('log', self.config)
        self.loggers.output_file_header('output')


    def _append_snapshot_to_trajectory(self) : 
        output = self.config.control.trajectory_output
        atoms = Atoms(self.system.types, positions=self.system.positions, cell=self.system.cell, pbc=self.system.pbc)
        write(output, atoms, append=True)

    def _save(self) : 
        self.reference_table.save('reference_table.pickle')
        with open(self.config.control.visited_environments_output, 'wb') as file : 
            pickle.dump(self.visited_environment, file)

    def _close(self) : 
        self.loggers.info('log', ':=> End of simulation')
        sys.exit()