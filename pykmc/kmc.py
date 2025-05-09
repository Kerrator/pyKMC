from pykmc import System, Engine, Config, NeighborsList, AtomicEnvironment, Catalog, PointSetRegistration, Logger, ActiveEventTable
import random 
import numpy as np
from ase.io import write
from ase import Atoms
import ase.geometry
from .algorithms import * 
import sys
import pandas as pd
from .rate_constant import compute_rate_Eyring
from scipy.spatial import cKDTree


class KMC() : 

    def __init__(self, config) : 
        self.config = config 
        self.logger = None
        self.system = None 
        self.engine = None 
        self.neighbors_list = None 
        self.atomic_environment = None 
        self.catalog = None
        self.visited_environment = set(['crystal'])
            
    def run(self) : 
        
        ###### START ###### 
        self.initialize()

        nkmc_steps = self.config['Control']['nkmc_steps']
        time = 0.0 #in seconds
        nsearch = self.config['EventSearch']['nsearch']


        #Write initial step to file : 
        self._append_snapshot_to_trajectory()


        self.logger.logger.info('===========================')
        self.logger.logger.info('= Starting KMC simulation =')
        self.logger.logger.info('===========================')
        ####### KMC Loop ########
        self.logger.first_line_table()
        for step in range(nkmc_steps) :

            #########################################################
            #FIND NEW GENERIC EVENT AND UPDATE REFERENCE EVENT TABLE#
            #########################################################
            #Find new atomic environments that have not been visited
            new_environment = list(set(self.atomic_environment.atomic_environment_list).difference(self.visited_environment)) 

            #List of atoms(central) on which we gonna perfom an event search
            central_atom_research_list = self.central_atoms_research(new_environment, nsearch)
            #Count number of tentative to prevent empty catalog : 
            MAX_TRIES = 5
            tries = 0 

            while tries < MAX_TRIES : 
                #Fro all idx in central_atom_research_list
                fails = 0 #to count the number of event search fails
                
                for idx in central_atom_research_list : 

                    #do an event search 
                    results = self.engine.search_event(self.system, idx)

                    if results != None : 
                    #add results in catalog 
                        if self.config['Control']['reconstruction'] : #if reconstruction, need to center event to prevent pbc problem
                            results = (*self._center_event_positions(results[0], results[1], results[2], results[3]), *results[3:])
                        is_new, in_e_bounds = self.catalog.add_event(*results, self.neighbors_list.neighbors_list['rcut'], self.system.cell)

                    else : #failed 
                        fails += 1
                
                if len(self.catalog.catalog) > 0 : 
                    #idx_event_catalog, delta_t = self._select_event() 
                    break #end while loop
                else : 
                    tries += 1 
                    self.logger.logger.debug('Empty catalog after {} searches, retrying'.format(len(central_atom_research_list)))
            else : #if not breack encounter : 
                self.logger.logger.debug('Emtpy catalog after {} tries, closing simulation'.format(MAX_TRIES))
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


            #idx_event_catalog, delta_t = self._select_event() 
            ##Select central atom having same atomic environment as the event
            #idx_atom_apply_event = self._select_central_atom_idx(idx_event_catalog)
            #is_reconstruction = self._apply_event(idx_atom_apply_event, idx_event_catalog)

            #update time : 
            #if is_reconstruction : 
            #    time += delta_t
            #write config to file 
            self._append_snapshot_to_trajectory()
            #if no reconstruction, new catalog
            if not self.config['Control']['reconstruction'] : 
                self.catalog = Catalog(self.config)
            else : #update visited environments 
                l_ids = list(set(self.atomic_environment.atomic_environment_list)) 
                self.visited_environment.update(set(l_ids).difference(self.visited_environment))

                #self.logger.table_line_info_kmc(step, time, len(list(set(self.atomic_environment.atomic_environment_list))), len(self.catalog.catalog), idx_event_catalog, self.catalog.catalog.loc[idx_event_catalog].at['energy_barrier'], is_reconstruction )
                self.logger.table_line_info_kmc(step, time, len(list(set(self.atomic_environment.atomic_environment_list))), len(self.catalog.catalog),  active_table.active_events.loc[idx_selected_event].at['energy_barrier'], active_table.active_events.loc[idx_selected_event].at['k'] )
            #update neighborlist : 
            self.neighbors_list = NeighborsList(self.system, self.config) 
            #update atomic environment 
            self.atomic_environment = AtomicEnvironment(self.config, self.neighbors_list.neighbors_list['rnei'], self.neighbors_list.neighbors_list['rcut'])
        
        self.catalog.catalog.to_pickle('catalog.pickle')


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
        l_k = np.array([active_table.active_events.loc[i].at['k'] for i in range(len(active_table.active_events))])
        idx_selected_event, delta_t = rejection_free(l_k)
        return idx_selected_event, delta_t
    def _select_event_generic(self) : 
        """ 
        """
        #Find all possible event
        if self.config['Control']['reconstruction'] : 
            l_env = list(set(self.atomic_environment.atomic_environment_list))
            if l_env == ['crystal'] : 
                print('only crystal atoms')
                self._close()
            l_catalog = [i for i in range(len(self.catalog.catalog)) if self.catalog.catalog.loc[i].at['event_id'] in l_env ]
        else  : # all events in catalog are possible 
            l_catalog = [i for i in range(len(self.catalog.catalog))]
        #Get constant rate of possible events
        l_k = np.array([self.catalog.catalog.loc[l_catalog[i]].at['k'] for i in range(len(l_catalog))])
        #Apply algorithm select event : 
        idx_selected_event, delta_t = rejection_free(l_k)
        return l_catalog[idx_selected_event], delta_t 

    def _select_central_atom_idx(self, idx_event_catalog) : 
        """ 
        """
        if self.config['Control']['reconstruction'] : 
            id_hash = self.catalog.catalog.loc[idx_event_catalog].at['event_id'] 
            possible = [i for i,e in enumerate(self.atomic_environment.atomic_environment_list) if e == id_hash]
            return random.choice(possible) 
        else : 
            return self.catalog.catalog.loc[idx_event_catalog].at['atom_index'] 

    def _apply_event(self, idx_selected_event, active_table ) : 
        """ 
        """
        new_positions = active_table.active_events.loc[idx_selected_event].at['final_positions'] 
        self.system.update_positions(new_positions)


    def _apply_event_generic(self, idx_atom_apply_event, idx_event_catalog) : 
        """ 
        """
        if self.config['Control']['reconstruction'] :
            rmat, tr, perm, dh = PointSetRegistration(self.config, self.system, self.catalog, self.neighbors_list, idx_event_catalog, idx_atom_apply_event).run()
            if rmat is None or dh > self.config['PSR']['hausdorff_dist_thr'] : 
                return False 
            else :
                current_positions = self.system.positions.copy()
                #initial potential energy
                Eini = self.engine.compute_potential_energy(self.system)
                #go to saddle point 
                neighbors = self.neighbors_list.get_neighbors('rcut', idx_atom_apply_event)
                new_positions = np.zeros((len(self.catalog.catalog.loc[idx_event_catalog].at['saddle_positions']), 3))
                for i in range(len(new_positions)) : 
                    new_positions[i] = np.matmul(rmat, self.catalog.catalog.loc[idx_event_catalog].at['saddle_positions'][i]) + tr 
                new_positions[:] = new_positions[perm]
                self.system.update_positions(new_positions, atom_idx = neighbors)
                #saddle potential energy 
                Esad = self.engine.compute_potential_energy(self.system)
                #check if energy barrier consistent : 
                dE = Esad-Eini
                if abs(dE-self.catalog.catalog.loc[idx_event_catalog]['energy_barrier']) < 0.5 : 

                    new_positions = np.zeros((len(self.catalog.catalog.loc[idx_event_catalog].at['final_positions']), 3))
                    for i in range(len(new_positions)) : 
                        new_positions[i] = np.matmul(rmat, self.catalog.catalog.loc[idx_event_catalog].at['final_positions'][i]) + tr 
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
            final_positions = self.catalog.catalog.loc[idx_event_catalog].at['final_positions'] 
            #updat positions : 
            self.system.update_positions(final_positions, atom_idx = neighbors)


    def _center_event_positions(self, min1positions, saddlepositions, min2positions, move_atom_idx) : 
        #Translate atoms so that the atom that moves the most is at the center of the cell at start event, prevent pbc problem with psr 
        cell = self.system.cell
        ax, ay, az = cell[0][0], cell[1][1], cell[2][2] 
        dx, dy, dz = ax/2 - min1positions[move_atom_idx][0], ay/2 - min1positions[move_atom_idx][1], az/2 - min1positions[move_atom_idx][2]
        displacement = np.array([dx, dy, dz])
        min1positions = self._center_positions(min1positions, displacement, cell)
        saddlepositions = self._center_positions(saddlepositions, displacement, cell)
        min2positions = self._center_positions(min2positions, displacement, cell)
        return min1positions, saddlepositions, min2positions
    

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
        subset_reference_event_table = self.catalog.catalog[self.catalog.catalog['event_id'].isin(self.atomic_environment.atomic_environment_list)] 

        #Loop on subset : 
        for idx, dfevent in subset_reference_event_table.iterrows() : 
            #For each dfevent Series, need to find all atoms on which we can perform the event 
            l_atoms_refine = [i for i,e in enumerate(self.atomic_environment.atomic_environment_list) if e == dfevent['event_id']]
            #for each atoms refine 
            for at_idx in l_atoms_refine : 
                #Need to go to saddle point applying PSR : 
                rmat, tr, perm, dh = PointSetRegistration(self.config, self.system, dfevent, self.neighbors_list, 0, at_idx).run()
                if rmat is None or dh > self.config['PSR']['hausdorff_dist_thr'] : 
                    print("PSR FAIL")
                else : 
                    #Go saddle point : 
                        #Apply PSR to generic event
                    saddle_positions = self._transform_positions(dfevent.at['saddle_positions'], rmat, tr, perm) 
                        #Get atomic environment atoms
                    neighbors = self.neighbors_list.get_neighbors('rcut', at_idx)
                        #Move system to saddle point
                    self.system.update_positions(saddle_positions, atom_idx = neighbors)

                    #When at saddle positions refine with partn
                    results = self.engine.refine_event(self.system, at_idx)

                    if results is not None : 
                    #Generate dfevent series from refine event results 
                        dfactive = self._build_refined_event_series(current_positions, at_idx, results[0], results[2], results[3], results[4])
                        #Check if dE coherent 
                        if abs(dfactive.at['energy_barrier']-dfevent.at['energy_barrier']) < 0.1 : 
                            active_table.add_event(dfactive)
                        else : 
                            print("ERROR: delta energy refinement, generic event energy = {}, refine event energy = {} ".format(dfevent.at['energy_barrier'], dfactive.at['energy_barrier']))
                            atoms = Atoms(np.array(self.system.types)[neighbors], positions=dfevent.at['initial_positions'])
                            write('refinefail.xyz', atoms, append=True)
                            atoms = Atoms(np.array(self.system.types)[neighbors], positions=dfevent.at['saddle_positions'])
                            write('refinefail.xyz', atoms, append=True)
                            atoms = Atoms(np.array(self.system.types)[neighbors], positions=dfevent.at['final_positions'])
                            write('refinefail.xyz', atoms, append=True)
                            atoms = Atoms(np.array(self.system.types)[neighbors], positions=current_positions[neighbors])
                            write('refinefail.xyz', atoms, append=True)
                            atoms = Atoms(np.array(self.system.types)[neighbors], positions=self.system.positions[neighbors])
                            write('refinefail.xyz', atoms, append=True)
                            atoms = Atoms(np.array(self.system.types)[neighbors], positions=dfactive.at['final_positions'][neighbors])
                            write('refinefail.xyz', atoms, append=True)
                    else : 
                        print("refine FAILED")
                    #Back to current positions :
                    self.system.update_positions(current_positions) 

                    #Need to do the same for symetries : 
                    for sym_matrix, perm_matrix in zip(dfevent.at['sym_matrix'], dfevent.at['sym_perm'])  : 
                        #Displacement between current positions and saddle_positions : 
                        displacements = saddle_positions-current_positions[neighbors] 
                        apply_displacements = self._transform_positions(displacements, sym_matrix, 0, perm_matrix)
                        new_positions = current_positions[neighbors]+apply_displacements
                        self.system.update_positions(new_positions, atom_idx = neighbors)
                        atoms = Atoms(self.system.types, positions=self.system.positions, cell=self.system.cell, pbc=self.system.pbc)
                        write('test.xyz', atoms, append=True)
                        self.system.update_positions(current_positions)




        return active_table
    
    def _transform_positions(self, positions, transformation_matrix, translation_matrix, permutation_matrix) : 
        transform_positions = positions @ transformation_matrix.T + translation_matrix 
        return transform_positions[permutation_matrix]
    
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

    def initialize(self) : 
        self.logger = Logger(self.config) 
        self.logger.title()
        self.logger.write_parameter()
        self.logger.logger.info('=> Reading configuration file : {}'.format(self.config['Control']['config_file']))
        self.system = System.create_from_file(self.config['Control']['config_file'])
        self.logger.logger.info('=> Initializing E/F {} Engine'.format(self.config['Control']['engine']))
        self.engine = Engine(self.config)
        #minimize 
        self.logger.logger.info('=> Minimizing the system')
        new_positions = self.engine.minimize(self.system)
        self.system.update_positions(new_positions)
        self.neighbors_list = NeighborsList(self.system, self.config) 
        self.atomic_environment = AtomicEnvironment(self.config, self.neighbors_list.neighbors_list['rnei'], self.neighbors_list.neighbors_list['rcut'])
        if self.config['Control']['catalog'] is not None : 
            self.logger.logger.info('=> Reading catalog file {}'.format(self.config['Control']['catalog']))
        else : 
            self.logger.logger.info('=> Initilizing Catalog')
        self.catalog = Catalog(self.config)
        self.logger.new_line()

    def _append_snapshot_to_trajectory(self) : 
        output = self.config['Control']['output_file']
        atoms = Atoms(self.system.types, positions=self.system.positions, cell=self.system.cell, pbc=self.system.pbc)
        write(output, atoms, append=True)

    def _close(self) : 
        self.logger.logger.info('End of simulation')
        sys.exit()