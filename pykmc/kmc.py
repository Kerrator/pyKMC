from pykmc import System, Engine, Config, NeighborsList, AtomicEnvironment, Catalog
import random 
import numpy as np
from ase.io import write
from ase import Atoms
import ase.geometry
from .algorithms import * 
import sys


class KMC() : 

    def __init__(self, config) : 
        self.config = config 
        self.system = None 
        self.engine = None 
        self.neighbors_list = None 
        self.atomic_environment = None 
        self.catalog = None
        self.visited_environment = set(['crystal'])
            
    def run(self) : 
        
        ###### START ###### 
        self._initialize()
        nkmc_steps = self.config['Control']['nkmc_steps']
        time = 0
        nsearch = self.config['EventSearch']['nsearch']

        #Write initial step to file : 
        self._append_snapshot_to_trajectory()
        ####### KMC Loop ########
        for step in range(nkmc_steps) :
            #Find new atomic environments that have not been visited
            new_environment = list(set(self.atomic_environment.atomic_environment_list).difference(self.visited_environment)) 

            #If only atoms with cristalline environment, close kmc simulation
            if len(new_environment) == 0 : 
                print('only cristalline atoms') 
                self._close()

            #List of atoms(central) on which we gonna perfom an event search
            central_atom_research_list = self._central_atoms_research(new_environment, nsearch)
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
                    idx_event_catalog, delta_t = self._select_event() 
                    break #end while loop
                else : 
                    tries += 1 
                    print('retry event searches, empty catalog')
            else : #if not breack encounter : 
                print('emtpy catalog avec {} tries, closing simulation'.format(MAX_TRIES))
                self._close()

            #Select central atom having same atomic environment as the event
            idx_atom_apply_event = self._select_central_atom_idx(idx_event_catalog)
            self._apply_event(idx_atom_apply_event, idx_event_catalog)

            #update time : 
            time += delta_t
            #write config to file 
            self._append_snapshot_to_trajectory()
            #if no reconstruction, new catalog
            if not self.config['Control']['reconstruction'] : 
                self.catalog = Catalog(self.config)
            else : #update visited environments 
                pass

            #update neighborlist : 
            self.neighbors_list = NeighborsList(self.system, self.config) 
            #update atomic environment 
            self.atomic_environment = AtomicEnvironment(self.config, self.neighbors_list.neighbors_list['rnei'], self.neighbors_list.neighbors_list['rcut'])


    def _central_atoms_research(self, new_environment, nsearch) : 
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
    
    def _select_event(self) : 
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

    def _apply_event(self, idx_atom_apply_event, idx_event_catalog) : 
        """ 
        """
        if self.config['Control']['reconstruction'] :
            #psr try 
            #reconstruction 
                #energy 
            pass 
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

    def _initialize(self) : 
        self.system = System.create_from_file(self.config['Control']['config_file'])
        self.engine = Engine(self.config)
        #minimize 
        new_positions = self.engine.minimize(self.system)
        self.system.update_positions(new_positions)
        self.neighbors_list = NeighborsList(self.system, self.config) 
        self.atomic_environment = AtomicEnvironment(self.config, self.neighbors_list.neighbors_list['rnei'], self.neighbors_list.neighbors_list['rcut'])
        self.catalog = Catalog(self.config)

    def _append_snapshot_to_trajectory(self) : 
        output = self.config['Control']['output_file']
        atoms = Atoms(self.system.types, positions=self.system.positions, cell=self.system.cell, pbc=self.system.pbc)
        write(output, atoms, append=True)

    def _close(self) : 
        sys.exit()