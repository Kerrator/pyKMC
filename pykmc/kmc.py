from pykmc import System, Engine, Config, NeighborsList, AtomicEnvironment, Catalog
import random 
import numpy as np
from ase.io import write
from ase import Atoms
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
                    results = self.engine.search_event(self.system, central_atom_research_list[-1])

                    if results != None : 
                    #add results in catalog 
                        is_new = self.catalog.add_event(*results, self.neighbors_list.neighbors_list['rcut'])
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
            pass 
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
            pass 
        else : 
            return self.catalog.catalog.loc[idx_event_catalog].at['atom_index'] 

    def _apply_event(self, idx_atom_apply_event, idx_event_catalog) : 
        """ 
        """
        if self.config['Control']['reconstruction'] : 
            pass 
        else : 
            #neigbors of central atoms : 
            neighbors = self.neighbors_list.neighbors_list['rcut'][idx_atom_apply_event]
            final_positions = self.catalog.catalog.loc[idx_event_catalog].at['final_positions'] 
            #updat positions : 
            self.system.update_positions(final_positions, atom_idx = neighbors)

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