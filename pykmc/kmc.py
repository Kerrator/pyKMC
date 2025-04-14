from pykmc import System, Engine, Config, NeighborsList, AtomicEnvironment, Catalog
import random 
import numpy as np
from ase.io import write
from ase import Atoms


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
        for step in range(1) :
            #Find new atomic environments that have not been visited
            new_environment = list(set(self.atomic_environment.atomic_environment_list).difference(self.visited_environment)) 

            #List of atoms(central) on which we gonna perfom an event search
            central_atom_research_list = self.central_atoms_research(new_environment, nsearch)

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
        
            self._append_snapshot_to_trajectory()

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