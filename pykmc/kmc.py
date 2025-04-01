from pykmc import System, Engine, Config, NeighborsList, AtomicEnvironment, Catalog
import random 


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

        ####### KMC Loop ####
        for step in range(1) :
            #Find new atomic environments that have not been visited
            new_environment = list(set(self.atomic_environment.atomic_environment_list).difference(self.visited_environment)) 
            #List of atoms(central) on which we gonna perfom an event search
            central_atom_research_list = self.central_atoms_research(new_environment, nsearch)
            #results = self.engine.search_event(self.system, a)
            #print(results[-1]) 

    def central_atoms_research(self, new_environment, nsearch) : 
        """ 
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