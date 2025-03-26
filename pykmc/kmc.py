from pykmc import System, Engine, Config, NeighborsList, AtomicEnvironment, Catalog

class KMC() : 

    def __init__(self, config) : 
        self.config = config 
        self.system = None 
        self.engine = None 
        self.neighbors_list = None 
        self.atomic_environment = None 
        self.catalog = None
        self.time = None

            
    def run(self) : 
        
        ###### START ###### 
        self._initialize()
        nkmc_steps = self.config['Control']['nkmc_steps']
        self.time = 0

        ####### KMC Loop ####
        for step in range(nkmc_steps) :
            if step != 0 : 
                print('update nl et ae')
            


    def _initialize(self) : 
        self.system = System.create_from_file(self.config['Control']['config_file'])
        engine = Engine(self.config)
        #minimize 
        new_positions = engine.minimize(self.system)
        self.system.update_positions(new_positions)
        self.neighbors_list = NeighborsList(self.system, self.config) 
        self.atomic_environment = AtomicEnvironment(self.config, self.neighbors_list.neighbors_list['rnei'], self.neighbors_list.neighbors_list['rcut'])
        self.catalog = Catalog(self.config)