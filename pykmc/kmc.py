from pykmc import System, Engine, Config, NeighborsList, AtomicEnvironment, Catalog

class KMC() : 

    def __init__(self, config) : 
        self.config = config 
        self.system = None 
        self.engine = None 
        self.neighbors_list = None 
        self.atomic_environment = None 
        self.catalog = None

            
    def run(self) : 
        
        ###### START ###### 
        print('running')
