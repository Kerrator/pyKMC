
from .environments import cna

class AtomicEnvironment() :

    def __init__(self, config, neighbors_list) : 
        self.style = config['AtomicEnvironment']['style']
        self.neighbors_list = neighbors_list
        if self.style == 'cna' :
            self.atomic_environment_list = self.compute_cna(neighbors_list)

    def compute_cna(self, neighbors_list) : 
        test = cna(neighbors_list)
        print(test)
        return cna(neighbors_list)
