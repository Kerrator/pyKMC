
from .environments import cna, graph

class AtomicEnvironment() :

    def __init__(self, config, neighbors_list, environment_list=None) : 
        self.style = config['AtomicEnvironment']['style']
        self.neighbors_list = neighbors_list
        self.environment_list = environment_list
        if self.style == 'cna' :
            self.atomic_environment_list = self.compute_cna(neighbors_list)
        if self.style == 'graph' : 
            self.atomic_environment_list = self.compute_graph(neighbors_list, environment_list)

    def compute_cna(self, neighbors_list) : 
        return cna(neighbors_list)
    
    def compute_graph(self, neighbors_list, environment_list) : 
        return graph(neighbors_list, environment_list) 
