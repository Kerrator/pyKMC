import numpy as np
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
        if self.style == 'cna/graph' : 
            self.atomic_environment_list = self.compute_cnagraph(neighbors_list, environment_list)

    def compute_cna(self, neighbors_list) : 
        return cna(neighbors_list)
    
    def compute_graph(self, neighbors_list, environment_list) : 
        return graph(neighbors_list, environment_list) 
    
    def compute_cnagraph(self, neighbors_list, environment_list) : 
        list_hash = cna(neighbors_list) 
        non_crystal_idx = np.where(np.array(list_hash)=='noncrystal')[0].astype(int).tolist()
        list_graphs_hash = graph(neighbors_list, environment_list, non_crystal_idx)
        for i,idx in enumerate(non_crystal_idx) : 
            list_hash[idx] = list_graphs_hash[i]
        return list_hash