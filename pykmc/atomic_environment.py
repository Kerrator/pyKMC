import numpy as np
from .environments import cna, graph

#TODO modif in config radd_cna ==> now it s number of neighbors
class AtomicEnvironment() :

    def __init__(self, config, neighbors_list, environment_list=None) : 
        self.config = config
        self.style = config['AtomicEnvironment']['style']
        self.neighbors_list = neighbors_list
        self.environment_list = environment_list
        match self.config['AtomicEnvironment']['style'] :
            case 'cna' :
                self.atomic_environment_list = self.compute_cna(neighbors_list)
            case  'graph' : 
                self.atomic_environment_list = self.compute_graph(neighbors_list, environment_list)
            case 'cna/graph' : 
                self.atomic_environment_list = self.compute_cnagraph(neighbors_list, environment_list)
            case _ : 
                raise Exception('Atomic environment style unknown')

    def compute_cna(self, neighbors_list) : 
        return cna(neighbors_list)
    
    def compute_graph(self, neighbors_list, environment_list) : 
        return graph(neighbors_list, environment_list) 
    
    def compute_cnagraph(self, neighbors_list, environment_list) : 
        list_hash = cna(neighbors_list) 
        non_crystal_idx = np.where(np.array(list_hash)=='noncrystal')[0].astype(int).tolist()
        #If radd_cna != None add neighbors of non crystal from cna
        n_neighbors = self.config['AtomicEnvironment']['radd_cna']
        if n_neighbors != None :
            tmp = []
            for i in range(n_neighbors) : 
                for idx in non_crystal_idx : 
                    tmp += neighbors_list[idx]
            non_crystal_idx += tmp
            non_crystal_idx = list(set(non_crystal_idx))
        list_graphs_hash = graph(neighbors_list, environment_list, non_crystal_idx)
        for i,idx in enumerate(non_crystal_idx) : 
            list_hash[idx] = list_graphs_hash[i]
        return list_hash