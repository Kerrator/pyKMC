from scipy.spatial import cKDTree


class NeighborsList : 

    def __init__(self, system, config) : 
        self.system = system 
        self.rnei = config['AtomicEnvironment']['rnei']
        self.rcut = config['AtomicEnvironment']['rcut']
        self.neighbors_list = {'rnei' : [], 'rcut': []}
        self._build_neighbors_list()

    def _build_neighbors_list(self) : 
        """ 
        """
        positions = self.system.positions
        positions[positions<0] = 0
        box = [self.system.cell[0][0], self.system.cell[1][1], self.system.cell[2][2]]
        tree = cKDTree(positions, boxsize=box)

        
        for i in range(len(positions)) : 
            neighbors = tree.query_ball_point(positions[i], self.rnei)
            neighbors.remove(i)
            self.neighbors_list['rnei'].append(neighbors)
            neighbors = tree.query_ball_point(positions[i], self.rcut)
            neighbors.remove(i)
            self.neighbors_list['rcut'].append(neighbors)

    def get_neighbors(self, rcut, idx) : 
        return self.neighbors_list[rcut][idx]
    
    def update_neighbors(self, list_atoms) : 
        pass
