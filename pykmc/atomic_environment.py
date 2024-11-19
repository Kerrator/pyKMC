from .utilities import modify_lammps_data_2D
from ase.io.lammpsdata import write_lammps_data
import numpy as np
from mpi4py import MPI
from lammps import lammps
from executorlib import Executor
import pynauty
from ase.neighborlist import NeighborList
from itertools import chain
from ase import Atoms
from profiling_decorator import profile
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist




class AtomicEnvironment() : 

    def __init__(self, atoms, atomenv_style, atomenv_params, potential=None, dimension=3, nprocs=1) : 
        #Initialization of class parameters 
        self.atoms = atoms
        self.atomenv_style = atomenv_style
        self.atomenv_params = atomenv_params
        self.potential = potential
        self.dimension = dimension
        self.nprocs = nprocs

        self.list_env = None
        self.dict_env = None
    @profile
    def run(self): 
        """
        Run similar atomic environment search based on topology_style
        """
        #TODO voir comment recuperer l'erreur de la fonction appelée avec exe.submit(), c'est un enfer a debugger sinon
        with Executor(max_cores=self.nprocs, cores_per_worker=self.nprocs) as exe : 
            match self.atomenv_style : 
                case "cna":
                    fs = exe.submit(self.cna)
                    self.list_env = fs.result() 
                #case "hausdorff_dist" : 
                #    self.hausdorff_dist()
                case "graph_nauty" : 
                    fs = exe.submit(self.graph_nauty)
                    if self.nprocs == 1 : 
                        self.list_env = fs.result()
                    else : 
                        self.list_env = list(chain(*fs.result()))
                case _:
                    raise Exception("Atomic environment style not known")
        #To dict : 
        #List of different atomic environment : 
        diff_env = set(self.list_env)
        diff_env = list(diff_env) 

        ##List of dictionnaries of different Topo ID and 
        self.dict_env = []
        for ID in diff_env : 
            #index with same ID : 
            indexsame = [i for i,e in enumerate(self.list_env) if e == ID]
            tmp = {"ID" : ID, 
                   "atom index" : indexsame}
            self.dict_env.append(tmp)


    def write_to_file(self) : 
        """
        write similar atomic environment to file as a list of dict using yaml.
        """

    def cna(self) : 
        """ 
        compute CNA using lammps. 
        return a list of ID : "crist" if the atom has a cristalline environement (ie CNA = 1,2,3 or 4) and "notcrist" if not (ie CNA=5)  
        """

        #for MPI : 
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()


        #Write lammps data file : 
        lammps_data_file = 'initial_config_minimization.lmp'
        if rank == 0 :
            write_lammps_data(lammps_data_file, self.atoms, masses=True)
            if self.dimension == 2 : 
                modify_lammps_data_2D(lammps_data_file)

        #lammps: 
        lmp = lammps(comm=comm) 
        lmp.command('units metal')
        lmp.command('atom_style atomic')
        lmp.command('dimension 3') 
        lmp.command('boundary p p p')
        lmp.command('atom_modify sort 0 1')
        lmp.command('read_data {}'.format(lammps_data_file))
        for key, val in self.potential.items() : 
            lmp.command('{} {}'.format(key, val))
        lmp.command('compute c1 all cna/atom {}'.format(self.atomenv_params['rnei']))
        lmp.command('run 0')
        #Extract cna
        cna_array = lmp.numpy.extract_compute("c1", 1,1)

        #Lammps does not sort by atom index so we extract them too
        id = lmp.numpy.extract_atom("id")
        id = id-1 #Lammps index start at 1

        lmp.close()
        #Gather values
        result = np.column_stack((id, cna_array)) 
        global_result = comm.gather(result, root=0)
        if rank == 0 : 
            #flaten arrays
            global_result = np.concatenate(global_result)
            #sort by atom index 
            global_result = global_result[global_result[:,0].argsort()]
            list_topo = [] 
            for element in global_result[:,1] : 
                if int(element) == 5 : 
                    list_topo.append('notcrist') 
                else : 
                    list_topo.append('crist')
        else : 
            list_topo = None
        list_topo = comm.bcast(list_topo, root=0)
        return list_topo
    def graph_nauty(self) :
        """
        Compute pynauty certificate based on graph canonical form
        need rnei 
        need rcut
        """ 
        #TODO : check if rnei and rcut in atomenv_params
        rnei = self.atomenv_params['rnei']
        rcut = self.atomenv_params['rcut']

        #Setup Parallisation : 
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()
            #Split index atoms in approximatively even number sublist
        split = np.array_split(range(self.atoms.get_global_number_of_atoms()), nprocs)
        local_index = split[rank] 

        #Create graphs 
        #list_g = make_graph1(self.atoms, local_index, rnei, rcut)
        #list_g = make_graph2(self.atoms, local_index, rnei, rcut)
        #list_g = make_graph3(self.atoms, local_index, rnei, rcut)
        list_g = make_graph4(self.atoms, local_index, rnei, rcut)
        list_topo = [] 
        for g in list_g : 
            list_topo.append(pynauty.certificate(g))######
        return list_topo
    
    #def hausdorff_dist(self) : 
    #    """
    #    Compute compute Hausdorff distance between structures
    #    using the CShDA algorithm and IRA
    #    """
    #    #For the moment, we select the first atom and its environment and compute the Hausdorff distance with all other atoms environments 
    #    #if the distance is lower than the threshold, atoms are considereted to have the same environment and we attribut them an ID
    #    #We repeat until all atoms have an ID 

    #    list_index = [i for i in range(self.atoms.get_global_number_of_atoms())] 

    #    res = []
    #    for ind in list_index : 
    #     
def make_graph1(atoms, list_id, rnei, rcut) : 
    """
    Create graph for all atoms with index in the list_id
    Using ASE get_distances
    """
    #index all atoms to compute distances : 
    ind = np.linspace(0, atoms.get_global_number_of_atoms()-1, atoms.get_global_number_of_atoms()).astype(int)
    #liste graphe
    list_g = []
    for k in list_id : 
        dist = atoms.get_distances(k, ind, mic=True)

        N = len(np.where(dist<rcut)[0])
        #index in rcut sphere
        indin = np.where(dist<rcut)[0]
        g = pynauty.Graph(N)

        #create graphe connextion
        for i in range(N) : 
            tmp = np.where(atoms.get_distances(indin[i], indin, mic=True) < rnei)[0]
            tmp = tmp.tolist()
            tmp.remove(i)
            g.connect_vertex(i, tmp)
        list_g.append(g)
    return list_g

#@profile 
def make_graph2(atoms, list_id, rnei, rcut) : 
    """ 
    Make graph using neighborlist ASE 
    """
    list_g = [] 
    #Create NeighborList for all atoms : 
    cutoffs = atoms.get_global_number_of_atoms()*[rcut/2] 
    nlall = NeighborList(cutoffs, self_interaction=True, bothways=True)
    nlall.update(atoms) 
    #loop over all atoms in list_id : 
    for k in list_id : 
        #create subatoms (otherwise conflict with the way pynauty constructs graph)
        subatomsnei = nlall.get_neighbors(k)[0].tolist()
        subatoms = Atoms(positions=atoms.get_positions()[subatomsnei], cell=atoms.get_cell(), pbc=True)
        #Create Neighborlist for subatoms 
        cut = subatoms.get_global_number_of_atoms()*[rnei/2]
        nl = NeighborList(cut, self_interaction=False, bothways=True)
        nl.update(subatoms)        

        g = pynauty.Graph(len(subatomsnei))

        #Loop over all neighbor 
        for i in range(len(subatomsnei)):
            g.connect_vertex(i, nl.get_neighbors(i)[0].tolist())
        list_g.append(g)

    return list_g
def make_graph3(atoms, list_id, rnei, rcut) : 
    """
    Test using scipy cKDTree, it means that we need do deal with pbc by hand 
    """
    list_g = [] 
    positions = atoms.get_positions() 
    cell = atoms.get_cell()
    #get positions and extend to take into account the PBC 
    shifts = np.array([[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)])
    replicated_positions = np.vstack([positions + np.dot(shift, cell) for shift in shifts])
    # Construire le KDTree avec les positions répliquées
    tree = cKDTree(replicated_positions)
    
    # Construire un graphe pour chaque atome
    for atom_idx in list_id:
        # Trouver les indices des voisins dans le rayon rcut
        neighbors_idx = tree.query_ball_point(positions[atom_idx], rcut)

        # Renuméroter les indices localement (0 à number_of_vertices-1)
        neighbors_global = list(neighbors_idx)  # Inclut les atomes répliqués #????
        local_to_global = {local_idx: global_idx for local_idx, global_idx in enumerate(neighbors_global)}
        global_to_local = {v: k for k, v in local_to_global.items()}


        #CDIST
        # Utiliser cdist pour calculer toutes les distances entre les voisins
        #dist_matrix = cdist(replicated_positions[neighbors_global], replicated_positions[neighbors_global])

        ## Construire le dictionnaire d'adjacence avec indices locaux
        #adjacency_dict = {global_to_local[i]: [] for i in neighbors_global}

        ## Associer correctement les indices dans dist_matrix
        #for i in range(len(neighbors_global)):
        #    for j in range(len(neighbors_global)):
        #        if i != j:  # Pas d'auto-boucle
        #            if dist_matrix[i, j] <= rnei:
        #                adjacency_dict[i].append(j)
        #END CDIST 

        #Local Tree

        # Construire le dictionnaire d'adjacence avec indices locaux
        adjacency_dict = {local_idx: [] for local_idx in local_to_global.keys()}

        # Trouver les voisins dans le rayon rnei en utilisant le deuxième KDTree
        for i, ind in enumerate(neighbors_global):
            # Recherche des voisins dans le rayon rnei
            neighbors_local = tree.query_ball_point(replicated_positions[ind], rnei)
            for neighbor in neighbors_local:
                if neighbor != ind and neighbor in neighbors_global:  # Ne pas ajouter soi-même comme voisin
                    adjacency_dict[i].append(global_to_local[neighbor])  # Utilisation des indices locaux
        #END Local Tree

        
        graph = pynauty.Graph(
            number_of_vertices=len(neighbors_idx),
            adjacency_dict=adjacency_dict,
            directed=False
        )
        
        list_g.append(graph)
    return list_g

def make_graph4(atoms, list_id, rnei, rcut) : 
    """
    Test using scipy cKDTree, with boxsize 
    """
    #TODO could try to use cdist insteed of the neighbor_local tree search (should be more efficiant for few atoms) but 
    #need to duplicated local environement
    list_g = [] 
    positions = atoms.get_positions() 
    cell = atoms.get_cell()

    # Construire le KDTree avec les positions répliquées
    tree = cKDTree(positions, boxsize=np.diag(cell))
    
    # Construire un graphe pour chaque atome
    for atom_idx in list_id:
        # Trouver les indices des voisins dans le rayon rcut
        atominenv_idx = tree.query_ball_point(positions[atom_idx], rcut)
        atominenv_idx = list(atominenv_idx)

        # Renuméroter les indices localement (0 à number_of_vertices-1)
        local_to_global = {local_idx: global_idx for local_idx, global_idx in enumerate(atominenv_idx)}
        global_to_local = {v: k for k, v in local_to_global.items()}
        
        # Construire le dictionnaire d'adjacence avec indices locaux
        adjacency_dict = {local_idx: [] for local_idx in local_to_global.keys()}


        # Trouver les voisins dans le rayon rnei en utilisant le deuxième KDTree
        for i, ind in enumerate(atominenv_idx):
            # Recherche des voisins dans le rayon rnei
            neighbors_local = tree.query_ball_point(positions[ind], rnei)
            for neighbor in neighbors_local:
                if neighbor != ind and neighbor in atominenv_idx:  # Ne pas ajouter soi-même comme voisin
                    adjacency_dict[i].append(global_to_local[neighbor])  # Utilisation des indices locaux

        
        graph = pynauty.Graph(
            number_of_vertices=len(atominenv_idx),
            adjacency_dict=adjacency_dict,
            directed=False
        )
        
        list_g.append(graph)
    return list_g


        



