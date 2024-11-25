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

    def __init__(self, system, atomenv_style, atomenv_params, dimension=3, nprocs=1) : 
        self.system = system
        self.atomenv_style = atomenv_style
        self.atomenv_params = atomenv_params
        self.dimension = dimension
        self.nprocs = nprocs

    def run(self): 
        """
        Run similar atomic environment search based on topology_style
        """
        #TODO voir comment recuperer l'erreur de la fonction appelée avec exe.submit(), c'est un enfer a debugger sinon
        with Executor(max_cores=self.nprocs, cores_per_worker=self.nprocs) as exe : 
            match self.atomenv_style : 
                case "cna":
                    fs = exe.submit(self.cna)
                    if self.nprocs == 1 :
                        list_env = fs.result() 
                    else : 
                        list_env = fs.result()[0]
                case "graph" : 
                    fs = exe.submit(self.graph_nauty)
                    if self.nprocs == 1 : 
                        list_env = fs.result()
                    else : 
                        list_env = list(chain(*fs.result()))
                case "cna/graph" : 
                    fs = exe.submit(self.cna_graph_nauty)
                    if self.nprocs == 1 : 
                        list_env = fs.result()
                    else : 
                        list_env = fs.result()[0] 
                case _:
                    raise Exception("Atomic environment style not known")

        #From list of atomic environment we create a dictionary, and update system.environment
        diff_env = set(list_env)
        diff_env = list(diff_env) 

        ##List of dictionnaries of different Topo ID and 
        dict_env = []
        for ID in diff_env : 
            #index with same ID : 
            indexsame = [i for i,e in enumerate(list_env) if e == ID]
            tmp = {"ID" : ID, 
                   "atom index" : indexsame}
            dict_env.append(tmp)
        self.system.environment = dict_env


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
            write_lammps_data(lammps_data_file, self.system, masses=True)
            if self.dimension == 2 : 
                modify_lammps_data_2D(lammps_data_file)

        #TODO Voir comment on gère les paramètres de bases de lammps (meme histoire que minimization)
        #lammps: 
        lmp = lammps(comm=comm, cmdargs=['-log', 'log_cna.lammps']) 
        lmp.command('units metal')
        lmp.command('atom_style atomic')
        lmp.command('dimension 3') 
        lmp.command('boundary p p p')
        lmp.command('atom_modify sort 0 1')

        lmp.command('read_data {}'.format(lammps_data_file))
        lmp.command('pair_style zero {} full'.format(self.atomenv_params['rcut'])) #CNA need a potential (i guess is for the neighborlist)
        lmp.command('pair_coeff * *')
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
        split = np.array_split(range(self.system.get_global_number_of_atoms()), nprocs)
        local_index = split[rank] 

        #Create graphs , can use other commented functions that we tested 
        #Here, need diagonal cell for box_size in k-d tree
        list_g = make_graph(self.system, local_index, rnei, rcut)
        list_topo = [] 
        for g in list_g : 
            list_topo.append(pynauty.certificate(g))
        return list_topo
    

    def cna_graph_nauty(self) : 
        """
        Compute CNA, and for non cristalline environement, compute the graph certificate
        """
        #TODO use the cna methods, but with executorlib might be problematic
        #TODO Better gather results

        #Compute CNA : 
        
        #for MPI : 
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()

        #Write lammps data file : 
        lammps_data_file = 'initial_config_minimization.lmp'
        if rank == 0 :
            write_lammps_data(lammps_data_file, self.system, masses=True)
            if self.dimension == 2 : 
                modify_lammps_data_2D(lammps_data_file)

        #TODO Voir comment on gère les paramètres de bases de lammps (meme histoire que minimization)
        #lammps: 
        lmp = lammps(comm=comm, cmdargs=['-log', 'log_cna.lammps']) 
        lmp.command('units metal')
        lmp.command('atom_style atomic')
        lmp.command('dimension 3') 
        lmp.command('boundary p p p')
        lmp.command('read_data {}'.format(lammps_data_file))
        lmp.command('pair_style zero {} full'.format(self.atomenv_params['rcut'])) #CNA need a potential (i guess is for the neighborlist)
        lmp.command('pair_coeff * *')
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
            #Find non cristalline atoms : 
            noncrist_atom_index = [i for i,e in enumerate(global_result[:,1]) if e == 5]
            #Split index atoms in approximatively even number sublist
            split = np.array_split(noncrist_atom_index, nprocs)
        else : 
            split = None
        #Send split to all procs 
        split = comm.bcast(split, root=0)
        #local index
        local_index = split[rank]
        list_g = make_graph(self.system, local_index, self.atomenv_params['rnei'], self.atomenv_params['rcut'])
        #gather graphs : 
        result = np.column_stack((local_index, np.array(list_g)))
        graphs = comm.gather(result, root=0)
        if rank == 0 : 
            graphs = np.concatenate(graphs)
            list_topo = [] 
            for i in range(self.system.get_global_number_of_atoms()) : 
                if i in graphs[:,0] :
                    ind = np.where(graphs[:,0] == i)[0][0] 
                    list_topo.append(pynauty.certificate(graphs[:,1][ind]))
                else : 
                    list_topo.append('crist')
            return list_topo
        

def make_graph(atoms, list_id, rnei, rcut) : 
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
        #TODO Not necessary
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



#    def cna_graph_nauty2(self) : 
#        """
#        Full use of Lammps to compute CNA and graph using neighborlist 
#        Use two potential pair style with rcut and rnei cutoff to define two different neighborlist
#        """
#        
#        #for MPI : 
#        comm = MPI.COMM_WORLD
#        rank = comm.Get_rank()
#        nprocs = comm.Get_size()
#
#
#        #Write lammps data file : 
#        lammps_data_file = 'initial_config_minimization.lmp'
#        if rank == 0 :
#            write_lammps_data(lammps_data_file, self.atoms, masses=True)
#            if self.dimension == 2 : 
#                modify_lammps_data_2D(lammps_data_file)
#
#
#        #Lammps : 
#        lmp = lammps(comm=comm)
#            #basic parameters
#        lmp.command('units metal')
#        lmp.command('atom_style atomic')
#        lmp.command('dimension 3') 
#        lmp.command('boundary p p p')
#        lmp.command('atom_modify map yes')
#            #read data file
#        lmp.command('read_data {}'.format(lammps_data_file))
#            #potential
#        lmp.command('pair_style zero 5.0 full')
#        lmp.command('pair_coeff * *')
#            #compute cna
#        lmp.command('neighbor 0.0 bin')
#        lmp.command('neigh_modify every 1 delay 0 check yes')
#        lmp.command('compute c1 all cna/atom {}'.format(self.atomenv_params['rnei']))
#        lmp.command('run 0')
#
#
#        #Extract cna
#        cna_array = lmp.numpy.extract_compute("c1", 1,1)
#        print(len(cna_array))
#        #Extract atom id 
#        tag = lmp.numpy.extract_atom("id")
#        tag -= 1
#        print(len(tag))
#        #id = id-1 #Lammps index start at 1
#        ##Find non cristalline atoms id
#        #non_crist_id = np.where(cna_array == 5)[0]
#
#        ## Find neighbor list rcut 
#        nlidx_rcut = lmp.find_pair_neighlist('zero')
#        nl_rcut = lmp.numpy.get_neighlist(nlidx_rcut)
#        #print(nl_rcut)
#        ## Find neighbor list rnei 
#        #nlidx_rnei = lmp.find_pair_neighlist('lj/cut')
#        #nl_rnei = lmp.numpy.get_neighlist(nlidx_rnei)
#
#        _, a = nl_rcut.get(0)
#        print(a)
#
#        #test = lmp.map(1000)
#        #print(test)
#        print("LAMMPS version:", lmp.version())
#
#
#
#
#
#        #list_topo = []
#        #for i in range(len(id)) : 
#        #    if i in non_crist_id : 
#        #        print(i, ' Non cristalline atom: ', id[i])
#        #        _, env = nl_rcut.get(i)
#        #        #all atoms in the environment of the central non cristalline atom
#        #        env = np.append(env, i)
#        #        #env = [id[i] for i in env]
#        #        print(len(env))
#
#        ##        adjacency_dict = {id: [] for id in env}
#
#                #Loop over atoms in environment
#         #       for j in env : 
#         #           #find neighbor atoms define by rnei 
#         #           _, neigh = nl_rnei.get(id[j])
#         #           #neighbor atoms need to be in the environement : 
#         #           print('neigh', neigh)
#         #           #neigh = np.intersect1d(env,neigh)
#         #           #print(neigh)
#
#
#
#    def cna_graph_nauty3(self) : 
#        """
#        Full use of Lammps to compute CNA and find pair list and distances for non cristalline atoms 
#        """
#        
#        #for MPI : 
#        comm = MPI.COMM_WORLD
#        rank = comm.Get_rank()
#        nprocs = comm.Get_size()
#
#
#        #Write lammps data file : 
#        lammps_data_file = 'initial_config_minimization.lmp'
#        if rank == 0 :
#            write_lammps_data(lammps_data_file, self.atoms, masses=True)
#            if self.dimension == 2 : 
#                modify_lammps_data_2D(lammps_data_file)
#
#
#        #Lammps : 
#        lmp = lammps(comm=comm)
#            #basic parameters
#        lmp.command('units metal')
#        lmp.command('atom_style atomic')
#        lmp.command('dimension 3') 
#        lmp.command('boundary p p p')
#            #read data file
#        lmp.command('read_data {}'.format(lammps_data_file))
#            #potential
#        lmp.command('pair_style zero 5.0 full')
#        lmp.command('pair_coeff * *')
#            #compute cna
#        lmp.command('neighbor 0.0 bin')
#        lmp.command('neigh_modify every 1 delay 0 check yes')
#        lmp.command('compute c1 all cna/atom {}'.format(self.atomenv_params['rnei']))
#        lmp.command('compute c2 all property/local patom1 patom2')
#        lmp.command('compute dist all pair/local dist')
#        lmp.command('run 0')
#
#
#        #Extract cna
#        cna_array = lmp.numpy.extract_compute("c1", 1,1)
#        #Extract atom id 
#        tags = lmp.numpy.extract_atom("id")
#        #Find non cristalline atoms id
#        non_crist_id = np.where(cna_array == 5)[0]
#        non_crist_id = tags[non_crist_id]
#        print(non_crist_id)
#        # look up the neighbor list
#        nlidx = lmp.find_pair_neighlist('zero')
#        nl = lmp.numpy.get_neighlist(nlidx)
#
#
#        #pairs distance of atoms in rcut define by pair style
#        dists = lmp.numpy.extract_compute('dist',  2, 1)
#        #pairs of atoms in rcut define by pair style
#        pairs = lmp.numpy.extract_compute('c2', 2, 2)
#
#        #all pairs distances
#        allpairs = comm.allgather(pairs)
#        allpairs = list(chain(*allpairs)) #to flat 
#        allpairs = np.array(allpairs)
#        #all distances : 
#        alldists = comm.allgather(dists)
#        alldists = list(chain(*alldists))
#
#
#        #! TOCHANGE np where on big system taking to much time
#        list_topo = []
#        for i in tags : 
#            if i in non_crist_id : 
#                env1 = [int(p[1]) for p in allpairs if p[0] == i]
#                env2 = [int(p[0]) for p in allpairs if p[1] == i]
#                env = env1+env2
#                env.append(i) #add central atom
#                #Create graph for the ith atom : 
#                map_graph_index = {e: i for i,e in enumerate(env)}
#                adjacency_dict = {i: [] for i in range(len(env))}
#                #Double loop on atoms in the atomic environment of i 
#                for id1 in env :
#                    for id2 in env : 
#                        if id1 != id2 : 
#                            k = np.where(((id1 == allpairs[:,0]) & (id2 == allpairs[:,1]))|((id1 == allpairs[:,1]) & (id2 == allpairs[:,0])))[0]
#                            if len(k) > 0 : 
#                                k = k[0]
#                                if  alldists[k] < 3.0 : 
#                                    adjacency_dict[map_graph_index[id1]].append(map_graph_index[id2])
##                print(adjacency_dict)
#                graph = pynauty.Graph(number_of_vertices=len(env),
#                                      adjacency_dict=adjacency_dict,
#                                    )
#                list_topo.append(pynauty.certificate(graph))
#            else : 
#                list_topo.append('crist')
#
#
#
#
#        #sort list based on tags : 
#        list_topo = [e for i, e in sorted(zip(tags, list_topo))]
#
#        return list_topo
#
#
#
#    def cna_graph_nauty4(self) : 
#        """
#        without neighboor from lammps
#        """
#        
#        #for MPI : 
#        comm = MPI.COMM_WORLD
#        rank = comm.Get_rank()
#        nprocs = comm.Get_size()
#
#
#        #Write lammps data file : 
#        lammps_data_file = 'initial_config_minimization.lmp'
#        if rank == 0 :
#            write_lammps_data(lammps_data_file, self.atoms, masses=True)
#            if self.dimension == 2 : 
#                modify_lammps_data_2D(lammps_data_file)
#
#
#        #Lammps : 
#        lmp = lammps(comm=comm)
#            #basic parameters
#        lmp.command('units metal')
#        lmp.command('atom_style atomic')
#        lmp.command('dimension 3') 
#        lmp.command('boundary p p p')
#            #read data file
#        lmp.command('read_data {}'.format(lammps_data_file))
#            #potential
#        lmp.command('pair_style zero 5.0 full')
#        lmp.command('pair_coeff * *')
#            #compute cna
#        lmp.command('neighbor 0.0 bin')
#        lmp.command('neigh_modify every 1 delay 0 check yes')
#        lmp.command('compute c1 all cna/atom {}'.format(self.atomenv_params['rnei']))
#        lmp.command('run 0')
#
#
#        #Extract cna
#        cna_array = lmp.numpy.extract_compute("c1", 1,1)
#        #Extract atom id 
#        tags = lmp.numpy.extract_atom("id")
#        #Find non cristalline atoms id
#        non_crist_id = np.where(cna_array == 5)[0]
#        non_crist_id = tags[non_crist_id]
#
#        # look up the neighbor list
#        nlidx = lmp.find_pair_neighlist('zero')
#        nl = lmp.numpy.get_neighlist(nlidx)
#        
#        positions = self.atoms.get_positions() 
#        cell = self.atoms.get_cell()
#        # Construire le KDTree avec les positions répliquées
#        tree = cKDTree(positions, boxsize=np.diag(cell))
#
#        list_topo = []
#        for i in tags : 
#            if i in non_crist_id :
#                atominenv_idx = tree.query_ball_point(positions[i], self.atomenv_params['rcut'])
#                atominenv_idx = list(atominenv_idx)
#                # Renuméroter les indices localement (0 à number_of_vertices-1)
#                map_graph_index = {e: i for i,e in enumerate(atominenv_idx)} 
#                #adjacency_dict = {i: [] for i in range(len(atominenv_idx))}
##                adjacency_dict = {}
#                
#                graph = pynauty.Graph(len(atominenv_idx))
#                for ind in atominenv_idx:
#                    # Recherche des voisins dans le rayon rnei
#                    neighbors_local = tree.query_ball_point(positions[ind], self.atomenv_params['rnei'])
#                    neighbors_local.remove(ind)
#                    neighbors_local = list(set(neighbors_local).intersection(atominenv_idx))
##                    adjacency_dict[map_graph_index[ind]] = [map_graph_index[e] for e in neighbors_local]
##                print(adjacency_dict)
#                #graph = pynauty.Graph(number_of_vertices=len(atominenv_idx),
#                #                      adjacency_dict=adjacency_dict,
#                 #                   )
#                    graph.connect_vertex(map_graph_index[ind],[map_graph_index[e] for e in neighbors_local] )
#                list_topo.append(pynauty.certificate(graph))
#                #print(adjacency_dict)
#            else : 
#                list_topo.append('crist')
#        #sort list based on tags : 
#        list_topo = [e for i, e in sorted(zip(tags, list_topo))]
#
#        return list_topo
#     
#  
#def make_graph1(atoms, list_id, rnei, rcut) : 
#    """
#    Create graph for all atoms with index in the list_id
#    Using ASE get_distances
#    """
#    #index all atoms to compute distances : 
#    ind = np.linspace(0, atoms.get_global_number_of_atoms()-1, atoms.get_global_number_of_atoms()).astype(int)
#    #liste graphe
#    list_g = []
#    for k in list_id : 
#        dist = atoms.get_distances(k, ind, mic=True)
#
#        N = len(np.where(dist<rcut)[0])
#        #index in rcut sphere
#        indin = np.where(dist<rcut)[0]
#        g = pynauty.Graph(N)
#
#        #create graphe connextion
#        for i in range(N) : 
#            tmp = np.where(atoms.get_distances(indin[i], indin, mic=True) < rnei)[0]
#            tmp = tmp.tolist()
#            tmp.remove(i)
#            g.connect_vertex(i, tmp)
#        list_g.append(g)
#    return list_g
#
##@profile 
#def make_graph2(atoms, list_id, rnei, rcut) : 
#    """ 
#    Make graph using neighborlist ASE 
#    """
#    list_g = [] 
#    #Create NeighborList for all atoms : 
#    cutoffs = atoms.get_global_number_of_atoms()*[rcut/2] 
#    nlall = NeighborList(cutoffs, self_interaction=True, bothways=True)
#    nlall.update(atoms) 
#    #loop over all atoms in list_id : 
#    for k in list_id : 
#        #create subatoms (otherwise conflict with the way pynauty constructs graph)
#        subatomsnei = nlall.get_neighbors(k)[0].tolist()
#        subatoms = Atoms(positions=atoms.get_positions()[subatomsnei], cell=atoms.get_cell(), pbc=True)
#        #Create Neighborlist for subatoms 
#        cut = subatoms.get_global_number_of_atoms()*[rnei/2]
#        nl = NeighborList(cut, self_interaction=False, bothways=True)
#        nl.update(subatoms)        
#
#        g = pynauty.Graph(len(subatomsnei))
#
#        #Loop over all neighbor 
#        for i in range(len(subatomsnei)):
#            g.connect_vertex(i, nl.get_neighbors(i)[0].tolist())
#        list_g.append(g)
#
#    return list_g
#
#def make_graph3(atoms, list_id, rnei, rcut) : 
#    """
#    Test using scipy cKDTree, it means that we need do deal with pbc by hand 
#    """
#    list_g = [] 
#    positions = atoms.get_positions() 
#    cell = atoms.get_cell()
#    #get positions and extend to take into account the PBC 
#    shifts = np.array([[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)])
#    replicated_positions = np.vstack([positions + np.dot(shift, cell) for shift in shifts])
#    # Construire le KDTree avec les positions répliquées
#    tree = cKDTree(replicated_positions)
#    
#    # Construire un graphe pour chaque atome
#    for atom_idx in list_id:
#        # Trouver les indices des voisins dans le rayon rcut
#        neighbors_idx = tree.query_ball_point(positions[atom_idx], rcut)
#
#        # Renuméroter les indices localement (0 à number_of_vertices-1)
#        neighbors_global = list(neighbors_idx)  # Inclut les atomes répliqués #????
#        local_to_global = {local_idx: global_idx for local_idx, global_idx in enumerate(neighbors_global)}
#        global_to_local = {v: k for k, v in local_to_global.items()}
#
#
#        #CDIST
#        # Utiliser cdist pour calculer toutes les distances entre les voisins
#        dist_matrix = cdist(replicated_positions[neighbors_global], replicated_positions[neighbors_global])
#
#        # Construire le dictionnaire d'adjacence avec indices locaux
#        adjacency_dict = {global_to_local[i]: [] for i in neighbors_global}
#
#        # Associer correctement les indices dans dist_matrix
#        for i in range(len(neighbors_global)):
#            for j in range(len(neighbors_global)):
#                if i != j:  # Pas d'auto-boucle
#                    if dist_matrix[i, j] <= rnei:
#                        adjacency_dict[i].append(j)
#        #END CDIST 
#
#        #Local Tree
#
#        # Construire le dictionnaire d'adjacence avec indices locaux
#        #adjacency_dict = {local_idx: [] for local_idx in local_to_global.keys()}
#
#        ## Trouver les voisins dans le rayon rnei en utilisant le deuxième KDTree
#        #for i, ind in enumerate(neighbors_global):
#        #    # Recherche des voisins dans le rayon rnei
#        #    neighbors_local = tree.query_ball_point(replicated_positions[ind], rnei)
#        #    for neighbor in neighbors_local:
#        #        if neighbor != ind and neighbor in neighbors_global:  # Ne pas ajouter soi-même comme voisin
#        #            adjacency_dict[i].append(global_to_local[neighbor])  # Utilisation des indices locaux
#        #END Local Tree
#
#        
#        graph = pynauty.Graph(
#            number_of_vertices=len(neighbors_idx),
#            adjacency_dict=adjacency_dict,
#            directed=False
#        )
#        
#        list_g.append(graph)
#    return list_g
#
#
#
#
#
#