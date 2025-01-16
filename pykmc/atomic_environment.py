from .utilities import modify_lammps_data_2D
from ase.io.lammpsdata import write_lammps_data
import numpy as np
from mpi4py import MPI
from lammps import lammps
from executorlib import Executor
import pynauty
from ase.neighborlist import NeighborList
#from itertools import chain
from ase import Atoms
#from profiling_decorator import profile
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
#from decimal import *
from subprocess import run

#TODO see doc convention when attributes and parameters are the same
#TODO write to file 
#TODO Voir comment on gère les paramètres de bases de lammps (meme histoire que minimization)
#TODO For each run procedure, check if the needed parameters are present
#TODO for the cna_graph() method should use the cna methods instead of rewriting it, but with executorlib might be problematic
#TODO Better gather results
#TODO use cell/verlet list for graphs
#TODO See if we can improve the way I connect in make_graph the graph atom index and system atom index (I think that some part are not necessary)

class AtomicEnvironment() :
    """
    Define and run the procedure to find the atomic environment of each atoms in the system

    Parameters
    ----------
    system : System Object
        system on which we perform the atomic environment search
    atomenv_style : str
        style use for the atomic environment search, can be 'cna', 'graph, 'cna/graph'
    atomenv_params : dict of str: float
        dictionaty of radius parameters defining the environment, 'rnei' : radius cutoff to define
        neirest neighbors, 'rcut' : radius cutoff to define the environment (for 'graph')
    dimension : int
        dimension of the system
    nprocs : int
        number of procs available
    backend : str
        parameter used by Executorlib, can be 'local', 'slurm_allocation', 'slurm_submission', by default 'local'
    
    Methods
    -------
    run() 
        run the atomic environment search procedure and update the system.environment
    write_to_file() 
    cna() 
        use Lammps to compute the Common Neighbor Analysis and attribute a 'crystal' or 'noncrystal' ID to each atoms
    graph_nauty()
        for each atoms, create a connectivity graph and attribute the corresponding ID based and nauty certificate 
    cna_graph_nauty() 
        use Lammps to compute the Common Neighbor Analysis, for atoms that does not have a crystalline environment 
        a connectivity graph is created. Attribute a 'crystal' or a nauty certificate ID to each atoms
    """        

    def __init__(self, system, atomenv_style, atomenv_params, dimension, nprocs, backend) : 
        self.system = system
        self.atomenv_style = atomenv_style
        self.atomenv_params = atomenv_params
        self.dimension = dimension
        self.nprocs = nprocs
        self.backend = backend

    def run(self): 
        """
        Run similar atomic environment search based on topology_style
        """

        #Run the atomic environment search
        with Executor(backend =self.backend) as exe : 
            match self.atomenv_style : 
                case "cna":
                    fs = exe.submit(self.cna, resource_dict={"cores": self.nprocs})
                case "graph" : 
                    fs = exe.submit(self.graph_nauty, resource_dict={"cores": self.nprocs})
                case "cna/graph" : 
                    fs = exe.submit(self.cna_graph_nauty, resource_dict={"cores": self.nprocs})
                case _:
                    self.system.logger.logger.error('ERROR:Atomic environment style not known')
                    raise Exception("Atomic environment style unknown")

        #Get results
        list_env = fs.result()
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

        #Add if debug  
        for i, e in enumerate(self.system.environment) : 
            self.system.logger.logger.debug("DEBUG: Atomic environment n° {} : {} atoms".format(i, len(e['atom index'])))
        self.system.logger.new_line()


    def write_to_file(self) : 
        """
        write similar atomic environment to file as a list of dict using yaml.
        """

    def cna(self) : 
        """ 
        compute CNA using lammps. 
        return a list of ID : "crystal" if the atom has a crystalline environement (ie CNA = 1,2,3 or 4) and "noncrystal" if not (ie CNA=5)  
        """

        #for MPI : 
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()

        #Write lammps data file : 
        lammps_data_file = 'initial_config_cna.lmp'
        if rank == 0 :
            write_lammps_data(lammps_data_file, self.system, masses=True)
            if self.dimension == 2 : 
                modify_lammps_data_2D(lammps_data_file)

        #Lammps: 
        lmp = lammps(comm=comm, cmdargs=['-log', 'log_cna.lammps', '-screen', 'none']) 
        lmp.command('units metal')
        lmp.command('atom_style atomic')
        lmp.command('dimension 3') 
        lmp.command('boundary p p p')
        lmp.command('atom_modify sort 0 1')
        lmp.command('read_data {}'.format(lammps_data_file))
        lmp.command('pair_style zero {} full'.format(self.atomenv_params['rcut'])) #CNA need a potential (I guess it is for the neighborlist)
        lmp.command('pair_coeff * *')
        lmp.command('compute c1 all cna/atom {}'.format(self.atomenv_params['rnei']))
        lmp.command('run 0')
        #Extract cna
        cna_array = lmp.numpy.extract_compute("c1", 1,1)

        #Lammps does not sort by atom index so we extract them too
        id = lmp.numpy.extract_atom("id")
        id = id-1 #Lammps index start at 1

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
                    list_topo.append('noncrystal') 
                else : 
                    list_topo.append('crystal')
            #Clean input file
            run('rm {}'.format(lammps_data_file), shell=True)
            lmp.close()
            return list_topo
    

    def graph_nauty(self) :
        """
        Compute pynauty certificate based on graph canonical form
        need rnei 
        need rcut
        """ 
        rnei = self.atomenv_params['rnei']
        rcut = self.atomenv_params['rcut']

        #MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()

        #Split index atoms in approximatively even number sublist
        split = np.array_split(range(self.system.get_global_number_of_atoms()), nprocs)
        local_index = split[rank] 

        #Create graphs , can use other commented functions that we tested (see commented make_graph functions) 
        #Here, need diagonal cell for box_size in k-d tree
        list_g = make_graph(self.system, local_index, rnei, rcut)
        list_topo = [] 
        for g in list_g : 
            #compute certificate
            list_topo.append(pynauty.certificate(g))
        return list_topo
    

    def cna_graph_nauty(self) : 
        """
        Compute CNA, and for non crystalline environement, compute the graph certificate
        """

        #for MPI : 
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()

        #Compute CNA : 
        #Write lammps data file : 
        lammps_data_file = 'initial_config_cna.lmp'
        if rank == 0 :
            write_lammps_data(lammps_data_file, self.system, masses=True)
            if self.dimension == 2 : 
                modify_lammps_data_2D(lammps_data_file)

        #lammps: 
        lmp = lammps(comm=comm, cmdargs=['-log', 'log_cna.lammps', '-screen', 'none']) 
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

        #Gather values
        result = np.column_stack((id, cna_array)) 
        global_result = comm.gather(result, root=0)

        #TEST : add also neighbors of non crystalline atoms : 
        #gather all positions 
        positions = lmp.gather_atoms("x", 1, 3)


        if rank == 0 : 
            #flaten arrays
            global_result = np.concatenate(global_result)
            #sort by atom index 
            global_result = global_result[global_result[:,0].argsort()]
            #Find non crystalline atoms : 
            noncrist_atom_index = [i for i,e in enumerate(global_result[:,1]) if e == 5]

            #Test : add also neighbors of non crystalline atoms : 
                #convert ctype positions into a numpy array
            positions = np.ctypeslib.as_array(positions)
            positions = np.reshape(positions, (-1, 3))
            neighbors = [] 
            ind = np.linspace(0, self.system.get_global_number_of_atoms()-1, self.system.get_global_number_of_atoms()).astype(int)
            for at_idx in noncrist_atom_index : 
                dist = self.system.get_distances(at_idx, ind, mic=True)
                #neighbors += np.where(dist <= self.atomenv_params['rnei'])[0].tolist()
                neighbors += np.where(dist <= self.atomenv_params['radd_cna'])[0].tolist()
            noncrist_atom_index += neighbors 
            noncrist_atom_index = list(set(noncrist_atom_index)) #remove duplicate






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
                    list_topo.append('crystal')
            #Clean input file
            run('rm {}'.format(lammps_data_file), shell=True)
            lmp.close()
            return list_topo
        

def make_graph(atoms, list_id, rnei, rcut) : 
    """
    Create graph, using scipy cKDTree, with boxsize to find neighbors 

    Parameters
    ----------
    atoms : System or Atoms Object
        current system with positions
    list_id : List[int]
        list of atoms index for which we create a graph
    rnei : float
        radial cutoff distance to define nearest neighbors
    rcut : float
        radial cutoff distance to define the environment
    Returns
    -------
    List[pynauty.Graph]
        list of pynauty graphs
    """    

    list_g = [] 
    positions = atoms.get_positions(wrap=True) 
    cell = atoms.get_cell()
    alat = cell[0][0]
    positions[positions<0] = 0
    tree = cKDTree(positions, boxsize=[alat]*3)
    
    # Creat graph for each atoms
    for atom_idx in list_id:
        #Find atoms index inside rcut
        atominenv_idx = tree.query_ball_point(positions[atom_idx], rcut)
        atominenv_idx = list(atominenv_idx)
        #To reorder indexes from 0 to number of vertexes-1 (for pynauty)
        local_to_global = {local_idx: global_idx for local_idx, global_idx in enumerate(atominenv_idx)}
        global_to_local = {v: k for k, v in local_to_global.items()}
        #Dictionary to map graph indexes to system indexes
        adjacency_dict = {local_idx: [] for local_idx in local_to_global.keys()}

        #Find nearest neighbor for each atoms in the environment
        for i, ind in enumerate(atominenv_idx):
            neighbors_local = tree.query_ball_point(positions[ind], rnei)
            for neighbor in neighbors_local:
                if neighbor != ind and neighbor in atominenv_idx:  #not have an atom that has itself as a neighbour
                    adjacency_dict[i].append(global_to_local[neighbor])  #Map 

        #Create graph 
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