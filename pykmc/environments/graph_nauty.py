import pynauty
import numpy as np


def graph(neighbors_list, environment_list, atom_idx= None) : 
    """ 
    """
    from mpi4py import MPI
    #MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    #Split index atoms in approximatively even number sublist
    if atom_idx == None : #graph for all atoms in system
        split = np.array_split(range(len(neighbors_list)), nprocs)
    else : 
        split = np.array_split(atom_idx, nprocs) #when using cna/graph
    local_index = split[rank] 

    list_g = make_graph(local_index, neighbors_list, environment_list)

    list_hash = [] 

    for g in list_g : 
        list_hash.append(pynauty.certificate(g))
    list_hash = comm.gather(list_hash, root=0)
    if rank == 0 : 
        list_hash = [gcertificate for e in list_hash for gcertificate in e]
        return list_hash 
    else : 
        return None

def make_graph(atoms_idx, neighbors_list, environment_list) : 
    """ 
    """
    list_g = [] 
    for idx in atoms_idx : 
        #To reorder indexes from 0 to number of vertexes-1 (for pynauty)
        local_to_global = {local_idx: global_idx for local_idx, global_idx in enumerate(environment_list[idx])}
        global_to_local = {v: k for k, v in local_to_global.items()}
        #Dictionary to map graph indexes to system indexes
        adjacency_dict = {local_idx: [] for local_idx in local_to_global.keys()}

        for i,at in enumerate(environment_list[idx]) : 
            for neighbor in neighbors_list[at] : 
                if neighbor in environment_list[idx] : 
                    adjacency_dict[i].append(global_to_local[neighbor]) 
        
        graph = pynauty.Graph(
            number_of_vertices=len(environment_list[idx]), 
            adjacency_dict=adjacency_dict, 
            directed=False
        )

        list_g.append(graph)

    return list_g



