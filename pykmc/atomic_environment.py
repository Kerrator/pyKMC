from .utilities import modify_lammps_data_2D
from ase.io.lammpsdata import write_lammps_data
import numpy as np
from mpi4py import MPI
from lammps import lammps
from executorlib import Executor


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

    def run(self): 
        """
        Run similar atomic environment search based on topology_style
        """
        with Executor(max_cores=self.nprocs, cores_per_worker=self.nprocs) as exe : 
            match self.atomenv_style : 
                case "cna":
                    fs = exe.submit(self.cna)
                    self.list_env = fs.result()
                    #return self.list_env
                case "hausdorff_dist" : 
                    self.hausdorff_dist()
                case _:
                    raise Exception("Atomic environment style not known")
        #To dict : 
        #List of different atomic environment : 
        #diff_env = set(self.list_env)
        #diff_env = list(diff_env) 

        ##List of dictionnaries of different Topo ID and 
        #self.dict_env = []
        #for ID in diff_env : 
        #    #index with same ID : 
        #    indexsame = [i for i,e in enumerate(self.list_env) if e == ID]
        #    tmp = {"ID" : ID, 
        #           "atom index" : indexsame}
        #    self.dict_env.append(tmp)


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