import random
from lammps import lammps
from mpi4py import MPI 
from .utilities import modify_lammps_data_2D
from ase.io.lammpsdata import write_lammps_data
import pypARTn2


class EventSearch() : 

    def __init__(self, system, search_style, search_params, dimension=3, nprocs=1) -> None:
        self.system = system 
        self.search_style = search_style
        self.search_params = search_params 
        self.dimension = dimension 
        self.nprocs = nprocs 


    def run(self) : 
        """ 
        Execute new event searchs 
        """
        #TODO Parallelization

        #Check if new atomic environment that are not in the catalog, if yes extract the environement id: 
        l_new_environement = self.new_environment()
        #For each id in l_new_environment, we will select randomly one atom with the corresponding ID (does this nsearch time)
        for id in l_new_environement : 
            #extract list of atoms in system.environment having the id 
            l_atoms = [dict['atom index'] for dict in self.system.environment if dict['ID'] == id][0] 
            #list of atoms on which we gonna do the search
            l_atoms_search = [random.choice(l_atoms) for _i in range(self.search_params['nsearch'])]
            #then we do a pART search and put the result of each search in self.system.catalog
            for atom_index in l_atoms_search : 
                self.pARTn_search(atom_index=atom_index)


    def new_environment(self) : 
        """ 
        Return list of atomic environements id of the current step that are not in the catalog
        """
        ids_catalog = self.system.catalog['event_id'].tolist()
        ids_current = [element['ID'] for element in self.system.environment]
        try:
            ids_current.remove('crist')
        except ValueError:
            pass
        l_new_environments = [ids for ids in ids_current if ids not in ids_catalog]
        return l_new_environments 
    
    def pARTn_search(self, atom_index, potential) : 
        """ 
        Use pARTn with Lammps to find events
        atom_index : atom on which we perform the search (the one that we init_push)
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

        #Setup pARTn : 
        artn = pypARTn2.artn( engine = "lmp" ) 








