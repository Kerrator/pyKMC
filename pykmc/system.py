from ase import Atoms
from ase.io import read
import pandas as pd

class System(Atoms):
    """
    Extension of the Atoms Ase object 
    """
    #TODO Should make a function that write lammps data file, or check if it s one instead of writing it every
    #time we call lammps 
    def __init__(self, file_path, catalog=None):
        atoms = read(file_path)  # Load configurations from file 
        super().__init__(symbols=atoms.get_chemical_symbols(),
                         positions=atoms.get_positions(),
                         cell=atoms.get_cell(),
                         pbc=atoms.get_pbc())

        self.environment = None

        #Note :  using Dataframe is usefull to do operation on it, but doing it that way, every time we 
        #add a new event, we have to copy the entire catalog. 
        #Dataframe are not made to be extended
        #could use dict of list 
        if catalog == None : 
            self.catalog = pd.DataFrame(columns=['event_id', 
                                                 'initial_positions', 
                                                 'saddle_positions', 
                                                 'final_positions', 
                                                 'energy_barrier', 
                                                 'k'])
        else : 
            self.catalog = pd.read_pickle(catalog) #for restart
        
        self.kmc_traj = None #for restart
        
    def minimize(self, minimization_style, minimization_params, potential, dimension=3, nprocs=1, backend='local') : 
        """ 
        Minimize the system and update system positions 
        """
        from .minimization import Minimization 
        minimizer = Minimization(self, minimization_style, minimization_params, potential, dimension, nprocs, backend)
        minimizer.run()

    def find_environment(self, environment_style, environement_params, dimension=3, nprocs=1, backend='local') : 
        """ 
        Find atomic environment for each atoms in System and create a dictionary 
        """
        from .atomic_environment import AtomicEnvironment 
        atomic_environment = AtomicEnvironment(self, environment_style, environement_params, dimension, nprocs, backend)
        atomic_environment.run()

    def event_search(self, search_style, search_params, potential, dimension=3, nprocs=1, backend='local') : 
        """ 
        For each atomic environment ID that are not in the catalog (except 'crist'), we launch 20 ARTn search 
        and add event to the catalog
        """
        from .event import EventSearch 
        event_search = EventSearch(self, search_style, search_params, potential, dimension, nprocs, backend)
        event_search.run()

    def kmc(self, kmc_parameters, minimization_params, atomenv_params, eventsearch_params, potential,dimension=3, backend='local') :
        """
        """ 
        from .kmc import KMC 
        kmc = KMC(self, kmc_parameters, minimization_params, atomenv_params, eventsearch_params, potential,dimension, backend)
        kmc.run()

    def point_set_registration(self, psr_style, idx_cat,central_atom_index, rcutevent, dimension=3, nprocs=1, backend='local') : 
        from .point_set_registration import PointSetRegistration 
        psr = PointSetRegistration(self, psr_style, idx_cat,central_atom_index, rcutevent, dimension, nprocs, backend)
        psr.run()