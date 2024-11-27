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
        if catalog == None : 
            self.catalog = pd.DataFrame(columns=['event_id', 
                                                 'initial_positions', 
                                                 'saddle_positions', 
                                                 'final_position', 
                                                 'energy_barrier', 
                                                 'k'])
        else : 
            self.catalog = catalog #for restart
        
    def minimize(self, minimization_style, minimization_params, potential, dimension=3, nprocs=1, backend='local') : 
        """ 
        Minimize the system and update system positions 
        """
        from .minimization import Minimization 
        minimizer = Minimization(self, minimization_style, minimization_params, potential, dimension, nprocs, backend)
        minimizer.run()

    def find_environment(self, environment_style, environement_params, dimension=3, nprocs=1) : 
        """ 
        Find atomic environment for each atoms in System and create a dictionary 
        """
        from .atomic_environment import AtomicEnvironment 
        atomic_environment = AtomicEnvironment(self, environment_style, environement_params, dimension, nprocs)
        atomic_environment.run()

    def event_search(self, search_style, search_params, potential, dimension=3, nprocs=1) : 
        """ 
        For each atomic environment ID that are not in the catalog (except 'crist'), we launch 20 ARTn search 
        and add event to the catalog
        """
        from .event import EventSearch 
        event_search = EventSearch(self, search_style, search_params, potential, dimension, nprocs)
        event_search.run()