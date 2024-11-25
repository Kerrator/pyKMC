from ase import Atoms
from ase.io import read

class System(Atoms):
    """
    Extension of the Atoms Ase object 
    """

    def __init__(self, file_path):
        atoms = read(file_path)  # Load configurations from file 
        super().__init__(symbols=atoms.get_chemical_symbols(),
                         positions=atoms.get_positions(),
                         cell=atoms.get_cell(),
                         pbc=atoms.get_pbc())

        self.environment = None
        
    def minimize(self, minimization_style, minimization_params, potential, dimension=3, nprocs=1) : 
        """ 
        Minimize the system and update system positions 
        """
        from .minimization import Minimization 
        minimizer = Minimization(self, minimization_style, minimization_params, potential, dimension, nprocs)
        minimizer.run()

    def find_environment(self, environment_style, environement_params, dimension=3, nprocs=1) : 
        """ 
        Find atomic environment for each atoms in System and create a dictionary 
        """
        from .atomic_environment import AtomicEnvironment 
        atomic_environment = AtomicEnvironment(self, environment_style, environement_params, dimension, nprocs)
        atomic_environment.run()
