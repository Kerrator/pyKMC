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
        
    def minimize(self, minimization_style, minimization_params, potential, dimension=3, nprocs=1) : 
        """ 
        Minimize the system and update system positions 
        """
        from .minimization import Minimization 
        minimizer = Minimization(self, minimization_style, minimization_params, potential, dimension, nprocs)
        minimizer.run()
