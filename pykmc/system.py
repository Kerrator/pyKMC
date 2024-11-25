from ase import Atoms
from ase.io import read

class System(Atoms):

    def __init__(self, file_path):
        atoms = read(file_path)  # Load configurations from file 
        super().__init__(symbols=atoms.get_chemical_symbols(),
                         positions=atoms.get_positions(),
                         cell=atoms.get_cell(),
                         pbc=atoms.get_pbc())