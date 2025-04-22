from ase.io import read
import numpy as np
import ase.geometry

class System():
    """ 
    """ 
    def __init__(self) : 
        self.types = None
        self.positions = None 
        self.cell = None
        self.pbc = None
        self.index = None

    @classmethod
    def create_from_file(cls, file_path: str) : 
        #Create ase.Atoms from file
        try : 
            atoms = read(file_path) 
        except Exception as e : 
            raise ValueError(f"Can't create System from file {file_path}: {e}")

        #Create new System instance
        new_system = cls()
        #update attributes
        new_system.types = atoms.get_chemical_symbols() 
        new_system.positions = atoms.get_positions() 
        new_system.cell = atoms.get_cell()
        new_system.pbc = atoms.get_pbc()
        new_system.index = np.linspace(0, len(new_system.types)-1, len(new_system.types)).astype(int)

        return new_system

    def update_positions(self, new_positions, atom_idx=None) : 
        if atom_idx is None : 
            self.positions = new_positions
            self.positions = self.wrap_positions(self.positions, cell= self.cell, pbc=self.pbc)
            self.positions[self.positions < 0] = 0 #This is because I can have small negative number and it messes up with kdtree

        else : 
            self.positions[atom_idx] = new_positions
            self.positions = self.wrap_positions(self.positions, cell = self.cell, pbc=self.pbc)
            self.positions[self.positions < 0] = 0 #This is because I can have small negative number and it messes up with kdtree
    
    def wrap_positions(self, positions, cell, pbc=True) : 
        return ase.geometry.wrap_positions(positions=positions, cell=cell, pbc=pbc)