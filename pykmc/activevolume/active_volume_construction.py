from ase import Atoms, Atom
from ase.build import bulk
from lammps import lammps
import numpy as np
from ase.data import atomic_numbers, atomic_masses
from ..system import System
from ..config import Config

class ActiveVolume:
    '''
    Defines an active volume for a defect
    '''


    def __init__(self, system: System, lmp: , config : Config) -> None:
        self.og_system = system #System to have active volume defined
        self.lmp_instance = lmp
        self.config = config

    #need to define a self class for specifically active volumes

    def define_active_volume(self, central_atom_idx: int) -> None:

        '''
        Want to make it so the radius and such are defined in the config file later
        Need to pass the ID of the atom getting checked. It will be in ORIGINAL system units, so it'll need to be mapped

            Parameters
        ----------
        radius: float
            Radius of active volume in latsize units (can change to angstrom)
        ratio: float
            Ratio of the active volume that is allowed to move
        central_atom_idx : int
            The central atom index.
        '''

        #Defining parameters
        #Radius of whole active volume in Ang
        r_a=self.config.rnei*1.5 #Ensures AV is larger than topology analysis
        #Radius where atoms can move
        r_m=self.config.rnei*1.1

        #Build system
        self.lmp_instance.command('units metal')
        self.lmp_instance.command('atom_style atomic')
        self.lmp_instance.command('dimension 3')
        self.lmp_instance.command('boundary p p p')

        cell_x, cell_y, cell_z = self.og_system.cell.cellpar()[:3]

        #For testing, going to assume that the SIA is inserted into the center of the cell, and this is where the active volume will be
        center_x = cell_x / 2
        center_y = cell_y / 2
        center_z = cell_z / 2

        positions = self.og_system.positions

        center=positions[central_atom_idx]

        inner_movable_indices = []
        total_active_indices = []
        non_active_indices = []

        for i, pos in enumerate(positions):
            distance = np.linalg.norm(pos - center)
            if distance <= r_m:
                inner_movable_indices.append(i)
                total_active_indices.append(i) # Inner is also part of total active
            elif distance <= r_a:
                total_active_indices.append(i)
            else:
                non_active_indices.append(i)

        buffer_indices = list(set(total_active_indices) - set(inner_movable_indices))
        self.inner_movable_indices = np.array(sorted(inner_movable_indices)) # Ensure sorted for consistent indexing
        self.buffer_indices = np.array(sorted(buffer_indices))
        non_active_indices = sorted(non_active_indices)
        self.av_indices = sorted(np.concatenate([self.inner_movable_indices, self.buffer_indices])) #Sorts all the atoms

        cell = self.og_system.cell
        types = [self.og_system.types[i] for i in self.av_indices]
        positions = self.og_system.positions[self.av_indices]
        print("Number of atoms in AV", len(positions))
        pbc=self.og_system.pbc

        self.av = System(types, positions, cell, pbc, self.av_indices)

    def make_active_volume(self) -> None:

        '''
        Makes the active volume in lammps. This is where the map for ID's gets made
        '''

        # -------------------Setting up AV----------------------
        natoms = len(self.av.types)
        cell = self.av.cell
        xhi, yhi, zhi = cell[0][0], cell[1, 1], cell[2, 2]
        types = self.av.types
        positions_0 = self.av.positions
        x0 = positions_0.flatten()  # Lammps format
        #--------------------Now need to make the system in lammps so it can be refactored----------------------------

        # map type to int alphabetic order create a dictionary with atom id and mass, eg {'H' : {'ref': 1, 'mass' : 1.00}, 'Ni': {'ref' : 2, 'mass' : 58.69} }
        map_type = {
            atom_type: {"ref": i + 1, "mass": atomic_masses[atomic_numbers[atom_type]]}
            for i, atom_type in enumerate(sorted(set(types)))
        }
        types = [map_type[element]["ref"] for element in types]  # map to integer

        self.lmp_instance.command("units metal")
        self.lmp_instance.command("atom_style atomic")
        self.lmp_instance.command("dimension 3")
        self.lmp_instance.command("boundary p p p")
        self.lmp_instance.command("atom_modify sort 0 0.0")
        self.lmp_instance.command(
            "region box block 0.0 {} 0.0 {} 0.0 {}".format(xhi, yhi, zhi)
        )

        self.lmp_instance.command("create_box {} box".format(len(map_type)))

        #------------------------Create Atoms----------------------
        self.lmp_instance.create_atoms(natoms, self.av_indices, types, x0) #Creates atoms in original indexed order
        # Set masses
        for key in map_type.keys():
            self.lmp_instance.command(
                "mass {} {}".format(map_type[key]["ref"], map_type[key]["mass"])
            )
        # Label atoms name to type :
        self.lmp_instance.command(
            "labelmap atom "
            + " ".join(f"{int(e['ref'])} {key}" for key, e in map_type.items())
        )

        self.lmp_instance.command('write_dump all custom ./pre_compressed_av.dump id x y z')

        #------------------Now want refactored atoms-----------------------------
        self.lmp_instance.command('reset_atoms id')
        #---------------------------Now Map-------------------------------

        self.lmp_instance.command('write_dump all custom ./post_compressed_av.dump id x y z')
        # Now need to gather the positions in order to create map

        x1 = self.lmp_instance.gather_atoms("x", 1, 3) # Since lammps ID starts from 1, we can use this to map ID's without calling for them.
        x1 = np.ctypeslib.as_array(x1)
        self.lmp_positions = np.reshape(x1, (-1, 3))

        self.map = self.map_atom_indices() #This will make the map needed for pARTn

        lammps_inner_movable_ids = []
        lammps_buffer_ids = []
        # Convert target arrays to sets for efficient O(1) average time lookup
        inner_movable_set = set(self.inner_movable_indices)
        buffer_set = set(self.buffer_indices)

        for i in range(len(self.map)):
            # self.map[i] represents an original ID
            # Check if this original ID is present in the set of inner_movable_indices
            if self.map[i] in inner_movable_set:
                lammps_inner_movable_ids.append(i + 1)  # i is the 0-based new_ID, +1 if LAMMPS IDs are 1-based

            # Check if this original ID is present in the set of buffer_indices
            if self.map[i] in buffer_set:
                lammps_buffer_ids.append(i + 1)  # i is the 0-based new_ID, +1 if LAMMPS IDs are 1-based


        self.lmp_instance.command('pair_style eam/alloy')
        self.lmp_instance.command('pair_coeff * * ../../Cu01.eam.alloy Cu')
        self.lmp_instance.command('thermo 10')
        self.lmp_instance.command('thermo_style custom step temp pe lx ly lz press pxx pyy pzz')
        #
        # #Defining groups of atoms
        #
        self.lmp_instance.command(f"group inner_movable id {' '.join(map(str, lammps_inner_movable_ids))}")
        if len(lammps_buffer_ids) > 0:
            self.lmp_instance.command(f"group buffer id {' '.join(map(str, lammps_buffer_ids))}")
            self.lmp_instance.command("fix f_buffer buffer setforce 0.0 0.0 0.0")
        else:
            print('No buffer atoms defined')


        #--------------------------Active volume now defined, can run pARTn and such on it

        '''
        What ever code that's needed goes here
        '''




    def map_atom_indices(self) -> np.ndarray:
        """
        Create a mapping from new IDs (after 'reset_ids all compress yes')
        back to original IDs (before reset) using positions.

        Parameters:
            combined_array_0 (ndarray): N x 4 array [old_ID, x, y, z] before reset
            atom_positions (ndarray): N x 3 array [x, y, z] after reset (ordered by new_ID)

        Returns:
            new_to_old (dict): {new_ID: old_ID}
        """
        old_IDs = self.av_indices
        old_positions = self.av.positions

        # Round positions to avoid floating-point mismatch issues
        precision = 12
        old_positions_rounded = np.round(old_positions, precision)
        atom_positions_rounded = np.round(self.lmp_positions, precision)

        # Convert positions to structured array for fast set lookups
        dtype = [('x', float), ('y', float), ('z', float)]

        old_struct = old_positions_rounded.copy(order='C').view(dtype).squeeze()
        new_struct = atom_positions_rounded.copy(order='C').view(dtype).squeeze()

        # Build a dictionary from position → old ID
        pos_to_old = {tuple(p): id_ for p, id_ in zip(old_struct, old_IDs)}

        # Map each new position to old ID
        new_to_old = np.array([pos_to_old[tuple(p)] for p in new_struct], dtype=int)

        return new_to_old


    def shift_atoms(self) -> None:

        x3 = self.lmp_instance.gather_atoms("x", 1, 3)  # Since lammps ID starts from 1, we can use this to map ID's without calling for them.
        x3 = np.ctypeslib.as_array(x3)
        self.lmp_new_positions = np.reshape(x3, (-1, 3))

        for i in range(len(self.lmp_positions)):

            self.og_system.positions[self.map[i]]=self.lmp_new_positions[i]

