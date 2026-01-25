from ase import Atoms, Atom
from ase.build import bulk
from lammps import lammps
import numpy as np
from ase.data import atomic_numbers, atomic_masses
from ase.geometry import find_mic
from ..system import System
from ..config import Config
from ..enginemanager.lmpi.lammps_operations import initialize_parameters, initialize_potential

class ActiveVolume:
    '''
    Defines an active volume for a defect
    '''


    def __init__(self, engine, config: Config, system: System, central_atom_idx: int) -> None:
        self.engine = engine
        self.config = config
        self.system = system
        self.central_atom_idx = central_atom_idx
    #need to define a self class for specifically active volumes

    def define_AV(self):
        # Defining parameters
        # Radius of whole active volume in Ang
        r_a = self.config.partn.r_act  # Ensure AV is larger than topology analysis
        # Defines the radius of atoms that can move.
        r_m = self.config.partn.r_mov

        # NEED TO ADD WARNING IF R_A<R_M

        center = self.system.positions[self.central_atom_idx]

        inner_movable_idx = []
        buffer_idx = []
        total_active_idx = []
        non_active_idx = []

        for i, pos in enumerate(self.system.positions):
            diff = pos - center
            diff_mic, distance = find_mic(diff, self.system.cell, pbc=True)
            if np.abs(distance) <= r_m:
                inner_movable_idx.append(i)
                total_active_idx.append(i)  # Inner is also part of total active
            elif np.abs(distance) > r_m and np.abs(distance) <= r_a:  # Can change to make it between r_m and r_a
                buffer_idx.append(i)
                total_active_idx.append(i)
            else:
                non_active_idx.append(i)

        self.buffer_idx = np.array(sorted(buffer_idx))
        self.av_idx = np.array(sorted(total_active_idx))

        self.av_positions = self.system.positions[self.av_idx]

    def make_AV(self):
        self.atom_map = np.array(self.av_indices, dtype=int)

        # Define the buffer group based on the new LAMMPS IDs
        # We need to find which index in 'av_indices' corresponds to 'buffer_indices'
        engine_buffer_ids = []
        buffer_set = set(self.buffer_indices)
        for i, original_id in enumerate(self.av_indices):
            if original_id in buffer_set:
                engine_buffer_ids.append(i + 1)  # LAMMPS IDs are 1-based

        if engine_buffer_ids:
            self.engine.command(f"group buffer id {' '.join(map(str, engine_buffer_ids))}")
            self.engine.command("fix f_buffer buffer setforce 0.0 0.0 0.0")
        else:
            self.engine.command(f"group buffer empty")
            self.engine.command("fix f_buffer buffer setforce 0.0 0.0 0.0")
            print('No buffer atoms defined')

        self.engine.command('run 0 post no')

    def reset(self) -> None:
        """
        Clear lammps instance, preps it for the new sim:
        """

        self.engine.command('clear')
        initialize_parameters(self.engine)
        # Create cell
        xhi, yhi, zhi = self.system.cell[0][0], self.system.cell[1, 1], self.system.cell[2, 2]
        self.engine.command(
            "region box block 0.0 {} 0.0 {} 0.0 {}".format(xhi, yhi, zhi)
        )
        self.engine.command('create_box 1 box')  # NEEDS TO BE UPDATED FOR ALLOYS
        initialize_potential(self.engine, self.config)

    def clear(self):
        '''
        Clears lammps instance
        '''
        self.engine.command('clear')

    def redefine_atoms(self, positions) -> None:
        '''
            Check to see if current lammps system has enough atoms
            If not, deletes all atoms then redefines them
            '''

        # if hasattr(engine, 'engine_comm'):
        #     engine.engine_comm.Barrier()

        # num_atoms = engine.lmp.get_natoms()
        # if len(positions) == num_atoms: # Enough atoms in system
        #     set_positions(engine, positions)
        #     engine.command('fix 1 all setforce 0.0 0.0 0.0')
        #     engine.command('run 0')
        #     engine.command('unfix 1')
        # else: #Uneven number of atoms
        # if num_atoms!=0:
        #     engine.command('delete_atoms group all')
        type = [1] * len(positions)
        # print('Creating atoms for refinement atoms')
        new_positions = positions.flatten().astype(np.float64)
        ids = np.arange(1, len(positions) + 1, dtype=np.int32)
        self.engine.lmp.create_atoms(len(positions), ids, type, x=new_positions)
        self.engine.command("comm_style tiled")
        self.engine.command("balance 1.1 rcb")
        self.engine.command("neigh_modify every 1 delay 0 check yes")
        self.engine.command('fix 1 all setforce 0.0 0.0 0.0')
        self.engine.command('run 0')
        self.engine.command('unfix 1')

