import pandas as pd 
from ase import Atoms 
from ase.io import write
import numpy as np

def write_event_traj(catalog,index) : 
    """ 
    write initial positions, saddle positions and final positions to a file
    """ 
    traj = [] 
    col = ['initial_positions', 'saddle_positions', 'final_positions'] 
    move_atom_idx = catalog.loc[index].at['move_atom_idx']
    for c in col : 
        positions = catalog.loc[index].at[c]
        atoms = Atoms(positions = positions)
        z = atoms.get_global_number_of_atoms()*[30]
        z[move_atom_idx] = 35
        atoms.set_atomic_numbers(z)
        traj.append(atoms) 

    write('event_'+str(index)+'.xyz', traj)

catalog = pd.read_pickle('catalog.pickle') 
for i in range(len(catalog)):
    write_event_traj(catalog, i)
    print('Energy of event {} is {}'.format(i, catalog.loc[i].at['energy_barrier']))


