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
    for c in col : 
        positions = catalog.loc[index].at[c]
        atoms = Atoms(positions = positions, cell=[8.0,8.0, 8.0], pbc=True)
        traj.append(atoms) 
    write('event_'+str(index)+'.xyz', traj)

catalog = pd.read_pickle('catalog.pickle') 
for i in range(len(catalog)):
    write_event_traj(catalog, i)
    print('Energy of event {} is {}'.format(i, catalog.loc[i].at['energy_barrier']))


