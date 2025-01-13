import pandas as pd 
from ase import Atoms 
from ase.io import write
import numpy as np
from ast import literal_eval

#def write_event_traj(catalog,index) : 
#    """ 
#    write initial positions, saddle positions and final positions to a file
#    """ 
#    traj = [] 
#    col = ['initial_positions', 'saddle_positions', 'final_positions'] 
#    for c in col : 
#        positions = catalog.loc[index].at[c]
#        print(positions)
#        #atoms = Atoms(positions = positions, cell=[17.6,17.6, 17.6], pbc=True)
#        atoms = Atoms(positions = positions)
#        traj.append(atoms) 
#
#    write('event_'+str(index)+'.xyz', traj)
#
#catalog = pd.read_pickle('catalog.pickle') 
#for i in range(len(catalog)):
#    write_event_traj(catalog, i)
#    print('Energy of event {} is {}'.format(i, catalog.loc[i].at['energy_barrier']))

catalog = pd.read_csv('catalog.csv') 
positions = catalog.loc[0].at['initial_positions']
positions = positions.to_list()
print(positions)
