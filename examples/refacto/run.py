from pykmc import System, Engine, Config, NeighborsList, AtomicEnvironment, Catalog
from ase.io import write
from ase import Atoms
import numpy as np

#Config
inputs_path = './input.in'
config = Config.from_file(inputs_path)
config_control = config['Control']

#Initilization
system = System.create_from_file(config_control['config_file']) 
engine = Engine(config) 
new_positions = engine.minimize(system) 
system.update_positions(new_positions)

#Neighbors
nl = NeighborsList(system, config)

#AtomicEnvironment
ae = AtomicEnvironment(config, nl.neighbors_list['rnei'], nl.neighbors_list['rcut'])

#Catalog : 
catalog = Catalog(config) 


