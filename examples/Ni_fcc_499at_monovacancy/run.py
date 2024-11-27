from ase.io import read, write
from pykmc.system import System
from pykmc.minimization import Minimization 
from pykmc.atomic_environment import AtomicEnvironment
from ase.visualize import view
import cProfile
import numpy as np
import sys

#PARAMETERS :

init_config_file = './Ni_monovacancy.xyz'

minimization = {'min_style'     : 'cg', 
                'minimize '     : '1.0e-6 1.0e-8 100 1000'}

potential = {'pair_style' : 'eam/alloy', 
             'pair_coeff' : '* * ./Ni_v6_2.0_LKBeland2016.eam Ni'} 

atomenv = {'rnei' : 3.01, 
           'rcut' : 4.0}

search_params = {'nsearch' : 5}

#1-Initialize the system : 
system = System(init_config_file)

#2-Minimize the system : 
system.minimize('lammps', minimization, potential, nprocs=1, backend='slurm_allocation')

#3-find atomic environement
#system.find_environment('cna', atomenv, nprocs=8)
#system.find_environment('graph', atomenv, nprocs=8)
#system.find_environment('cna/graph', atomenv, nprocs=8)
#print(system.environment)

#4-Generate catalog
#system.event_search('dimer', search_params, potential, nprocs=1)