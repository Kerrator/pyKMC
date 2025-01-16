import sys
from ase.io import read, write
from pykmc.system import System
from pykmc.minimization import Minimization 
from pykmc.atomic_environment import AtomicEnvironment
from ase.visualize import view
import cProfile
import numpy as np


#PARAMETERS :

init_config_file = './initial_config.xyz'

minimization = {'min_style'     : 'cg', 
                'minimize '     : '1.0e-6 1.0e-8 100 1000'}

potential = {'pair_style' : 'eam/alloy', 
             'pair_coeff' : '* * ./Ni_v6_2.0_LKBeland2016.eam Ni'} 

atomenv = {'rnei' : 3.01, 
           'rcut' : 5.0, 
           'radd_cna' : 0.0}

search_params = {'nsearch' : 10, 
                 'path_artnso' : '/root/programs/artn-plugin/lib/libartn-lmp.so'}
                # 'path_artnso' :'/home/hmoison/programs/artn-plugin/lib/libartn-lmp.so' }

kmc_parameters = {'nkmc_steps' : 100}
#1-Initialize the system : 
system = System(init_config_file, catalog='catalog.pickle')
#system = System(init_config_file)
#KMC 
system.kmc(kmc_parameters, minimization,atomenv, search_params, potential )

write('kmc_traj.xsf', system.kmc_traj)
system.catalog.to_pickle('catalog.pickle')
#system.catalog.to_csv('catalog.csv', index=False)
