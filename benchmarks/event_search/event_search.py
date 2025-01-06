""" 
Tests and benchmark event search
"""
from ase.io import read, write 
from pykmc.system import System 
from ase import Atoms 
import cProfile
from pstats import Stats

#Parameters : 
init_config_file = '../../examples/Ni_fcc_499at_monovacancy/Ni_monovacancy.xyz' 

minimization_params = {'min_style' : 'cg',
                       'minimize'  : '1.0e-6 1.0e-8 100 1000'} 
potential = {'pair_style' : 'eam/alloy', 
             'pair_coeff' : '* * ../../examples/Ni_fcc_499at_monovacancy/Ni_v6_2.0_LKBeland2016.eam Ni'}
atomicenv_params = {'rnei': 3.01,
                    'rcut'  : 6.0}
search_params = {'nsearch' : 10, 
                 'path_artnso' : '/root/programs/artn-plugin/lib/libartn-lmp.so'}
 
nprocs = 1
backend = 'local'
style_atomenv = 'cna/graph'
style_event = 'pARTn'

#Initialization 
system = System(init_config_file) 
#Minimization : 
system.minimize('lammps', minimization_params, potential, nprocs = nprocs, backend=backend)
#Atomic environment : 
with cProfile.Profile() as profile : 
    system.find_environment(style_atomenv, atomicenv_params, nprocs = nprocs, backend=backend)
#Event Search
with cProfile.Profile() as profile : 
    system.event_search(style_event, search_params, potential, nprocs=nprocs, backend=backend)
stats = Stats(profile)
print("Profiling Event Search with style = : ", style_event)
stats.print_stats(0)
#Print catalog to file 
system.catalog.to_pickle('catalog.pickle')
