""" 
Tests and benchmark minimization
"""
from ase.io import read, write 
from pykmc.system import System 
import cProfile
from pstats import Stats
import yaml

#Parameters : 
init_config_file = '/root/pyKMC/examples/Ni_fcc_4000at_monovacancy+sia/config_bug.xyz' 

atomicenv_params = {'rnei': 2.8,
                    'rcut'  : 7.0, 
                    'radd_cna' : 0.0}
nprocs = 8
backend = 'local'
style = 'cna/graph'

#Initialization 
system = System(config_file = init_config_file) 
#Minimization 
with cProfile.Profile() as profile : 
    system.find_environment(style, atomicenv_params, nprocs = nprocs, backend=backend)
stats = Stats(profile)
print("Profiling find atomic environment, style={} : ".format(style))
stats.print_stats(0)
#Save atomic environment to file
with open('atomic_environment.yml', 'w') as outfile:
    yaml.dump_all(system.environment, outfile, default_flow_style=False, sort_keys=False)
