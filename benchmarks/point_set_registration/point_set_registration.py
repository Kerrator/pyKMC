""" 
Tests and benchmark point set registration
"""
from ase.io import read, write 
from pykmc.system import System 
from ase import Atoms 
import cProfile
from pstats import Stats

#Parameters : 
init_config_file = '../../examples/Ni_fcc_2047at_monovacancy/initial_config.xyz' 


minimization_params = {'min_style' : 'cg',
                       'etol' : 1.0e-6, 
                       'ftol' : 1.0e-8,
                       'maxiter' : 1000,
                       'maxeval' : 1000}

potential = {'pair_style' : 'eam/alloy', 
             'pair_coeff' : '* * ../../examples/Ni_fcc_499at_monovacancy/Ni_v6_2.0_LKBeland2016.eam Ni'}
atomicenv_params = {'rnei': 3.01,
                    'rcut'  : 6.0, 
                    'radd_cna': 0.0}

psr_parameters = {'kmax_factor' : 2.0}
 
nprocs = 1
backend = 'local'
style_atomenv = 'cna/graph'

style_psr = 'ira'

#Initialization 
system = System(config_file=init_config_file, catalog_path='catalog.pickle')
#Minimization 
system.minimize('lammps', minimization_params, potential, nprocs = nprocs, backend=backend)
#Atomic environment
system.find_environment(style_atomenv, atomicenv_params, nprocs = nprocs, backend=backend)
#PSR
with cProfile.Profile() as profile : 
    system.point_set_registration(style_psr, psr_parameters, 0, 1024, 7.0, nprocs=nprocs, backend=backend)
stats = Stats(profile)
print("Profiling psr = : ", style_psr)
stats.print_stats(0)
