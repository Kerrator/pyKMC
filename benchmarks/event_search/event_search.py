""" 
Tests and benchmark event search
"""
from ase.io import read, write 
from pykmc.system import System 
from ase import Atoms 
import cProfile
from pstats import Stats
import pandas as pd

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
                    'rcut'  : 5.0, 
                    'radd_cna' : 0.0}
search_params = {'nsearch' : 1,
                 'path_artnso' : '/root/programs/artn-plugin/lib/libartn-lmp.so', 
                 'rcutenv' : 7.0, 
                 'emax_event' : 5.0, 
                 'emin_event' : 0.2,
                 'partn_dmax' : 6.0, 
                 'partn_verbose' : 2, 
                 'partn_ninit' : 2, 
                 'partn_forc_thr' : 0.01,
                 'partn_push_mode' : 'rad', 
                 'partn_push_dist_thr' : 3.0, 
                 'partn_push_step_size' : 0.2, 
                 'partn_eigen_step_size' : 0.2, 
                 'partn_lanczos_disp' : 0.0005,
                 'partn_nsmooth' : 3, 
                 'partn_nperp' : 5,
                 'k0' : 1, 
                 'T' : 300.0, 
                 }
nprocs = 8
backend = 'local'
style_atomenv = 'cna'
style_event = 'pARTn'
reconstruction = False

catalog = pd.DataFrame(columns=['event_id', 
                                                     'initial_positions', 
                                                     'saddle_positions', 
                                                     'final_positions', 
                                                     'energy_barrier', 
                                                     'k', 
                                                     'move_atom_idx', 
                                                     'id_saddle'])

#Initialization 
system = System(config_file = init_config_file)
#Minimization : 
system.minimize('lammps', minimization_params, potential, nprocs = nprocs, backend=backend)
#Atomic environment : 
system.find_environment(style_atomenv, atomicenv_params, nprocs = nprocs, backend=backend)
#Event Search
with cProfile.Profile() as profile : 
    system.event_search(style_event, search_params, atomicenv_params,potential, reconstruction = reconstruction, nprocs=nprocs, backend=backend)
stats = Stats(profile)
print("Profiling Event Search with style = : ", style_event)
stats.print_stats(0)
#Print catalog to file 
system.catalog.to_pickle('catalog.pickle')
