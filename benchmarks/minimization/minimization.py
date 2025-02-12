""" 
Tests and benchmark minimization
"""
from ase.io import read, write 
from pykmc.system import System 
from ase import Atoms 
import cProfile
from pstats import Stats

#Parameters : 
init_config_file = 'initial_config.xyz'
#init_config_file = '../../examples/Ni_fcc_2047at_monovacancy/initial_config.xyz' 

minimization_params = {'min_style' : 'cg',
                       'etol' : 1.0e-6, 
                       'ftol' : 1.0e-8,
                       'maxiter' : 1000,
                       'maxeval' : 1000}
potential = {'pair_style' : 'eam/alloy', 
             'pair_coeff' : '* * ../../examples/Ni_fcc_499at_monovacancy/Ni_v6_2.0_LKBeland2016.eam Ni'}
nprocs = 8
backend = 'local'

#Initialization 
system = System(config_file = init_config_file) 
traj = [Atoms(positions = system.positions, cell = system.cell, symbols = system.symbols)]
#Minimization 
with cProfile.Profile() as profile : 
    system.minimize('lammps', minimization_params, potential, nprocs = nprocs, backend=backend)
stats = Stats(profile)
print("Profiling minimization : ")
stats.print_stats(0)
traj.append(Atoms(positions = system.positions, cell = system.cell, symbols = system.symbols))
#Save to file before and after minimization
write('traj_minimization.xyz', traj)
