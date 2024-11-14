from ase.io import read, write
from pykmc.minimization import Minimization 
from ase.visualize import view

#PARAMETERS : 
init_config_file = './initial_config.xyz'

minimization = {'min_style'     : 'cg', 
                'minimize '     : '1.0e-6 1.0e-8 100 1000'}

potential = {'pair_style' : 'eam/alloy', 
             'pair_coeff' : '* * ./Ni_v6_2.0_LKBeland2016.eam Ni'} 

#1-Read initial configuration 
atoms = read(init_config_file)

#2-Run Minimization
atoms = Minimization(atoms, 'lammps', minimization, potential, dimension=3, nprocs=8).run()
