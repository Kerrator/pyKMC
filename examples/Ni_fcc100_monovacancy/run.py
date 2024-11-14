from ase.io import read, write
from pykmc.minimization import Minimization 
from pykmc.atomic_environment import AtomicEnvironment
from ase.visualize import view

#PARAMETERS : 
#init_config_file = './initial_config.xyz'
init_config_file = '/Users/hugomoison/Postdoc/projet_kmc/examples/Ni_monovacancy/Ni_monovacancy.xyz'

minimization = {'min_style'     : 'cg', 
                'minimize '     : '1.0e-6 1.0e-8 100 1000'}

potential = {'pair_style' : 'eam/alloy', 
             'pair_coeff' : '* * ./Ni_v6_2.0_LKBeland2016.eam Ni'} 

atomenv = {'rnei' : 3.01}

#1-Read initial configuration 
atoms = read(init_config_file)

#2-Run Minimization
#atoms = Minimization(atoms, 'lammps', minimization, potential, dimension=2, nprocs=8).run()

atenv = AtomicEnvironment(atoms, 'cna',atomenv , potential, dimension=3)
atenv.run()
print(atenv.list_env)
