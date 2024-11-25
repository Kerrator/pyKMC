from ase.io import read, write
from pykmc.system import System
from pykmc.minimization import Minimization 
from pykmc.atomic_environment import AtomicEnvironment
from ase.visualize import view
import cProfile
import numpy as np

#PARAMETERS : 
#init_config_file = './initial_config.xyz'
init_config_file = '/Users/hugomoison/Postdoc/projet_kmc/examples/Ni_monovacancy/Ni_monovacancy.xyz'
#init_config_file = '/Users/hugomoison/Postdoc/Vault/Ni_cystal_with_defect/Ni_cristal_108at/Ni_cristal_108at.xyz'
#init_config_file = '/Users/hugomoison/Postdoc/Vault/Ni_cystal_with_defect/Ni_cristal_4000at/Ni_cristal_4000at_4defects.xyz'
#init_config_file = '/Users/hugomoison/Postdoc/Vault/Ni_cystal_with_defect/Ni_cristal_10976at/Ni_cristal_10976at_3defects.xyz'
#init_config_file = '/Users/hugomoison/Postdoc/Vault/Ni_cystal_with_defect/Ni_cristal_32000at_50defects/Ni_cristal_32000at_50defects_10defects.xyz'
#init_config_file = '/Users/hugomoison/Postdoc/Vault/Ni_cystal_with_defect/Ni_cristal_108000at_1defects/Ni_cristal_108000at_1defects_1defects.xyz'


minimization = {'min_style'     : 'cg', 
                'minimize '     : '1.0e-6 1.0e-8 100 1000'}

potential = {'pair_style' : 'eam/alloy', 
             'pair_coeff' : '* * ./Ni_v6_2.0_LKBeland2016.eam Ni'} 

atomenv = {'rnei' : 3.01, 
           'rcut' : 5.5}

#1-Initialize the system : 
system = System(init_config_file)

np.savetxt('test0', system.get_positions())

system.minimize('lammps', minimization, potential)

np.savetxt('test1', system.get_positions())

#atoms = read(init_config_file)
#atoms.set_pbc(True)

#2-Run Minimization
#atoms = Minimization(atoms, 'lammps', minimization, potential, dimension=2, nprocs=8).run()

#atenv = AtomicEnvironment(atoms, 'cna',atomenv , potential, dimension=3, nprocs=2)
#atenv = AtomicEnvironment(atoms, 'hausdorff_dist',atomenv , potential, dimension=3)
#atenv = AtomicEnvironment(atoms, 'graph_nauty',atomenv, dimension=3, nprocs=8)
#atenv = AtomicEnvironment(atoms, 'cna/graph_nauty',atomenv, dimension=3, nprocs=1)
#atenv.run()
#cProfile.run('atenv.run()')
#z = 499*[1]
#for i in range(len(atenv.dict_env)) : 
#    list_at = atenv.dict_env[i]['atom index'] 
#    for a in list_at : 
#        z[a] = i+35
##
#atoms.set_atomic_numbers(z)
#view(atoms)
###
#print(atenv.dict_env)
#