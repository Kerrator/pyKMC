from ase.io import read, write
from pykmc.minimization import Minimization 
from pykmc.atomic_environment import AtomicEnvironment
from ase.visualize import view
import cProfile

#PARAMETERS : 
#init_config_file = './initial_config.xyz'
#init_config_file = '/Users/hugomoison/Postdoc/projet_kmc/examples/Ni_monovacancy/Ni_monovacancy.xyz'
#init_config_file = '/Users/hugomoison/Postdoc/Vault/Ni_cystal_with_defect/Ni_cristal_4000at/Ni_cristal_4000at_4defects.xyz'
init_config_file = '/Users/hugomoison/Postdoc/Vault/Ni_cystal_with_defect/Ni_cristal_10976at/Ni_cristal_10976at_3defects.xyz'

minimization = {'min_style'     : 'cg', 
                'minimize '     : '1.0e-6 1.0e-8 100 1000'}

potential = {'pair_style' : 'eam/alloy', 
             'pair_coeff' : '* * ./Ni_v6_2.0_LKBeland2016.eam Ni'} 

atomenv = {'rnei' : 3.00, 
           'rcut' : 3.00}

#1-Read initial configuration 
atoms = read(init_config_file)
atoms.set_pbc(True)

#2-Run Minimization
#atoms = Minimization(atoms, 'lammps', minimization, potential, dimension=2, nprocs=8).run()

#atenv = AtomicEnvironment(atoms, 'cna',atomenv , potential, dimension=3)
#atenv = AtomicEnvironment(atoms, 'hausdorff_dist',atomenv , potential, dimension=3)
atenv = AtomicEnvironment(atoms, 'graph_nauty',atomenv, dimension=3, nprocs=1)
atenv.run()
#cProfile.run('atenv.run()')
#z = 499*[1]
#for i in range(len(atenv.dict_env)) : 
#    list_at = atenv.dict_env[i]['atom index'] 
#    for a in list_at : 
#        z[a] = i+35
#
#atoms.set_atomic_numbers(z)
#view(atoms)
##
print(atenv.dict_env)
#