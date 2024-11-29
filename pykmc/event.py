import random
from lammps import lammps
from mpi4py import MPI 
from .utilities import modify_lammps_data_2D
from ase.io.lammpsdata import write_lammps_data
from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate
from ase.calculators.lammpsrun import LAMMPS
from ase import Atoms
from subprocess import run
from executorlib import Executor
import pypARTn2
from scipy.spatial import cKDTree
import numpy as np
import pandas as pd


class EventSearch() : 

    def __init__(self, system, search_style, search_params, potential, dimension, nprocs, backend) -> None:
        self.system = system 
        self.search_style = search_style
        self.search_params = search_params 
        self.potential = potential
        self.dimension = dimension 
        self.nprocs = nprocs 
        self.backend = backend


    def run(self) : 
        """ 
        Execute new event searchs 
        """
        #TODO Parallelization

        #Check if new atomic environment that are not in the catalog, if yes extract the environement id: 
        l_new_environement = self.new_environment()
        #For each id in l_new_environment, we will select randomly one atom with the corresponding ID (does this nsearch time)
        for id in l_new_environement : 
            #extract list of atoms in system.environment having the id 
            l_atoms = [dict['atom index'] for dict in self.system.environment if dict['ID'] == id][0] 
            #list of atoms on which we gonna do the search
            l_atoms_search = [random.choice(l_atoms) for _i in range(self.search_params['nsearch'])]
            #then we do a pART search and put the result of each search in self.system.catalog
            for atom_index in l_atoms_search : 
                with Executor(backend=self.backend, max_cores=self.nprocs) as exe : 
                    fs = exe.submit(self.pARTn_search, atom_index, self.potential )
                if fs.result() is not None : 
                    dfevent = pd.Series({'event_id' : id , 
                                    'initial_positions' : fs.result()[0], 
                                    'saddle_positions': fs.result()[1], 
                                    'final_position': fs.result()[2], 
                                    'energy_barrier': fs.result()[3], 
                                    'k' : 1})

                    self.system.catalog = pd.concat([self.system.catalog, dfevent.to_frame().T], ignore_index=True) 

    def new_environment(self) : 
        """ 
        Return list of atomic environements id of the current step that are not in the catalog
        """
        ids_catalog = self.system.catalog['event_id'].tolist()
        ids_current = [element['ID'] for element in self.system.environment]
        try:
            ids_current.remove('crist')
        except ValueError:
            pass
        l_new_environments = [ids for ids in ids_current if ids not in ids_catalog]
        return l_new_environments 
    
    def pARTn_search(self, atom_index, potential) : 
        """ 
        Use pARTn with Lammps to find events
        atom_index : atom on which we perform the search (the one that we init_push)
        """
        
        #for MPI : 
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()
           #clear 
        run('rm min* sad* initp.xyz latest_eigenvec.xyz artn.in artn.out', shell=True)
        #TEST doing research on subsystem around atom_index, rcutsearch, could be usefull for large system² 
        #tree = cKDTree(self.system.positions, boxsize=np.diag(self.system.cell)) #initialize kd tree 
        #atominenv_idx = tree.query_ball_point(self.system.positions[atom_index], 5.0) #atom in atomic env 
        #atominenv_idx = list(atominenv_idx)
        #newcentralindex = atominenv_idx.index(atom_index)
        ##TODO deal with atoms type
        #subsystem = Atoms(positions = self.system.positions[atominenv_idx], cell=self.system.cell)

        #Write lammps data file : 
        lammps_data_file = 'initial_config_minimization.lmp'
        if rank == 0 :
            write_lammps_data(lammps_data_file, self.system, masses=True)
            #write_lammps_data(lammps_data_file, subsystem, masses=True)
            if self.dimension == 2 : 
                modify_lammps_data_2D(lammps_data_file)


        #Setup pARTn : 
        #if rank == 0 : #write artn.in file 
        #    artninfile = 'artn.in'
        #    file = open(artninfile, 'w')
        #    file.write('&ARTN_PARAMETERS\n')
        #    file.write("engine_units='lammps/metal'\n")
        #    file.write("verbose 1\n")
        #    file.write("lpush_final = .true.\n")
        #    file.write("lmove_nextmin = .true.\n")
        #    file.write("ninit = 1\n")
        #    file.write("forc_thr = 0.01\n")
        #    file.write("push_step_size = 0.2\n")
        #    file.write("push_mode = 'list'\n")
        #    file.write("push_ids = {}\n".format(atom_index))
        #    file.write('nsmooth = 1')
        #    file.close()
        #Setup Lammps : 
        lmp = lammps()
        artn = pypARTn2.artn(engine='lmp')
        lmp.command("units metal")
        lmp.command('atom_style atomic')
        lmp.command("dimension 3")
        lmp.command("boundary p p p")
        lmp.command("read_data {}".format(lammps_data_file))
            #Potential : 
        for key, val in potential.items() : 
            lmp.command("{} {}".format(key, val))
        lmp.command("plugin load {}".format(self.search_params['path_artnso']))
        lmp.command("fix 10 all artn dmax 8.0")
        lmp.command("min_style fire")
        #SETUP ARTN
        artn.set('engine_units', 'lammps/metal')
        artn.set('verbose',1)
        artn.set("lpush_final", True)
        artn.set("lmove_nextmin", False) #if true fortran runtime error when event not found
        artn.set("ninit", 1)
        artn.set("forc_thr", 0.01)
        artn.set("push_step_size",  0.2)
        artn.set("push_mode" ,'list')
        artn.set("push_ids", [atom_index])
        artn.set('nsmooth',  1)
        #Run
        lmp.command("minimize 1e-3 1e-3 1000 1000")
        lmp.close()

        #Need to extract min 1, min 3, saddle positions and energy barrier
        #TODO We only want positions of the rcut environement 
        #TODO Should check if initpositions correspond to min1 or min2
        #TODO Should add backward reaction

         


#        with open('./artn.out', 'r') as output : 
#            lines = output.readlines() 
#        if 'ifail:  1' not in lines[-1]: 
#            delr1 = [e for e in lines if 'DEBRIEF(RLX :1)' in e][0].split()[27]
#            delr2 = [e for e in lines if 'DEBRIEF(RLX :2)' in e][0].split()[27]
#        
#            dE_forward = [e for e in lines if 'forward  E_act' in e][-1].split()[3]
#            dE_backward = [e for e in lines if 'backward E_act' in e][-1].split()[3]
#
#
#            min1positions = np.loadtxt('min1.xyz', skiprows=2, usecols=(1,2,3))
#            min2positions = np.loadtxt('min2.xyz', skiprows=2, usecols=(1,2,3))
#            saddlepositions = np.loadtxt('sad1.xyz', skiprows=2, usecols=(1,2,3))
#
#            #reading artn.out file : 
#            with open('./artn.out', 'r') as output : 
#                lines = output.readlines() 
#            delr1 = [e for e in lines if 'DEBRIEF(RLX :1)' in e][0].split()[27]
#            delr2 = [e for e in lines if 'DEBRIEF(RLX :2)' in e][0].split()[27]
#        
#            dE_forward = [e for e in lines if 'forward  E_act' in e][-1].split()[3]
#            dE_backward = [e for e in lines if 'backward E_act' in e][-1].split()[3]
#
#            if delr1 < delr2 : 
#                return min1positions, saddlepositions, min2positions, dE_forward
#            else : 
#                return min2positions, saddlepositions, min1positions, dE_forward
#        else : 
#            return None
#        #then need to extract configurations, energy barrer
#

    def dimer_search(self, atom_index, potential): 
        """ 
        Use Dimer search with ASE and lammps
        """
        print('yes')

        # Set up LAMMPS calculator
        run('export ASE_LAMMPSRUN_COMMAND=/Users/hugomoison/Programmes/lammps-29Aug2024/src', shell=True)
        lammps_command = ["lmp_mpi"]  
        lammps_parameters = {'pair_style': 'eam', 'pair_coeff': ['* * ./Ni_v6_2.0_LKBeland2016.eam Ni']}
        files = ['Ni_v6_2.0_LKBeland2016.eam']
        #initial potential energy : 
        lammps_calc = LAMMPS(files=files , lammps_command=lammps_command,**lammps_parameters)

        atoms = Atoms(positions=self.system.positions, cell=self.system.cell, pbc=self.system.pbc)
        atoms.calc = lammps_calc

        # Calculate the energy of the system with LAMMPS
        atoms.get_potential_energy()



        #setup dimer : 
        dcontrol = DimerControl(logfile='dimer_search.log', initial_eigenmode_method='displacement', displacement_method='vector', displacement_center=atom_index, displacement_radius=4.0)
        d_atoms = MinModeAtoms(atoms, dcontrol)










