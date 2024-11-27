import random
from lammps import lammps
from mpi4py import MPI 
from .utilities import modify_lammps_data_2D
from ase.io.lammpsdata import write_lammps_data
from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate
from ase.calculators.lammpsrun import LAMMPS
from ase import Atoms
from subprocess import run
#import pypARTn2


class EventSearch() : 

    def __init__(self, system, search_style, search_params, potential, dimension=3, nprocs=1) -> None:
        self.system = system 
        self.search_style = search_style
        self.search_params = search_params 
        self.potential = potential
        self.dimension = dimension 
        self.nprocs = nprocs 


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
            #for atom_index in l_atoms_search : 
            for atom_index in [l_atoms_search[0]] : 
                #self.pARTn_search(atom_index=atom_index)
                self.dimer_search(atom_index=atom_index, potential = self.potential)


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

        #Write lammps data file : 
        lammps_data_file = 'initial_config_minimization.lmp'
        if rank == 0 :
            write_lammps_data(lammps_data_file, self.system, masses=True)
            if self.dimension == 2 : 
                modify_lammps_data_2D(lammps_data_file)

        #Setup pARTn : 
        artn = pypARTn2.artn( engine = "lmp" ) 
        artn.set("engine_units","lammps/metal" )
        artn.set('lpush_final = .true.')
        artn.set('lmove_nextmin = .true.')
        artn.set('ninit = 3')
        artn.set('forc_thr = 0.01')
        artn.set('push_step_size = 0.3')
        artn.set("push_mode = 'list'")
        artn.set('push_ids = {}'.format(atom_index))


        #Setup Lammps : 
        lmp = lammps()
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
        lmp.command("minimize 1e-3 1e-3 1000 1000")

    def dimer_search(self, atom_index, potential): 
        """ 
        Use Dimer search with ASE and lammps
        """
        print('yes')

        # Set up LAMMPS calculator
        run('export ASE_LAMMPSRUN_COMMAND=/Users/hugomoison/Programmes/lammps-29Aug2024/src', shell=True)
        lammps_command = ["lmp"]  
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










