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
import pynauty
from .atomic_environment import make_graph 
import logging

#TODO Parallelization. Depending on nprocs launch searches in parallel
#TODO Change hardcoded upper/lower dE barrier limit selection
#TODO Add graph ID to saddle_position (to check event reconstruction)
#TODO Add Compute k, for the moment k=1 for all event
#TODO Don't understand why inside executor I need to add logging.basicConfig. Otherwise it does not print to log 
#TODO Add different event search style 
#TODO Add option to do the search on a subsystem (--> will be usefull for large systems)
#TODO pARTn commands should not be hardcoded
#TODO rcutevent should not be hardcoded
#TODO What is the value that we should use for the condition delr1 < 0.2 or delr2 < 0.2
#TODO Since now we add the backward reaction to the catalog, it is not needed to check if min1 or min2 is close to the initial configuration and return the corresponding positions
#TODO Better logs 
#TODO See if we can append artn logs, could be usefull while debugging
#TODO Should think of a better way to compute graph, certificate for the backward reaction
#TODO Add logs number of event found/fail
#TODO parameter for backward reaction graph
#TODO find a way to not print in terminal ira errors (we log them)
#FIXME Problem position are wrapped --> pbc problem 

class EventSearch() : 
    """
    Define and run event search procedure    

    Attributes
    ----------
    system : System Object
        the current system
    search_style : str
        style use for the event search, can be 'pARTn', 'dimer'
    search_params : dict 
        parameter needed by the style used to perform the event search
    potential : dict of str: str
        commands to define the potential used by the program defined by minimization_style
    dimension : int, optional
        dimension of the system, by default 3
    nprocs : int, optional
        number of procs available, by default 1
    backend : str, optional
        parameter used by Executorlib, can be 'local', 'slurm_allocation', 'slurm_submission', by default 'local'

    Methods 
    ------- 
    run() 
        run the event search and update the catalog 
    new_environment() 
        find list of environment ID of the current system that are not in the catalog
    pARTn_search(atom_index, potential)
        run an event search using pARTn with atom_index as the central atom
    """

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
        Execute new event searches 
        """

        #Check if new atomic environment that are not in the catalog, if yes extract the environement id: 
        l_new_environement = self.new_environment()
        #For each id in l_new_environment, we will select randomly one atom with the corresponding ID (does this nsearch time)
        for id in l_new_environement : 
            #extract list of atoms in system.environment having the id 
            l_atoms = [dict['atom index'] for dict in self.system.environment if dict['ID'] == id][0] 
            #list of atoms on which we gonna do the search
            self.system.logger.logger.info(':> Launching {} event searches'.format(self.search_params['nsearch']))
            l_atoms_search = [random.choice(l_atoms) for _i in range(self.search_params['nsearch'])]
            #then we do a pART search and put the result of each search in self.system.catalog
            for atom_index in l_atoms_search : 
                #run event search
                with Executor(backend=self.backend, max_cores=self.nprocs) as exe : 
                    fs = exe.submit(self.pARTn_search, atom_index, self.potential )
                if fs.result() is not None :
                    #upper and lower limit : 
                    if fs.result()[3] > 0.1 and fs.result()[3] < 5.0 :  
                        dfevent = pd.Series({'event_id' : id , 
                                    'initial_positions' : fs.result()[0], 
                                    'saddle_positions': fs.result()[1], 
                                    'final_positions': fs.result()[2], 
                                    'energy_barrier': fs.result()[3], 
                                    'k' : 1})

                        self.system.catalog = pd.concat([self.system.catalog, dfevent.to_frame().T], ignore_index=True)
                        #Add reverse event : 
                        #compute finale positions ID : 
                        g = make_graph(self.system, [fs.result()[4]], 3.0, 7.0 )
                        reverse_id = pynauty.certificate(g[0])
                        dfevent = pd.Series({'event_id' : reverse_id, 
                                    'initial_positions' : fs.result()[2], 
                                    'saddle_positions': fs.result()[1], 
                                    'final_positions': fs.result()[1], 
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
            ids_current.remove('crystal') #remove cystalline environment
        except ValueError:
            pass
        l_new_environments = [ids for ids in ids_current if ids not in ids_catalog]
        return l_new_environments 
    
    def pARTn_search(self, atom_index, potential) : 
        """
        Use pARTn with Lammps to find new event

        Parameters
        ----------
        atom_index : int
            index of the central atom on which we perform the event search
        potential : dict of str:str
            commands to define the potential used by the program defined by minimization_style

        Returns
        -------
        (np.array, np.array, np.array, float, int)
            positions of the initial minimum, saddle point, final minimum, the energy barrier and central atom_index
            None if no event have been found
        """
        #Logs
        logging.basicConfig(filename='pykmc.log', filemode='a', level=logging.DEBUG, format='%(message)s')
        self.system.logger.logger.info('> Launching pARTn search')

        #MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()

        #TEST search on subsystem        
        #Create a subsystem base on central atom neighbor lsh z ist 
        #rcutevent = self.system.cell[0][0]
        #rcutevent = 8.0
        #ind = np.linspace(0, self.system.get_global_number_of_atoms()-1, self.system.get_global_number_of_atoms()).astype(int)
        #dist = self.system.get_distances(atom_index, ind, mic=True)
        #neighbor_list = np.where(dist<rcutevent)[0]
        #subsystem = Atoms(positions=self.system.get_positions()[neighbor_list], cell=self.system.get_cell(), pbc=True)
        #atom_index = np.where((subsystem.get_positions() == (self.system.positions[atom_index][0], self.system.positions[atom_index][1], self.system.positions[atom_index][2])).all(axis=1))[0][0]

        #Write lammps data file : 
        lammps_data_file = 'initial_config_minimization.lmp'
        if rank == 0 :
            write_lammps_data(lammps_data_file, self.system, masses=True)
            #write_lammps_data(lammps_data_file, subsystem, masses=True)
            if self.dimension == 2 : 
                modify_lammps_data_2D(lammps_data_file)

        #Setup Lammps : 
        lmp = lammps(comm=comm,cmdargs=['-log', 'log.pARTn.lammps', '-screen', 'none'])
        artn = pypARTn2.artn(engine='lmp')
        lmp.command("units metal")
        lmp.command('atom_style atomic')
        lmp.command("dimension 3")
        lmp.command("boundary p p p")
        lmp.command('atom_modify sort 0 1')
        lmp.command("read_data {}".format(lammps_data_file))
            #Potential : 
        for key, val in potential.items() : 
            lmp.command("{} {}".format(key, val))
        lmp.command("plugin load {}".format(self.search_params['path_artnso']))
        lmp.command("fix 10 all artn dmax 8.0")
        lmp.command("min_style fire")
        #SETUP ARTN
        artn.set('engine_units', 'lammps/metal')
        artn.set('verbose',2)
        artn.set("lpush_final", True)
        artn.set("lmove_nextmin", False) #if true fortran runtime error when event not found
        artn.set("ninit", 2)
        artn.set("forc_thr", 0.01)
        artn.set('push_mode', 'rad')
        artn.set('push_dist_thr', 3.0)
        artn.set("push_step_size",  0.4)
        artn.set("push_ids", [atom_index])
        artn.set('eigen_step_size', 0.2)
        artn.set('lanczos_disp', 0.0005)
        artn.set('nsmooth',  3)
        artn.set('nperp', 5)
        #Run
        lmp.command("minimize 1e-3 1e-3 1000 1000")

        #Need to extract min 1, min 2, saddle positions and energy barrier
        err = artn.get_runparam("error_message")
        if not err : 
            delr1 = artn.extract('delr_min1') 
            delr2 = artn.extract('delr_min2')

            E_sad = artn.extract("etot_sad")
            E_min1 = artn.extract("etot_min1")
            E_min2 = artn.extract("etot_min2")
            dE_forward = E_sad - E_min1 
            dE_backward = E_sad - E_min2 
 
            min1positions = artn.extract("tau_min1")
            min2positions = artn.extract("tau_min2")
            saddlepositions = artn.extract("tau_sad")

            #save only atoms in rcutenv of atom_index
            rcutevent = 7.0
            ind = np.linspace(0, self.system.get_global_number_of_atoms()-1, self.system.get_global_number_of_atoms()).astype(int)
            dist = self.system.get_distances(atom_index, ind, mic=True)
            neighbor_list = np.where(dist<rcutevent)[0]

            #Check if min1 or min2 close to the original configuration
            if delr1 < 0.2 or delr2 < 0.2 :  
                if delr1 < delr2 :  
                    self.system.logger.logger.info('Find one event with dE barrier = {} eV'.format(dE_forward))
                    return min1positions[neighbor_list], saddlepositions[neighbor_list], min2positions[neighbor_list], dE_forward, atom_index 
                else : 
                    #return min2positions, saddlepositions, min1positions, dE_forward
                    self.system.logger.logger.info('Find one event with dE barrier = {} eV'.format(dE_backward))
                    return min2positions[neighbor_list], saddlepositions[neighbor_list], min1positions[neighbor_list], dE_backward, atom_index
            else :
                self.system.logger.logger.error('ERROR: minima too far away from initial configuration')
                return None
        else : 
            self.system.logger.logger.error('ERROR: pARTn error : {} '.format(err))
            return None

#    def dimer_search(self, atom_index, potential): 
#        """ 
#        Use Dimer search with ASE and lammps
#        """
#        # Set up LAMMPS calculator
#        run('export ASE_LAMMPSRUN_COMMAND=/Users/hugomoison/Programmes/lammps-29Aug2024/src', shell=True)
#        lammps_command = ["lmp_mpi"]  
#        lammps_parameters = {'pair_style': 'eam', 'pair_coeff': ['* * ./Ni_v6_2.0_LKBeland2016.eam Ni']}
#        files = ['Ni_v6_2.0_LKBeland2016.eam']
#        #initial potential energy : 
#        lammps_calc = LAMMPS(files=files , lammps_command=lammps_command,**lammps_parameters)
#
#        atoms = Atoms(positions=self.system.positions, cell=self.system.cell, pbc=self.system.pbc)
#        atoms.calc = lammps_calc
#
#        # Calculate the energy of the system with LAMMPS
#        atoms.get_potential_energy()
#        #setup dimer : 
#        dcontrol = DimerControl(logfile='dimer_search.log', initial_eigenmode_method='displacement', displacement_method='vector', displacement_center=atom_index, displacement_radius=4.0)
#        d_atoms = MinModeAtoms(atoms, dcontrol)










