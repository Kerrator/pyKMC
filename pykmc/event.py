import random
from lammps import lammps
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
from .config import Parameters
import math as m

#TODO Parallelization. Depending on nprocs launch searches in parallel
#TODO Don't understand why inside executor I need to add logging.basicConfig. Otherwise it does not print to log 
#TODO Add different event search style 
#TODO Add option to do the search on a subsystem (--> will be usefull for large systems)
#TODO What is the value that we should use for the condition delr1 < 0.2 or delr2 < 0.2
#TODO Since now we add the backward reaction to the catalog, it is not needed to check if min1 or min2 is close to the initial configuration and return the corresponding positions
#TODO See if we can append artn logs, could be usefull while debugging
#TODO Should think of a better way to compute graph, certificate for the backward reaction
#TODO find a way to not print in terminal ira errors (we log them)
#TODO Should put loop over nsearch in the serach_ira function to not compute multiple times the same things 
#TODO Don't forget to readd reverse event
#TODO search_without_reconstruction should be already parallized but need to be checked on local(non docker) and slurm
#TODO Check if for reconstruction == False, atol and rtol used in add_event_without_reconstruction() are ok


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

    def __init__(self, system, search_style, search_params, atomenv_params, potential, reconstruction, dimension, nprocs, backend) -> None:
        self.system = system 
        self.search_style = search_style
        self.search_params = search_params 
        self.atomenv_params = atomenv_params
        self.potential = potential
        self.reconstruction = reconstruction
        self.dimension = dimension 
        self.nprocs = nprocs 
        self.backend = backend


    def run(self) : 
        """ 
        Execute new event searches 
        """

        #Check if we want reconstruction or not : 
        match self.reconstruction : 
            case True : 
                self.search_with_reconstruction()
            case False : 
                self.search_without_reconstruction()
            case _: 
                raise Exception("Wrong reconstruction value in 'Control', must be True or False")


    def search_with_reconstruction(self) : 
        
        #TEMPORARY FIX write config file here so no problem parallelization
        lammps_data_file = 'initial_config_minimization.lmp'
        write_lammps_data(lammps_data_file, self.system, masses=True)

        #Check if new atomic environment that are not in the catalog, if yes extract the environement id: 
        l_new_environement = self.new_environment()
        #For each id in l_new_environment, we will select randomly one atom with the corresponding ID (does this nsearch time)
        #TODO Remplacer par executor list fonction

        l_atoms = []
        for id in l_new_environement :
            #list of atoms that have id in l_new_environment : 
            atom_idx =  [dict['atom index'] for dict in self.system.environment if dict['ID'] == id][0]
            #We select nsearch atoms randomly in this atom_idx 
            atom_idx = [random.choice(atom_idx) for _i in range(self.search_params['nsearch'])]
            #extend total list of atoms on which we gonna do an event search
            l_atoms.extend(atom_idx)

        #For each atoms in atom_idx we do an event search 
        with Executor(backend=self.backend, max_workers=self.nprocs) as exe : 
            l_fs = [exe.submit(self.pARTn_search, atom_index, resource_dict={"cores" : 1}) for atom_index in l_atoms]
        #For each results, we add the event to the catalog 
        
        for fs in l_fs : 
            if fs.result() is not None : 
                energy_barrier = fs.result()[3] 
                if self.search_params['emin_event'] < energy_barrier < self.search_params['emax_event'] : 
                    dfevent = pd.Series({'event_id' : fs.result()[4] , 
                                        'initial_positions' : fs.result()[0], 
                                        'saddle_positions': fs.result()[1], 
                                        'final_positions': fs.result()[2], 
                                        'energy_barrier': fs.result()[3], 
                                        'k' : self.compute_rate_Eyring(fs.result()[3]), 
                                        'id_saddle' : fs.result()[5], 
                                        'id_final': fs.result()[6], 
                                        'move_atom_idx': fs.result()[7]})
                    self.system.catalog = pd.concat([self.system.catalog, dfevent.to_frame().T], ignore_index=True)







    #    for id in l_new_environement : 
    #        #extract list of atoms in system.environment having the id 
    #        l_atoms = [dict['atom index'] for dict in self.system.environment if dict['ID'] == id][0] 
    #        #list of atoms on which we gonna do the search
    #        #self.system.logger.logger.info(':> Launching {} event searches'.format(self.search_params['nsearch']))
    #        l_atoms_search = [random.choice(l_atoms) for _i in range(self.search_params['nsearch'])]
    #        #then we do a pART search and put the result of each search in self.system.catalog

    #        for atom_index in l_atoms_search : 
    #            #run event search
    #            with Executor(backend=self.backend, max_cores=self.nprocs) as exe : 
    #                fs = exe.submit(self.pARTn_search, atom_index )
    #                if fs.result() is not None :
    #                    #upper and lower limit : 
    #                    if fs.result()[3] > self.search_params['emin_event'] and fs.result()[3] < self.search_params['emax_event'] : 
    #                        dfevent = pd.Series({'event_id' : id , 
    #                                    'initial_positions' : fs.result()[0], 
    #                                    'saddle_positions': fs.result()[1], 
    #                                    'final_positions': fs.result()[2], 
    #                                    'energy_barrier': fs.result()[3], 
    #                                    'k' : self.compute_rate_Eyring(fs.result()[3]), 
    #                                    'move_atom_idx' : fs.result()[4],
    #                                    'id_saddle' : fs.result()[5]})

    #                        self.system.catalog = pd.concat([self.system.catalog, dfevent.to_frame().T], ignore_index=True)


    #                        #Add reverse event : 
                            #compute finale positions ID : 
                            #g = make_graph(self.system, [fs.result()[4]], 3.0, 5.0 )
                            #reverse_id = pynauty.certificate(g[0])
                            #dfevent = pd.Series({'event_id' : id, 
                            #            'initial_positions' : fs.result()[2], 
                            #            'saddle_positions': fs.result()[1], 
                            #            'final_positions': fs.result()[0], 
                            #            'energy_barrier': fs.result()[3], 
                            #            'k' : 1, 
                            #            'central_atom' : fs.result()[4], 
                            #            'from_id' : id})
                            #self.system.catalog = pd.concat([self.system.catalog, dfevent.to_frame().T], ignore_index=True)

    def search_without_reconstruction(self) : 
        """ 
        Search when reconstruction input is set to False.
        Made to be use with atomic environment style = 'cna' 
        For each non cristalline atom we launch nsearch event search
        """
        #List of atoms that have non cristalline environement 
        l_atoms = [dict['atom index'] for dict in self.system.environment if dict['ID'] == 'noncrystal'][0]
        #for each atom in l_atoms we launch nsearch event searches 
        l_atoms *= self.search_params['nsearch']

        #TEMPORARY FIX write config file here so no problem parallelization
        lammps_data_file = 'initial_config_minimization.lmp'
        write_lammps_data(lammps_data_file, self.system, masses=True)

        with Executor(backend=self.backend, max_workers=self.nprocs) as exe : 
            l_fs = [exe.submit(self.pARTn_search, atom_index, resource_dict={"cores" : 1}) for atom_index in l_atoms]
        #Loop over list results and add event to the catalog : 
        for i,fs in enumerate(l_fs) : 
            if fs.result() is not None : 
                energy_barrier = fs.result()[3] 
                if self.search_params['emin_event'] < energy_barrier < self.search_params['emax_event'] : 
                    dfevent = pd.Series({'atom_index' : l_atoms[i] , 
                                         'final_positions' : fs.result()[2], 
                                         'energy_barrier' : fs.result()[3],
                                         'k' : self.compute_rate_Eyring(fs.result()[3])})
                    #Check if event already in catalog : 
                    if len(self.system.catalog) > 0 : 
                        self.add_event_without_reconstruction(dfevent)
                    else : 
                        self.system.catalog = pd.concat([self.system.catalog, dfevent.to_frame().T], ignore_index=True)


    def new_environment(self) : 
        """ 
        Return list of atomic environements id of the current step that are not in the catalog
        and not have been visited
        """
        ids_catalog = self.system.catalog['event_id'].tolist()
        ids_current = [element['ID'] for element in self.system.environment]
        try:
            ids_current.remove('crystal') #remove cystalline environment
        except ValueError:
            pass
        l_new_environments = [ids for ids in ids_current if ids not in ids_catalog]
        #remove ids if in visited environment 
        l_new_environments = list(set(l_new_environments).difference(self.system.visited_environment))
        #self.system.logger.logger.info('> Found {} new environments'.format(len(l_new_environments)))
        return l_new_environments 
    
    def compute_rate_Eyring(self, dE) : 
        """ 
        Compute the rate constant based on eq 11 of https://www.frontiersin.org/journals/chemistry/articles/10.3389/fchem.2019.00202/full 
        """
        p = Parameters() 
        T = self.search_params['T'] 
        k0 = self.search_params['k0'] 
        return k0*((p.kb*T)/p.h)*m.exp(-dE/(p.kb*T))
    
    def add_event_without_reconstruction(self, dfevent) : 
        """ 
        Search if event is already in the catalog, if not, add the event to the catalog
        """
        atol = 1e-3 
        rtol = 1e-3 

        #Only select rows with same atom index 
        subset = self.system.catalog[self.system.catalog["atom_index"] == dfevent['atom_index']]

        #Check if we have final positions of the event close to at least one final positions in the subset 
        if not subset["final_positions"].apply(lambda pos : np.allclose(pos, dfevent["final_positions"], atol=atol, rtol=rtol)).any() : 
            #if not add event to the catalog : 
            self.system.catalog = pd.concat([self.system.catalog, dfevent.to_frame().T], ignore_index=True)
            


    def pARTn_search(self, atom_index) : 
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
        #self.system.logger.logger.info('> Launching pARTn search')

        from mpi4py import MPI 
        #MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()

        #TEST put atom_index at the center of the cell to fix pbc problems : 
        #New Atoms
        #cell = self.system.get_cell()
        #positions = self.system.get_positions()
        #atoms = Atoms(symbols=self.system.get_chemical_symbols(), positions = positions, cell = cell, pbc=True)
        #if self.reconstruction : 
        #    ax, ay, az = cell[0][0], cell[1][1], cell[2][2]
        #    dx, dy, dz = ax/2 - positions[atom_index][0], ay/2 - positions[atom_index][1], az/2 - positions[atom_index][2]
        #    atoms.translate(np.array([dx, dy, dz]))

        #Write lammps data file : 
        lammps_data_file = 'initial_config_minimization.lmp'
        #if rank == 0 :
            #write_lammps_data(lammps_data_file, self.system, masses=True)
            #write_lammps_data(lammps_data_file, atoms, masses=True)
            #write_lammps_data(lammps_data_file, subsystem, masses=True)
            #if self.dimension == 2 : 
            #    modify_lammps_data_2D(lammps_data_file)
        #Setup Lammps : 
        lmp = lammps(comm=comm,cmdargs=['-screen', 'none'])
        artn = pypARTn2.artn(engine='lmp')
        lmp.command("units metal")
        lmp.command('atom_style atomic')
        lmp.command("dimension 3")
        lmp.command("boundary p p p")
        lmp.command("read_data {}".format(lammps_data_file))
        lmp.command('atom_modify sort 40000000 0.0')
            #Potential : 
        lmp.command('pair_style {}'.format(self.potential['pair_style']))
        lmp.command('pair_coeff {}'.format(self.potential['pair_coeff']))
        #for key, val in potential.items() : 
        #    lmp.command("{} {}".format(key, val))
        lmp.command("plugin load {}".format(self.search_params['path_artnso']))
        lmp.command("fix 10 all artn dmax {}".format(self.search_params['partn_dmax']))
        lmp.command("min_style fire")
        #SETUP ARTN
        artn.set('engine_units', 'lammps/metal')
        artn.set('verbose',self.search_params['partn_verbose'])
        artn.set('struc_format_out', 'none')
        artn.set("lpush_final", True)
        artn.set("lmove_nextmin", False) #if true fortran runtime error when event not found
        artn.set("ninit", self.search_params['partn_ninit'])
        artn.set("forc_thr", self.search_params['partn_forc_thr'])
        artn.set('push_mode', self.search_params['partn_push_mode'])
        if self.search_params['partn_push_mode'] == 'rad' : 
            artn.set('push_dist_thr', self.search_params['partn_push_dist_thr'])
        artn.set("push_step_size",  self.search_params['partn_push_step_size'])
        artn.set("push_ids", [atom_index])
        artn.set('eigen_step_size', self.search_params['partn_eigen_step_size'])
        artn.set('lanczos_disp', self.search_params['partn_lanczos_disp'])
        artn.set('nsmooth',  self.search_params['partn_nsmooth'])
        artn.set('nperp', self.search_params['partn_nperp'])
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
            


            #eigenvec = artn.extract("eigen_sad")
            #print(eigenvec)
            

            #TEST Find Atoms that move the most  : 
            #dist = (min1positions-saddlepositions)**2
            #dist = dist.sum(axis=-1)
            #dist = np.sqrt(dist)
            #index_move = np.argmax(dist)

            rcutevent = self.search_params['rcutenv']
            ind = np.linspace(0, self.system.get_global_number_of_atoms()-1, self.system.get_global_number_of_atoms()).astype(int)

            if self.reconstruction : 
                cell = self.system.get_cell()
                #put event in event_traj en translate move atom at the center to prevent pbc problems
                ppositions = [min1positions, saddlepositions, min2positions]
                event_traj = [] 
                ax, ay, az = cell[0][0], cell[1][1], cell[2][2]
                #put event in traj_event
                for pp in ppositions : 
                    atoms = Atoms(positions=pp, cell=self.system.get_cell(), pbc=True)
#                    atoms.set_positions(atoms.get_positions(wrap=True))
                    event_traj.append(atoms)
                #find atom that move the most
                    #prevent lammps cell travel 
                    #neighbor of central atom
                dist = self.system.get_distances(atom_index, ind, mic=True)
                neighbor_list = np.where(dist<rcutevent)[0]
                min1positions_cent = min1positions[neighbor_list]
                saddlepositions_cent = saddlepositions[neighbor_list]
                min2positions_cent = min2positions[neighbor_list]
                if delr1 < delr2 : 
                    dist = (min1positions_cent-saddlepositions_cent)**2
                else : 
                    dist = (min2positions_cent-saddlepositions_cent)**2
                dist = dist.sum(axis=-1)
                dist = np.sqrt(dist)
                local_index_move = np.argmax(dist)
                index_move = neighbor_list[local_index_move]
                #translate atoms  : 
                if delr1 < delr2 : 
                    dx, dy, dz = ax/2 - event_traj[0].get_positions()[index_move][0], ay/2 - event_traj[0].get_positions()[index_move][1], az/2 - event_traj[0].get_positions()[index_move][2]
                else : 
                    dx, dy, dz = ax/2 - event_traj[2].get_positions()[index_move][0], ay/2 - event_traj[2].get_positions()[index_move][1], az/2 - event_traj[2].get_positions()[index_move][2]
                for i in range(len(event_traj)) : 
                    event_traj[i].translate(np.array([dx, dy, dz]))
                #Compute graph topo, used to check if event already in catalog : 
                id_event = pynauty.certificate(make_graph(event_traj[0], [index_move], self.atomenv_params['rnei'], self.atomenv_params['rcut'])[0])
                id_saddle = pynauty.certificate(make_graph(event_traj[1], [index_move], self.atomenv_params['rnei'], self.atomenv_params['rcut'])[0])
                id_final = pynauty.certificate(make_graph(event_traj[2], [index_move], self.atomenv_params['rnei'], self.atomenv_params['rcut'])[0])

                #find neighbor atom move : 
                if delr1 < delr2 : 
                    dist = event_traj[0].get_distances(index_move, ind, mic=True)
                else : 
                    dist = event_traj[2].get_distances(index_move, ind, mic=True)
                neighbor_list = np.where(dist<rcutevent)[0]
                min1positions = min1positions[neighbor_list]
                min2positions = min2positions[neighbor_list]
                saddlepositions = saddlepositions[neighbor_list]


            else :  
                #graph saddle point index_move 
#                tmp_pos = saddlepositions 
#                tmp_pos[tmp_pos < 0] = 0
#                atoms_saddle = Atoms(positions=tmp_pos, cell=self.system.get_cell(), pbc=True)
#                g_saddle = make_graph(atoms_saddle, [index_move], self.atomenv_params['rnei'], self.atomenv_params['rcut'])[0]
#                id_saddle = pynauty.certificate(g_saddle)
                dist = self.system.get_distances(atom_index, ind, mic=True)
                neighbor_list = np.where(dist<rcutevent)[0]


                min1positions = min1positions[neighbor_list]
                min2positions = min2positions[neighbor_list]
                saddlepositions = saddlepositions[neighbor_list]

            
            

            #save only atom in rcutenv of atom_index

  #          if self.reconstruction : 
  #              dist = self.system.get_distances(index_move, ind, mic=True)
  #          else : 
  #              dist = self.system.get_distances(atom_index, ind, mic=True)

  #          neighbor_list = np.where(dist<rcutevent)[0]


  #          min1positions = min1positions[neighbor_list]
  #          min2positions = min2positions[neighbor_list]
  #          saddlepositions = saddlepositions[neighbor_list]

            #TEST Find Atoms that move the most == > for graph topo saddle point : 
            #dist = (min1positions-saddlepositions)**2
            #dist = dist.sum(axis=-1)
            #dist = np.sqrt(dist)
            #index_move = np.argmax(dist)
            

            #Check if min1 or min2 close to the original configuration
            if delr1 < 0.2 or delr2 < 0.2 : 
                #if len(neighbor_list1) == len(neighbor_list2) and len(neighbor_list1) == len(neighbor_list3)  :
                    if delr1 < delr2 :
                        #self.system.logger.logger.info('Find one event with dE barrier = {} eV'.format(dE_forward))
                        #return min1positions, saddlepositions, min2positions, dE_forward, np.where(neighbor_list == atom_index)[0][0], atom_index 
                        #return min1positions, saddlepositions, min2positions, dE_forward, index_move, atom_index, index_move_prev
                        #return min1positions, saddlepositions, min2positions, dE_forward, index_move, id_saddle
                        if self.reconstruction : 
                            return min1positions, saddlepositions, min2positions, dE_forward, id_event, id_saddle, id_final, local_index_move
                        else : 
                            return min1positions, saddlepositions, min2positions, dE_forward
                        #return min1positions[neighbor_list], saddlepositions[neighbor_list], min2positions[neighbor_list], dE_forward, atom_index 
                    else : 
                        #return min2positions, saddlepositions, min1positions, dE_forward
                        #self.system.logger.logger.info('Find one event with dE barrier = {} eV'.format(dE_backward))
                        #return min2positions, saddlepositions, min1positions, dE_backward, np.where(neighbor_list == atom_index)[0][0], atom_index
                        #return min2positions, saddlepositions, min1positions, dE_backward, index_move, atom_index, index_move_prev
                        #return min2positions, saddlepositions, min1positions, dE_backward, index_move, id_saddle
                        if self.reconstruction : 
                            return min2positions, saddlepositions, min1positions, dE_backward, id_event, id_saddle, id_final, local_index_move
                        else : 
                            return min2positions, saddlepositions, min1positions, dE_backward
                        #return min2positions[neighbor_list], saddlepositions[neighbor_list], min1positions[neighbor_list], dE_backward, atom_index
                #else : 
                    #print('len not consistent')
            else :
                #self.system.logger.logger.error('ERROR: minima too far away from initial configuration')
                return None
        else : 
            #self.system.logger.logger.error('ERROR: pARTn error : {} '.format(err))
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










