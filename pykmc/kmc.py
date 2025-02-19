import random
from ase import Atoms
import numpy as np
import pandas as pd 
from ase.io import write
from ase.calculators.lammpslib import LAMMPSlib
from .atomic_environment import make_graph
import pynauty
import math as m
import sys

#TODO should extract energy system after minimization
#TODO using our own build in function to write outpufile traj file may be less time consuming than creating an Atoms and use ase.io.write

class KMC() : 
    """class to execute kmc simulation 
         
    Parameters
    ----------
    system : System Object
        the KMC system

    Attributes
    ----------
    system : System Object 
        the KMC system 
    control : dict 
        dictionnary with the 'Control' parameters from the input file
    potentiel : dict 
        dictionnary with the 'Potential' parameters from the input file
    minimization : dict 
        dictionnary with the 'Minimization' parameters from the input file
    atomicenvironment_parameters : dict 
        dictionnary with the 'AtomicEnvironment' parameters from the input file
    event_parameters : dict
        dictionnary with the 'EventSearch' parameters from the input file
    psr_parameters : dict 
        dictionnary with the 'PSR' parameters from the input file
    time : float 
        the current time of the simulation
    """ 
    
    def __init__(self, system) :
        self.system = system
        self.control = self.system.inputs['Control']
        self.potential = self.system.inputs['Potential']
        self.minimization = self.system.inputs['Minimization']
        self.atomicenvironment_parameters = self.system.inputs['AtomicEnvironment']
        self.event_parameters = self.system.inputs['EventSearch']
        self.psr_parameters = self.system.inputs['PSR']
        self.time = None

    def run(self) : 
        """
        Execute nkmc steps
        """

        #=====#
        #Start#
        #=====#
        self.system.logger.logger.info('===========================')
        self.system.logger.logger.info('= Starting KMC simulation =')
        self.system.logger.logger.info('===========================')

        nkmc_steps = self.system.inputs['Control']['nkmc_steps']
        self.time = 0

        #====================#
        #Loop over nkmc_steps#
        #====================#
        for step in range(nkmc_steps) :

            #==============# 
            #Initialization#
            #==============# 
            if step == 0 : 
                self.initial_step_kmc()
            
            #=================================#
            #Update atomic system.environments#
            #=================================#
            self.system.find_environment(self.atomicenvironment_parameters['style'], self.atomicenvironment_parameters)

            #==========================================#
            #Searching Events and update system.catalog#
            #==========================================#
            self.system.event_search(self.event_parameters['style'], self.event_parameters, self.atomicenvironment_parameters, self.potential, self.system.reconstruction, self.control['dimension'], self.control['nprocs'], self.control['backend'])

            #=============================#
            #Check if catalog is not empty#
            #=============================#
            if len(self.system.catalog) > 0 : 
                
                #===============================================#
                #Apply Event (and update time and write to log) #
                #===============================================#
                if self.system.reconstruction : 
                    results = self.apply_event_with_reconstruction() 

                    #write to log
                    if results == 'psr failed' : 
                        self.system.logger.logger.info("WARNING: At step {}, point set registration did not find a match, try to modify PSR parameters".format(step))
                    if results == 'select atom failes' : 
                        self.system.logger.logger.info("WARNING: At step {}, no event found for the current configuration, try to modify EventSearch parameters".format(step))
                    else : 
                       #update time 
                       self.time += results[1] 
                       self.system.logger.table_line_info_kmc(step, self.time, len(self.system.environment), len(self.system.catalog), results[0], self.system.catalog.loc[results[0]].at['energy_barrier'], results[2], results[3], results[4]) 
                       

                else : 
                    idx_cat, delta_t = self.apply_event_without_reconstruction()
                    #update time
                    self.time += delta_t
                    #write to log 
                    self.system.logger.table_line_info_kmc(step, self.time,len(self.system.environment), len(self.system.catalog), idx_cat, self.system.catalog.loc[idx_cat].at['energy_barrier']) 
                    #Detroy catalog#
                    self.system._initialize_catalog()

            else : 
                self.system.logger.logger.info("WARNING: At step {}, emtpy catalog after event searches, try to modify EventSearch parameters".format(step))

            #============#
            #Minimization#
            #============#
            self.system.minimize(self.minimization['style'], self.minimization, self.potential)

            #=======================================#
            #Append new configuration to output file#
            #=======================================#
            write(self.system.kmc_traj, Atoms(self.system.get_chemical_symbols(), 
                                                         positions=self.system.get_positions(), 
                                                         cell = self.system.get_cell(),
                                                         pbc=self.system.get_pbc()), append=True)



    def initial_step_kmc(self) : 
        """
        Procedure done at the start of the first step of a KMC simulation
        """
        #Minimization
        self.system.logger.logger.info(':> Minimization of the system')
        self.system.logger.new_line()
        self.system.minimize(self.minimization['style'], self.minimization, self.potential, self.control['dimension'], self.control['nprocs'], self.control['backend'])

        #First log table line 
        self.system.logger.first_line_table(self.system.reconstruction) 

        #Write first configuration to output file
        write(self.system.kmc_traj, Atoms(self.system.get_chemical_symbols(), 
                                          positions=self.system.get_positions(), 
                                          cell = self.system.get_cell(),
                                          pbc=self.system.get_pbc()), append=True)
        
    def apply_event_without_reconstruction(self) : 
        """ 
        Procedure to apply an event and update system.positions when `reconstruction == False`

        Returns
        ------- 
        idx_cat : int 
            index of the event in the catalog that have been chosen 
        delta_t : float 
            time associated to the event
        """
        #===============#
        #Choose an event#
        #===============#
        idx_cat, delta_t = self.select_event_rejection_free(self.system.reconstruction) 

        #================================#
        #Atom on which we apply the event#
        #================================#
        central_atom_index = self.system.catalog.loc[idx_cat].at['atom_index']

        #==============================================================#
        #update positions -> directly go to final position of the event#
        #==============================================================#
            #final positions
        coords = self.system.catalog.loc[idx_cat].at['final_positions']
            #find neighbors 
        rcutevent = self.event_parameters['rcutenv']
        ind = np.linspace(0, self.system.get_global_number_of_atoms()-1, self.system.get_global_number_of_atoms()).astype(int)
        dist = self.system.get_distances(central_atom_index, ind, mic=True)
        neighbor_list = np.where(dist<rcutevent)[0]
            #change system.positions
        #ugly
        c = 0 
        newpos = self.system.get_positions()
        for i in range(len(newpos)) : 
            if i in neighbor_list : 
                newpos[i] = coords[c]
                c +=1
        self.system.set_positions(newpos)
    
        return idx_cat, delta_t

    def apply_event_with_reconstruction(self) : 
        """
        Procedure to apply an event and update system.positions when `reconstruction == False`

        Returns
        ------- 
        if no event corresponding to the atomic environments return 'select atom failed' 
        if shape matching failed return 'psr failed' 
        else return : 
        idx_cat : int 
            index of the event in the catalog that have been chosen 
        delta_t : float 
            time associated to the event
        dh : float 
            Hausdorff distance value of the match
        reconstruction_de : boolean 
            `True`if energy barrier from the reconstruction match the one in the catalog, else `False`
        recontruction_topo : boolean 
            `True` if the topology at the saddle point after the reconstruction match the on in the catalog, else `False`
        """
        #==================================================================#
        #Updatate system.visited_environment (comes after the event_search)#
        #==================================================================#
        self.update_visited_environment()

        #===============#
        #Choose an event#
        #===============#
        idx_cat, delta_t = self.select_event_rejection_free(self.system.reconstruction)
        
        #================================#
        #Atom on which we apply the event#
        #================================#
        if idx_cat is not None : #In case we have all atomic environments that do not have event
            central_atom_index = self.select_central_atom(idx_cat)
        #Shape Matching
            rmat, tr, perm, dh = self.system.point_set_registration(self.psr_parameters['style'], self.psr_parameters, idx_cat, central_atom_index, self.event_parameters['rcutenv'])
            if rmat is not None : #in case ira did not find a match 
                reconstruction_de, reconstruction_topo = self.update_positions(idx_cat, central_atom_index, rmat, tr, perm, dh)
                return idx_cat, delta_t, dh, reconstruction_de, reconstruction_topo
            else : 
                return "psr failed"
        else : 
            return "select atom failed"



    def update_visited_environment(self) : 
        """
        Update visited environment 
        """
        lids = [d['ID'] for d in self.system.environment]
        self.system.visited_environment.update(set(lids).difference(self.system.visited_environment)) 
    
    def select_event_random(self) : 
        """ 
        return index in system.catalog of a random possible event

        Returns 
        ------- 
        atom index having a topology of an event in the catalog 
        """  
        #find list of event in catalog that have id in system.environment : 
        l_env = [dict['ID'] for dict in self.system.environment]
        l_catalog = [i for i in range(len(self.system.catalog)) if self.system.catalog.loc[i].at['event_id'] in l_env ]
        if len(l_catalog) > 0 : 
            return random.choice(l_catalog)
        else : 
            return None
        
    def select_event_rejection_free(self, reconstruction) : 
        """
        Select an event and return its catalog id based on the rejection free KMC algorithm

        Returns 
        ------- 
        idx_selected_event : int 
            index in the catalog of the selected event 
        delta_t : float 
            time associated to the rate constante of the selected event
        """
        if reconstruction : 
        #1-Find index list of all possible event in the catalog, ie events having IDs that are in the current system.environment
            l_env = [dict['ID'] for dict in self.system.environment]
            if l_env == ['crystal'] : 
                self.system.logger.logger.info('Only atoms with crystalline environment')
                self.close()
            l_catalog = [i for i in range(len(self.system.catalog)) if self.system.catalog.loc[i].at['event_id'] in l_env ]
        else : 
        #1- if reconstruction = False all events are possible : 
            l_catalog = [i for i in range(len(self.system.catalog))]
        #2-Get constant rate of possible events
        k = np.array([self.system.catalog.loc[l_catalog[i]].at['k'] for i in range(len(l_catalog))])
        #3-Compute constant rate cummulative sum
        k_cumulative = [np.sum(k[:i]) for i in range(1,len(k)+1)]
        #4-Get random number [0,1[
        rand1 = random.random() 
        #5-Find event index satisfy ki-1<rand1ktot<ki
        idx_selected_event = np.searchsorted(k_cumulative, rand1*k_cumulative[-1], side = 'left')
        #6-Compute associated delta_t
        delta_t = -m.log(random.random())/k_cumulative[-1]
        return l_catalog[idx_selected_event] ,delta_t

    def select_central_atom(self, idx_cat) : 
        """ 
        Find a central atom with same ID than the event at idx_cat in catalog

        Parameters
        ---------- 
        idx_cat : int 
            index in the catalog of an event

        Returns
        ------- 
        central_atom_index : int 
            index of an atom having the same topology than the event at idx_cat in the catalog
        """ 
        id = self.system.catalog.loc[idx_cat].at['event_id'] 
        atom_index_list = []
        for dic in self.system.environment : 
            if dic['ID'] == id : 
                atom_index_list = dic['atom index']
        #random atom : 
        if len(atom_index_list) > 0 : 
            central_atom_index = random.choice(atom_index_list)
        else : 
            central_atom_index = None
        return central_atom_index
    

 
    def update_positions(self, idx_cat, central_atom_index, rmat, tr, perm, dh) : 
        """_summary_

        Parameters
        ----------
        idx_cat : int
            index in the catalog of an event
        central_atom_index : int
            index of the atom on the system on which we apply the event
        rmat : (3,3) numpy.array of float
            rotation matrix from point set registration 
        tr : (3,) numpy.array of float
            translation matrix from point set registration
        perm : list of int
            permutation matrix from point set registration
        dh : float 
            Hausdorff distance value of the match

        Returns
        -------
        if dh > 0.5 return `None`, `None`
        else return 
            is_energy_saddle_ok : boolean 
                `True`if energy barrier from the reconstruction match the one in the catalog, else `False`
            is_topo_saddle_ok : boolean
                `True` if the topology at the saddle point after the reconstruction match the on in the catalog, else `False`
        """

        if dh < 0.05 :

            #Compute energy of the system : 
            Eini = self.compute_energy_lammps()

            #Save current positions in case reconstruction fail: 
            current_positions = self.system.get_positions()
            
            #Move system to the saddle point : 
                #Coordinates at the saddle point of the event
            coords = np.zeros((len(self.system.catalog.loc[idx_cat].at['saddle_positions']), 3))
                #Apply shape matching to saddle coordinates to match the current system
            coords = self.apply_ira_psr(coords, idx_cat, rmat, tr, perm, 'saddle_positions') 
                #Find neighbors of central_atom
            rcutevent = self.event_parameters['rcutenv']
            ind = np.linspace(0, self.system.get_global_number_of_atoms()-1, self.system.get_global_number_of_atoms()).astype(int)
            dist = self.system.get_distances(central_atom_index, ind, mic=True)
            neighbor_list = np.where(dist<rcutevent)[0]
                #change coordinates of the neighbors to go to the saddle point associated to the event
            move_atomsys_idx = self.change_neighbors_coords(neighbor_list, coords, idx_cat, perm, True)

                #Check reconstruction : 
            is_energy_saddle_ok = self.check_saddle_energy(idx_cat, Eini)
            is_topo_saddle_ok = self.check_saddle_topo(idx_cat, move_atomsys_idx) 
            
            if not is_energy_saddle_ok or not is_topo_saddle_ok : 
                self.system.set_positions(current_positions)
            else : 
                #Coordinates of the final position of the event
                coords = np.zeros((len(self.system.catalog.loc[idx_cat].at['final_positions']),3))
                #Apply shape matching to final coordinates to match the current system
                coords = self.apply_ira_psr(coords, idx_cat, rmat, tr, perm, 'final_positions') 
                #Change coordinates of the neighbors to go to the final positions associated to the event 
                self.change_neighbors_coords(neighbor_list, coords, idx_cat, perm, False)
            return is_energy_saddle_ok, is_topo_saddle_ok

        else : 
            return None, None

    def compute_energy_lammps(self) : 
        """ 
        Use ASE to compute the energy of the system 

        Returns
        ------- 
        Energy of the current system
        """
        #Setup the potential
        cmds = []
        cmds.append('pair_style {}'.format(self.potential['pair_style']))
        cmds.append('pair_coeff {}'.format(self.potential['pair_coeff']))

        #Lammps calculator 
        lammps = LAMMPSlib(lmpcmds=cmds, log_file='log.calc_energy.lammps')
        atoms = Atoms(self.system.get_chemical_symbols(), positions=self.system.get_positions(), cell=self.system.cell, pbc=True) 
        atoms.calc = lammps 

        return atoms.get_potential_energy()

    def apply_ira_psr(self,coords, idx_cat, rmat, tr, perm, which_pos)  : 
        """
        Apply point set registration to coords

        Parameters
        ----------
        coords : (N,3) numpy.array of float
            positions on which we apply the point set registration
        idx_cat : int
            index in the catalog of the event from which we apply the point set registration
        rmat : (3,3) numpy.array of float
            rotation matrix from point set registration 
        tr : (3,) numpy.array of float
            translation matrix from point set registration
        perm : list of int
            permutation matrix from point set registration
        which_pos : str
            should be 'saddle_positions' or 'final_positions' : from which positions of the event we apply the point set registration

        Returns
        -------
        coords : (N,3) numpy.array of float 
            coords after the point set registration
        """            
        for i in range(len(coords)) : 
            coords[i] = np.matmul(rmat, self.system.catalog.loc[idx_cat].at[which_pos][i]) + tr 
        coords[:] = coords[perm]
        return coords
    
    def change_neighbors_coords(self, neighbor_list, coords, idx_cat, perm, find_saddle_atom ) : 
        """
        Change positions of atoms in the system in the neighbors to the central atom of the event by coords

        Parameters
        ----------
        neighbor_list : (N,) numpy.array of int
            atom indices of atoms to which we change coordinates
        coords : (N,3) numpy.array of float
            new coordinates of the neighbor atoms
        idx_cat : int
            index of the event in the catalog
        perm : list of int
            permutation matrix from the point set registration
        find_saddle_atom : boolean
            if we whan to find the atomic index of that atom that move the most

        Returns
        -------
        if `find_saddle_atom == True` 
            move_atomsys_idx : int 
                atomic index of the atom that move the most after the change of coordinates
        """
        c = 0 
        newpos = self.system.get_positions()
        for i in range(len(newpos)) : 
            if i in neighbor_list : 
                if find_saddle_atom : 
                    if perm[c] == self.system.catalog.loc[idx_cat].at['move_atom_idx'] : 
                        move_atomsys_idx = i
                newpos[i] = coords[c]
                c +=1
        self.system.set_positions(newpos)
        if find_saddle_atom : 
            return move_atomsys_idx

    def check_saddle_energy(self, idx_cat, E_ini) : 
        """ 
        Check if the barrier energy is consistent with the one in the catalog after the reconstruction

        Parameters 
        ---------- 
        idx_cat : int 
            index of the event in the catalog 
        E_ini : float 
            energy of the system before going to the saddle point

        Returns
        ------- 
        `True` if the barrier energy is consistent with the one in the catalog after the reconstruction, else `False`
        """
        E_sad = self.compute_energy_lammps()
        dE = E_sad-E_ini
        return abs(dE-self.system.catalog.loc[idx_cat].at['energy_barrier']) < 0.5

    def check_saddle_topo(self, idx_cat, move_atomsys_idx) : 
        """ 
        Check if the topology of the saddle point is consistent with the one in the catalog after the reconstruction 

        Parameters 
        ---------- 
        idx_cat : int 
            index of the event in the catalog 
        move_atomsys_idx : int
            atomic index of the atom at the saddle point

        Returns
        ------- 
        `True` if the topology of the saddle opint is consistent with the one in the catalog after the reconstruction, else `False`

        """
        atoms = Atoms(positions=self.system.get_positions(), cell = self.system.get_cell(), pbc=True)
        g_saddle = make_graph(atoms, [move_atomsys_idx], self.atomicenvironment_parameters['rnei'], self.atomicenvironment_parameters['rcut'])[0]
        topo_saddle = pynauty.certificate(g_saddle)

        check = topo_saddle == self.system.catalog.loc[idx_cat].at['id_saddle']
#        return check 
        return True
    

    def close(self) : 
        self.system.logger.logger.info(':> Quit KMC simulation')
        sys.exit()
